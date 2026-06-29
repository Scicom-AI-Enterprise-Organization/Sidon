#!/usr/bin/env python3
"""Sidon call-centre restoration — inference.

Restores telephony / call-centre audio (narrowband, codec'd, noisy) to clean
48 kHz, using the two trained stages:

  input audio --resample 16k--> (call-centre FE: 24L w2v-BERT + LoRA) --features[T,1024]-->
      (DAC decoder, 188M) --> 48 kHz waveform

The FE LoRA adapter is **merged into the base weights** here (W_eff = W + (alpha/r)*B@A,
bias = trained bias), so inference needs NO `peft` — just transformers + descript-audio-codec.

Usage:
  python runpod/infer_callcentre.py --input audio --out-dir audio/out \
      --fe-adapter checkpoints/fe_adapter_full.pt --decoder checkpoints/decoder_only.pt

Stereo inputs (e.g. agent/customer on separate channels) are restored per-channel
and recombined to stereo. A `<name>_orig48k.wav` (naive-upsampled input, no model)
is also written for an apples-to-apples A/B listen.
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "4")

import argparse
import glob
import time

import numpy as np
import soundfile as sf
import torch
import torchaudio
from transformers import AutoFeatureExtractor, Wav2Vec2BertModel

import dac

SSL_MODEL = "facebook/w2v-bert-2.0"
FE_SR = 16000
SR_OUT = 48000
AUDIO_EXTS = (".wav", ".flac", ".mp3", ".ogg", ".opus", ".m4a")


def log(m: str) -> None:
    print(m, flush=True)


def load_fe(adapter_path: str, device: torch.device) -> Wav2Vec2BertModel:
    """Build the 24L w2v-BERT base and merge the trained LoRA adapter into it."""
    ck = torch.load(adapter_path, map_location="cpu")
    ad = ck["adapter"]
    scaling = ck["lora_alpha"] / ck["r"]
    layers = ck.get("layers", 24)
    model = Wav2Vec2BertModel.from_pretrained(SSL_MODEL, num_hidden_layers=layers, layerdrop=0.0)
    sd = model.state_dict()
    prefixes = sorted({k[: -len(".lora_A.default.weight")]
                       for k in ad if k.endswith(".lora_A.default.weight")})
    merged = 0
    for p in prefixes:                       # p e.g. encoder.layers.0.ffn1.output_dense
        A = ad[p + ".lora_A.default.weight"].float()   # (r, in)
        B = ad[p + ".lora_B.default.weight"].float()   # (out, r)
        delta = scaling * (B @ A)                       # (out, in)
        wkey = p + ".weight"
        sd[wkey] = sd[wkey].float() + delta.to(sd[wkey].dtype)
        bkey = p + ".base_layer.bias"                   # trained (lora_only) bias
        if bkey in ad:
            sd[p + ".bias"] = ad[bkey].to(sd[p + ".bias"].dtype)
        merged += 1
    model.load_state_dict(sd)
    model.to(device).eval()
    for q in model.parameters():
        q.requires_grad_(False)
    log(f"[fe] merged LoRA into {merged} output_dense layers (scaling={scaling}); "
        f"step {ck.get('step')}")
    return model


def load_decoder(path: str, device: torch.device):
    ck = torch.load(path, map_location="cpu")
    ch = ck.get("dec_channels", 3072)
    dec = dac.model.dac.Decoder(input_channel=1024, channels=ch, rates=[8, 5, 4, 3, 2])
    dec.load_state_dict(ck["decoder"])
    dec.to(device).eval()
    for q in dec.parameters():
        q.requires_grad_(False)
    log(f"[dec] DAC decoder channels={ch} ({sum(p.numel() for p in dec.parameters())/1e6:.1f}M); "
        f"step {ck.get('step')}")
    return dec


def _peak_norm(x: np.ndarray, peak: float = 0.95) -> np.ndarray:
    m = float(np.abs(x).max())
    return (x / m * peak).astype("float32") if m > 1e-6 else x.astype("float32")


@torch.no_grad()
def restore_channel(wav16: np.ndarray, fe, dec, proc, device, chunk_s: float, bf16: bool):
    """wav16: 1-D float32 @16k (peak-normalized) -> restored 1-D float32 @48k.
    Long inputs are windowed with output-domain linear crossfade (click-free)."""
    n = len(wav16)
    win = int(chunk_s * FE_SR)
    if n <= win:
        bounds = [(0, n)]
    else:
        ov = int(2.0 * FE_SR)                # 2 s overlap
        hop = win - ov
        bounds = [(s, min(s + win, n)) for s in range(0, n, hop)]
        bounds = [b for b in bounds if b[1] > b[0]]
    out = np.zeros(n * 3 + SR_OUT, dtype="float32")   # 16k->48k is x3; pad slack
    wsum = np.zeros_like(out)
    amp = torch.bfloat16 if (bf16 and device.type == "cuda") else torch.float32
    for s, e in bounds:
        seg = np.pad(wav16[s:e], (40, 40))
        feats_in = proc(seg, sampling_rate=FE_SR, return_tensors="pt")
        feats_in = {k: v.to(device) for k, v in feats_in.items()}
        with torch.autocast(device.type, dtype=amp, enabled=(amp == torch.bfloat16)):
            h = fe(**feats_in).last_hidden_state          # [1,T,1024]
            y = dec(h.transpose(1, 2))                    # [1,1,L] @48k
        y = y.squeeze().float().cpu().numpy()
        o0 = s * 3
        L = min(len(y), len(out) - o0)
        ramp = np.ones(L, dtype="float32")
        if len(bounds) > 1:                                # taper edges for crossfade
            r = min(int(2.0 * SR_OUT), L // 2)
            if r > 0:
                ramp[:r] = np.linspace(0, 1, r)
                ramp[-r:] = np.linspace(1, 0, r)
        out[o0:o0 + L] += y[:L] * ramp
        wsum[o0:o0 + L] += ramp
    valid = wsum > 1e-6
    out[valid] /= wsum[valid]
    return out[: n * 3]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="audio", help="audio file or directory")
    ap.add_argument("--out-dir", default="audio/out")
    ap.add_argument("--fe-adapter", default="checkpoints/fe_adapter_full.pt")
    ap.add_argument("--decoder", default="checkpoints/decoder_only.pt")
    ap.add_argument("--chunk", type=float, default=35.0, help="window seconds (single-shot if clip is shorter)")
    ap.add_argument("--mono", action="store_true", help="downmix to mono instead of per-channel")
    ap.add_argument("--bf16", type=int, default=1)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    dev = torch.device(a.device if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("medium")
    os.makedirs(a.out_dir, exist_ok=True)

    if os.path.isdir(a.input):
        files = sorted(f for f in glob.glob(os.path.join(a.input, "*"))
                       if f.lower().endswith(AUDIO_EXTS))
    else:
        files = [a.input]
    if not files:
        raise SystemExit(f"no audio under {a.input}")

    fe = load_fe(a.fe_adapter, dev)
    dec = load_decoder(a.decoder, dev)
    proc = AutoFeatureExtractor.from_pretrained(SSL_MODEL)

    for path in files:
        name = os.path.splitext(os.path.basename(path))[0]
        data, sr = sf.read(path, always_2d=True, dtype="float32")   # [N, C]
        if a.mono:
            data = data.mean(axis=1, keepdims=True)
        nch = data.shape[1]
        t0 = time.time()
        chans = []
        for c in range(nch):
            x = data[:, c]
            x16 = (torchaudio.functional.resample(torch.from_numpy(x)[None], sr, FE_SR)[0].numpy()
                   if sr != FE_SR else x)
            x16 = _peak_norm(x16, 0.95)
            y48 = restore_channel(x16, fe, dec, proc, dev, a.chunk, bool(a.bf16))
            chans.append(_peak_norm(y48, 0.97))
        L = max(len(c) for c in chans)
        chans = [np.pad(c, (0, L - len(c))) for c in chans]
        restored = np.stack(chans, axis=1)                          # [L, C]

        # naive-upsampled input (no model) for an A/B reference
        orig = torchaudio.functional.resample(torch.from_numpy(data.T), sr, SR_OUT).T.numpy()
        sf.write(os.path.join(a.out_dir, f"{name}_orig48k.wav"), _peak_norm(orig, 0.97), SR_OUT)
        sf.write(os.path.join(a.out_dir, f"{name}_restored48k.wav"), restored, SR_OUT)
        dur = data.shape[0] / sr
        log(f"[ok] {name}: {nch}ch {sr}Hz {dur:.1f}s -> 48k in {time.time()-t0:.1f}s "
            f"(RTF {(time.time()-t0)/dur:.2f})")

    log(f"[done] outputs in {a.out_dir}")


if __name__ == "__main__":
    main()
