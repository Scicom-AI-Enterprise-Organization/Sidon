#!/usr/bin/env python3
"""Sidon call-centre decoder stage (stage 2/3): train a DAC decoder + GAN to
reconstruct CLEAN 48 kHz waveform from the call-centre-finetuned FE's features.

  clean 48k (EARS/Expresso) --downsample 16k--> --telephony-degrade-->
      --(FROZEN call-centre FE: 24L w2v-BERT + LoRA from stage 1)--> features[T,1024]
      --(TRAINABLE DAC decoder)--> 48 kHz waveform
  loss = DAC multi-res mel (15x) + adversarial (2x) + feature-matching (1x)  vs the clean 48k window.

FE is frozen (stage 1 already trained it on telephony→clean distillation); here only
the decoder + discriminator train. Manual loop (mirrors the verified dry-run), wandb,
resumable checkpoints.
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import glob
import random
import time

import numpy as np
import soundfile as sf
import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset
from transformers import AutoFeatureExtractor, Wav2Vec2BertModel
from peft import LoraConfig, inject_adapter_in_model
from omegaconf import OmegaConf

import audiotools
import dac
from sidon.model.losses import DACLoss, GANLoss

SSL_MODEL = "facebook/w2v-bert-2.0"
LAYERS = 24
HIDDEN = 1024
SR = 48000          # decoder output / clean target
FE_SR = 16000       # FE input
AUDIO_EXTS = (".wav", ".flac", ".mp3", ".ogg", ".opus", ".m4a")

DAC_LOSS_CFG = OmegaConf.create({
    "stft_loss": {"window_lengths": [2048, 512]},
    "mel_loss": {"n_mels": [5, 10, 20, 40, 80, 160, 320],
                 "window_lengths": [32, 64, 128, 256, 512, 1024, 2048],
                 "mel_fmin": [0, 0, 0, 0, 0, 0, 0],
                 "mel_fmax": [None, None, None, None, None, None, None],
                 "pow": 1.0, "clamp_eps": 1.0e-5, "mag_weight": 0.0},
})
W = {"regression_loss": 15.0, "adv_gen": 2.0, "adv_feat": 1.0}


def log(m: str) -> None:
    print(m, flush=True)


def _winit(_):
    """Per-worker init: single-thread + decorrelate RNGs (persistent workers seed
    once, then advance across epochs so each window gets a fresh crop/degradation)."""
    torch.set_num_threads(1)
    info = torch.utils.data.get_worker_info()
    if info is not None:
        s = int(info.seed % (2 ** 31))
        random.seed(s); np.random.seed(s); torch.manual_seed(s)


def _infinite(loader):
    """Re-iterate a map-style DataLoader across epochs (re-shuffles each pass)."""
    while True:
        for batch in loader:
            yield batch


def telephony_degrade(x: np.ndarray, rng: random.Random) -> np.ndarray:
    """16 kHz clean -> telephony-degraded 16 kHz (same as stage 1)."""
    t = torch.from_numpy(x).float().view(1, -1)
    n = t.shape[-1]
    t = torchaudio.functional.resample(torchaudio.functional.resample(t, FE_SR, 8000), 8000, FE_SR)
    t = t[:, :n] if t.shape[-1] >= n else torch.nn.functional.pad(t, (0, n - t.shape[-1]))
    if rng.random() < 0.7:
        t = t.clamp(-1, 1)
        t = torchaudio.functional.mu_law_decoding(torchaudio.functional.mu_law_encoding(t, 256), 256)
    if rng.random() < 0.4:
        try:
            eff = torchaudio.io.AudioEffector(format="mp3", codec_config=torchaudio.io.CodecConfig(qscale=9))
            y = eff.apply(t.view(-1, 1), FE_SR).view(1, -1)
            t = y[:, :n] if y.shape[-1] >= n else torch.nn.functional.pad(y, (0, n - y.shape[-1]))
        except Exception:
            pass
    if rng.random() < 0.5:
        dur = n / FE_SR
        for _ in range(max(1, int(dur * 3 / 10))):
            d = rng.uniform(0.02, 0.15); s = rng.uniform(0, max(0.0, dur - d))
            t[:, int(s * FE_SR):int((s + d) * FE_SR)] = 0
    if rng.random() < 0.6:
        snr = rng.uniform(5, 25); p = t.pow(2).mean() + 1e-9
        t = t + torch.randn_like(t) * torch.sqrt(p / (10 ** (snr / 10)))
    m = t.abs().max()
    return (t / m * 0.95 if m > 1e-6 else t).view(-1).numpy().astype("float32")


class Clean48Telephony(Dataset):
    """Map-style: one item = (clean48 [win*SR], degraded16 [win*FE_SR]) for the same
    random window. One slot per `win` seconds of each file (longer files -> more
    slots), so one epoch ~= one pass over all the audio."""
    def __init__(self, files, win_s):
        self.win = win_s
        self.index = []        # (path, samplerate, frames, wlen_frames) per window-slot
        self.total_sec = 0.0
        for p in files:
            try:
                fi = sf.info(p)
            except Exception:  # noqa: BLE001
                continue
            if fi.samplerate <= 0 or fi.frames < int(0.5 * fi.samplerate):
                continue
            self.total_sec += fi.frames / fi.samplerate
            wlen = int(win_s * fi.samplerate)
            nwin = max(1, fi.frames // max(1, wlen))
            self.index.extend([(p, fi.samplerate, fi.frames, wlen)] * nwin)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        path, sr0, frames, wlen = self.index[idx]
        n48 = int(self.win * SR)
        n16 = int(self.win * FE_SR)
        start = random.randint(0, max(0, frames - wlen))
        try:
            x, _ = sf.read(path, start=start, frames=wlen if frames > wlen else -1,
                           dtype="float32", always_2d=False)
        except Exception:  # noqa: BLE001
            return np.zeros(n48, "float32"), np.zeros(n16, "float32")
        if x.ndim > 1:
            x = x[:, 0]
        t = torch.from_numpy(np.ascontiguousarray(x)).float().view(1, -1)
        c48 = (torchaudio.functional.resample(t, sr0, SR) if sr0 != SR else t).view(-1)
        m = c48.abs().max()
        if m <= 1e-6:
            return np.zeros(n48, "float32"), np.zeros(n16, "float32")
        c48 = (c48 / m * 0.95)
        c48 = torch.nn.functional.pad(c48, (0, max(0, n48 - c48.shape[-1])))[:n48].numpy().astype("float32")
        c16 = torchaudio.functional.resample(torch.from_numpy(c48).view(1, -1), SR, FE_SR).view(-1).numpy()
        d16 = telephony_degrade(c16, random)[:len(c16)]
        if len(d16) < len(c16):
            d16 = np.pad(d16, (0, len(c16) - len(d16)))
        return c48, d16


def load_frozen_fe(ckpt_path, dev):
    m = Wav2Vec2BertModel.from_pretrained(SSL_MODEL, num_hidden_layers=LAYERS, layerdrop=0.0)
    m = inject_adapter_in_model(LoraConfig(lora_alpha=16, lora_dropout=0.1, r=64,
                                           bias="lora_only", target_modules=["output_dense"]), m)
    sd = torch.load(ckpt_path, map_location="cpu")["student"]
    res = m.load_state_dict(sd, strict=False)
    log(f"[fe] loaded {ckpt_path}: missing={len(res.missing_keys)} unexpected={len(res.unexpected_keys)}")
    m.to(dev).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="/data/clean48k")
    ap.add_argument("--fe-ckpt", default="/Sidon/fe_callcentre/last.pt")
    ap.add_argument("--steps", type=int, default=50000)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--accum", type=int, default=4, help="grad accumulation; effective batch = batch*accum")
    ap.add_argument("--win", type=float, default=8.0)
    ap.add_argument("--bf16", type=int, default=1, help="1=bf16 autocast for FE+decoder fwd (halves memory on H100)")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--save-every", type=int, default=2000)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--dec-channels", type=int, default=1536,
                    help="DAC decoder base channels (capacity). 1536=52M, 2048=88M, 3072=188M, 4096=324M")
    ap.add_argument("--out", default="/Sidon/decoder_callcentre")
    ap.add_argument("--wandb-name", default="decoder-callcentre")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    dev = torch.device(a.device if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("medium"); torch.set_num_threads(1)
    os.makedirs(a.out, exist_ok=True)

    files = [p for p in glob.glob(os.path.join(a.data_root, "**", "*"), recursive=True)
             if p.lower().endswith(AUDIO_EXTS)]
    if not files:
        raise SystemExit(f"no audio under {a.data_root}")
    ds = Clean48Telephony(files, a.win)
    if len(ds) == 0:
        raise SystemExit(f"no usable audio under {a.data_root}")
    opt_steps_per_epoch = max(1, len(ds) // (a.batch * max(1, a.accum)))
    log(f"[data] {len(files)} files ~{ds.total_sec/3600:.1f}h -> {len(ds)} windows/epoch "
        f"({a.win}s); {opt_steps_per_epoch} opt-steps/epoch")

    proc = AutoFeatureExtractor.from_pretrained(SSL_MODEL)

    def collate(samples):
        c48 = torch.stack([torch.from_numpy(c) for c, _ in samples])
        d16 = [torch.nn.functional.pad(torch.from_numpy(d), (40, 40)).numpy() for _, d in samples]
        ssl = proc(d16, sampling_rate=FE_SR, return_tensors="pt", padding=True)
        return c48, ssl

    loader = DataLoader(ds, batch_size=a.batch, shuffle=True, drop_last=True,
                        num_workers=a.num_workers, collate_fn=collate, pin_memory=True,
                        worker_init_fn=_winit, persistent_workers=a.num_workers > 0)

    fe = load_frozen_fe(a.fe_ckpt, dev)
    decoder = dac.model.dac.Decoder(input_channel=HIDDEN, channels=a.dec_channels, rates=[8, 5, 4, 3, 2]).to(dev).train()
    disc = GANLoss(dac.model.discriminator.Discriminator(sample_rate=SR)).to(dev).train()
    reg = DACLoss(DAC_LOSS_CFG)
    log(f"[model] decoder {sum(p.numel() for p in decoder.parameters())/1e6:.1f}M + disc "
        f"{sum(p.numel() for p in disc.parameters())/1e6:.1f}M (FE frozen)")

    opt_g = torch.optim.AdamW(decoder.parameters(), lr=a.lr, weight_decay=0.01, betas=(0.8, 0.98))
    opt_d = torch.optim.AdamW(disc.parameters(), lr=a.lr, weight_decay=0.01, betas=(0.8, 0.98))

    last = os.path.join(a.out, "last.pt")
    step0 = 0
    if os.path.exists(last):
        ck = torch.load(last, map_location=dev)
        decoder.load_state_dict(ck["decoder"]); disc.load_state_dict(ck["disc"])
        opt_g.load_state_dict(ck["opt_g"]); opt_d.load_state_dict(ck["opt_d"]); step0 = ck["step"]
        log(f"[resume] step {step0}")

    wb = None
    if os.environ.get("WANDB_API_KEY") and os.environ.get("WANDB_MODE", "online") != "disabled":
        try:
            import wandb
            wb = wandb.init(project="sidon", name=a.wandb_name, id=a.wandb_name, resume="allow",
                            config={"layers": LAYERS, "steps": a.steps, "batch": a.batch,
                                    "accum": a.accum, "win_s": a.win, "lr": a.lr, "sr": SR,
                                    "dec_channels": a.dec_channels,
                                    "windows_per_epoch": len(ds),
                                    "opt_steps_per_epoch": opt_steps_per_epoch})
            log(f"[wandb] run: {wb.url}")
        except Exception as e:  # noqa: BLE001
            log(f"[wandb] disabled ({e})")

    use_amp = bool(a.bf16) and dev.type == "cuda"
    log(f"[train] steps={a.steps} batch={a.batch} accum={a.accum} (eff {a.batch*a.accum}) "
        f"win={a.win}s lr={a.lr} bf16={use_amp}")
    disc_params = list(disc.parameters())
    accum = max(1, a.accum)
    opt_g.zero_grad(set_to_none=True); opt_d.zero_grad(set_to_none=True)
    step = step0; micro = 0; nmicro = 0; t0 = time.time()
    acc = {"mel": 0.0, "adv_gen": 0.0, "adv_feat": 0.0, "d": 0.0, "g": 0.0}
    stop = False
    for c48, ssl in _infinite(loader):
        ssl = {k: v.to(dev) for k, v in ssl.items()}
        target = c48.to(dev)
        # bf16 autocast around the heavy FE + decoder forward (activations dominate
        # memory). The STFT/mel loss + GAN run in fp32 (cast pred back) for stability.
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            with torch.no_grad():
                feats = fe(**ssl).last_hidden_state.detach()
            pred = decoder(feats.transpose(1, 2))
        pred = pred.float()
        ml = min(pred.shape[-1], target.shape[-1]); bsz = target.shape[0]
        ps = audiotools.AudioSignal(pred[..., :ml].reshape(bsz, 1, -1), sample_rate=SR)
        ts = audiotools.AudioSignal(target[:, :ml].reshape(bsz, 1, -1), sample_rate=SR)
        mel = reg(ts, ps)["mel_loss"]
        # generator grads: freeze disc PARAMS so g_loss.backward doesn't pollute the
        # accumulated discriminator grads (gradient still flows through disc to decoder).
        for p in disc_params:
            p.requires_grad_(False)
        adv_gen, adv_feat = disc.generator_loss(ps.audio_data, ts.audio_data)
        g_loss = W["regression_loss"] * mel + W["adv_gen"] * adv_gen + W["adv_feat"] * adv_feat
        (g_loss / accum).backward()
        for p in disc_params:
            p.requires_grad_(True)
        # discriminator grads on detached pred (fresh graph)
        d_loss = disc.discriminator_loss(ps.audio_data.detach(), ts.audio_data)
        (d_loss / accum).backward()
        micro += 1; nmicro += 1
        for k, v in zip(acc, [mel, adv_gen, adv_feat, d_loss, g_loss]):
            acc[k] += float(v)
        if micro % accum == 0:
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0); opt_g.step(); opt_g.zero_grad(set_to_none=True)
            torch.nn.utils.clip_grad_norm_(disc_params, 1.0); opt_d.step(); opt_d.zero_grad(set_to_none=True)
            step += 1
            if step % a.log_every == 0:
                m = {k: v / nmicro for k, v in acc.items()}
                m["lr"] = opt_g.param_groups[0]["lr"]
                m["epoch"] = step / opt_steps_per_epoch
                log(f"[step {step}/{a.steps}] ep={m['epoch']:.2f} mel={m['mel']:.4f} adv_gen={m['adv_gen']:.4f} "
                    f"adv_feat={m['adv_feat']:.4f} d={m['d']:.4f} g={m['g']:.3f} lr={m['lr']:.1e} "
                    f"{(time.time()-t0)/max(1,step-step0):.2f}s/st")
                if wb is not None:
                    wb.log(m, step=step)
                acc = {k: 0.0 for k in acc}; nmicro = 0
            if step % a.save_every == 0 or step >= a.steps:
                torch.save({"step": step, "decoder": decoder.state_dict(), "disc": disc.state_dict(),
                            "opt_g": opt_g.state_dict(), "opt_d": opt_d.state_dict(),
                            "dec_channels": a.dec_channels}, last)
                log(f"[ckpt] {last} @ {step}")
            if step >= a.steps:
                stop = True
        if stop:
            break
    if wb is not None:
        wb.finish()
    log("[done] decoder call-centre stage complete.")


if __name__ == "__main__":
    main()
