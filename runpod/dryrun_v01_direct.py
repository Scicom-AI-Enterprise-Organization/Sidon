#!/usr/bin/env python3
"""Dry-run finetune from sidon-v0.1's frozen feature extractor — DIRECT loader.

No WebDataset, no shard packing: a small custom IterableDataset reads random
windows straight from the extracted sg-podcast mp3s (ffmpeg `-ss/-t`, so long
podcasts are never fully decoded), lightly degrades them in-process, and runs the
w2v-BERT feature extractor. Then, as in stage-3 finetuning, v0.1's feature
extractor stays FROZEN (its released weights are frozen TorchScript and can't be
loaded into a trainable module anyway) and we train a fresh DAC decoder + GAN on
top of its features.

  random mp3 window --(degrade)--> noisy 16k --(w2v-bert FE)--> input_features[160]
        --(frozen v0.1 FE, TorchScript)--> SSL features [T,1024]
        --(TRAINABLE DAC decoder)--------> 48 kHz waveform
  loss = DAC multi-res mel + adversarial/feature-matching vs the clean window.

Usage (on the pod, PYTHONPATH=/Sidon/src, HF_HOME=/hf_cache):
  python dryrun_v01_direct.py --steps 20 --batch 2 --win 10 --max-files 60
"""
from __future__ import annotations

# CRITICAL on many-core pods (this H100 box has 224 vCPUs): numpy/torch otherwise
# spin up a ~224-thread pool PER process, and with N dataloader workers that is
# N x 224 threads -> oversubscription that starves the GPU to ~1%. Must be set
# BEFORE numpy/torch import (mirrors ../neucodec-44k CLAUDE.md "CPU threads").
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import glob
import io
import random
import subprocess
import tempfile
import time

import numpy as np
import soundfile as sf
import torch
import torchaudio
from huggingface_hub import hf_hub_download
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, IterableDataset
from transformers import AutoFeatureExtractor

import audiotools
import dac
from sidon.model.losses import DACLoss, GANLoss

REPO = "sarulab-speech/sidon-v0.1"
SSL_MODEL = "facebook/w2v-bert-2.0"
HIDDEN = 1024
SR = 48_000

DAC_LOSS_CFG = OmegaConf.create({
    "stft_loss": {"window_lengths": [2048, 512]},
    "mel_loss": {
        "n_mels": [5, 10, 20, 40, 80, 160, 320],
        "window_lengths": [32, 64, 128, 256, 512, 1024, 2048],
        "mel_fmin": [0, 0, 0, 0, 0, 0, 0],
        "mel_fmax": [None, None, None, None, None, None, None],
        "pow": 1.0, "clamp_eps": 1.0e-5, "mag_weight": 0.0,
    },
})
W = {"regression_loss": 15.0, "adv_gen": 2.0, "adv_feature": 1.0}


def log(m: str) -> None:
    print(m, flush=True)


def probe_duration(path: str) -> float:
    try:
        from mutagen.mp3 import MP3
        return float(MP3(path).info.length)
    except Exception:  # noqa: BLE001
        return 0.0


def ffmpeg_window(path: str, start: float, win: float) -> np.ndarray | None:
    """Decode just [start, start+win) of an mp3 -> 48k mono float32 numpy."""
    with tempfile.NamedTemporaryFile(suffix=".wav") as tf:
        cmd = [
            "ffmpeg", "-v", "error", "-y", "-ss", f"{start:.3f}", "-t", f"{win:.3f}",
            "-i", path, "-ar", str(SR), "-ac", "1", "-c:a", "pcm_s16le", tf.name,
        ]
        if subprocess.run(cmd).returncode != 0:
            return None
        try:
            data, _ = sf.read(tf.name, dtype="float32")
        except Exception:  # noqa: BLE001
            return None
    return data if data.ndim == 1 else data[:, 0]


def degrade(clean: np.ndarray, rng: random.Random) -> np.ndarray:
    """Light, cheap degradation so the FE has something to denoise."""
    noisy = clean.copy()
    snr = rng.uniform(5.0, 30.0)
    sig_p = float(np.mean(noisy ** 2)) + 1e-9
    noise = np.random.default_rng(rng.randint(0, 2**31)).standard_normal(len(noisy)).astype("float32")
    noise *= np.sqrt(sig_p / (10 ** (snr / 10)))
    noisy = noisy + noise
    if rng.random() < 0.5:  # occasional band-limit
        sr2 = rng.choice([8000, 16000, 24000])
        t = torch.from_numpy(noisy)[None]
        t = torchaudio.functional.resample(t, SR, sr2)
        t = torchaudio.functional.resample(t, sr2, SR)
        noisy = t[0].numpy()[: len(clean)]
    m = np.abs(noisy).max()
    return (noisy / m * 0.95) if m > 1e-6 else noisy


class PodcastWindows(IterableDataset):
    def __init__(self, files: list[str], durs: dict[str, float], win: float, seed: int = 0):
        self.files = files
        self.durs = durs
        self.win = win
        self.seed = seed

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        wid = info.id if info else 0
        rng = random.Random(self.seed * 1000 + wid + 1)
        while True:
            path = rng.choice(self.files)
            dur = self.durs.get(path, 0.0)
            if dur < 1.5:
                continue
            start = rng.uniform(0, max(0.0, dur - self.win))
            clean = ffmpeg_window(path, start, self.win)
            if clean is None or len(clean) < int(0.5 * SR):
                continue
            m = np.abs(clean).max()
            if m > 1e-6:
                clean = clean / m * 0.95
            noisy = degrade(clean, rng)
            n16 = torchaudio.functional.resample(torch.from_numpy(noisy)[None], SR, 16000)[0].numpy()
            yield clean.astype("float32"), n16.astype("float32")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--win", type=float, default=10.0)
    ap.add_argument("--max-files", type=int, default=60)
    ap.add_argument("--data-root", default="/data/sg")
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--save", default="/Sidon/ckpt_v0.1/dryrun_decoder.pt")
    a = ap.parse_args()
    dev = torch.device(a.device if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("medium")
    torch.set_num_threads(1)  # main process; workers capped via worker_init_fn

    files = sorted(glob.glob(os.path.join(a.data_root, "**", "*.mp3"), recursive=True))
    if a.max_files > 0:
        files = files[: a.max_files]
    if not files:
        raise SystemExit(f"no mp3 under {a.data_root}")
    log(f"[data] {len(files)} mp3 files; probing durations …")
    durs = {f: probe_duration(f) for f in files}
    files = [f for f in files if durs[f] >= 1.5]
    log(f"[data] {len(files)} usable; total {sum(durs.values())/3600:.1f} h")

    processor = AutoFeatureExtractor.from_pretrained(SSL_MODEL)

    def collate(samples):
        cleans = [torch.from_numpy(c) for c, _ in samples]
        n16 = [np.pad(n, (40, 40)) for _, n in samples]
        feats = processor(n16, sampling_rate=16000, return_tensors="pt", padding=True)
        input_wav = torch.nn.utils.rnn.pad_sequence(cleans, batch_first=True)
        return input_wav, feats["input_features"]

    ds = PodcastWindows(files, durs, a.win)
    loader = DataLoader(ds, batch_size=a.batch, num_workers=a.num_workers,
                        collate_fn=collate, pin_memory=True,
                        worker_init_fn=lambda _: torch.set_num_threads(1))

    log(f"[load] {REPO} feature_extractor (frozen TorchScript) …")
    fe = torch.jit.load(hf_hub_download(REPO, "feature_extractor_cuda.pt"), map_location=dev).eval()

    decoder = dac.model.dac.Decoder(input_channel=HIDDEN, channels=1536, rates=[8, 5, 4, 3, 2]).to(dev).train()
    disc = GANLoss(dac.model.discriminator.Discriminator(sample_rate=SR)).to(dev).train()
    reg_loss = DACLoss(DAC_LOSS_CFG)
    log(f"[model] trainable DAC decoder: {sum(p.numel() for p in decoder.parameters())/1e6:.1f}M  (FE frozen v0.1)")

    opt_g = torch.optim.AdamW(decoder.parameters(), lr=1e-4, weight_decay=0.01, betas=(0.8, 0.98))
    opt_d = torch.optim.AdamW(disc.parameters(), lr=1e-4, weight_decay=0.01, betas=(0.8, 0.98))

    log(f"[train] dry-run: {a.steps} steps, batch={a.batch}, {a.win}s @ {SR} Hz")
    step, t0 = 0, time.time()
    for input_wav, input_features in loader:
        input_features = input_features.to(dev)
        target = input_wav.to(dev)
        with torch.no_grad():
            out = fe(input_features)
            feats = (out["last_hidden_state"] if isinstance(out, dict) and "last_hidden_state" in out
                     else (next(v for v in out.values() if torch.is_tensor(v) and v.ndim == 3) if isinstance(out, dict) else out))
            feats = feats.detach()
        pred = decoder(feats.transpose(1, 2))
        ml = min(pred.shape[-1], target.shape[-1])
        bsz = target.shape[0]
        pred_sig = audiotools.AudioSignal(pred[..., :ml].reshape(bsz, 1, -1), sample_rate=SR)
        tgt_sig = audiotools.AudioSignal(target[:, :ml].reshape(bsz, 1, -1), sample_rate=SR)

        mel = reg_loss(tgt_sig, pred_sig)["mel_loss"]
        d_loss = disc.discriminator_loss(pred_sig.audio_data.detach(), tgt_sig.audio_data)
        opt_d.zero_grad(); d_loss.backward(); torch.nn.utils.clip_grad_norm_(disc.parameters(), 1.0); opt_d.step()
        adv_gen, adv_feat = disc.generator_loss(pred_sig.audio_data, tgt_sig.audio_data)
        g_loss = W["regression_loss"] * mel + W["adv_gen"] * adv_gen + W["adv_feature"] * adv_feat
        opt_g.zero_grad(); g_loss.backward(); torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0); opt_g.step()

        step += 1
        log(f"[step {step:>3}/{a.steps}] mel={mel.item():.4f} adv_gen={adv_gen.item():.4f} "
            f"adv_feat={adv_feat.item():.4f} d={d_loss.item():.4f} g={g_loss.item():.4f} "
            f"| feats={tuple(feats.shape)} pred={tuple(pred.shape)} {(time.time()-t0)/step:.2f}s/step")
        if step >= a.steps:
            break

    os.makedirs(os.path.dirname(a.save), exist_ok=True)
    torch.save({"decoder": decoder.state_dict()}, a.save)
    log(f"[done] dry-run finetune complete; trainable decoder saved -> {a.save}")


if __name__ == "__main__":
    main()
