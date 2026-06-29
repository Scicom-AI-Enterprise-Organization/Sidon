#!/usr/bin/env python3
"""Sidon stage-1 feature-predictor finetune for CALL-CENTRE / telephony audio.

Per the plan: take the FULL 24-layer w2v-BERT 2.0 (NOT the 8-layer subset, and NOT
warm-started from the frozen-TorchScript v0.1 weights), add a fresh LoRA adapter,
and distill it to map TELEPHONY-DEGRADED speech features back to the CLEAN features
a frozen teacher produces from the same clean audio:

  clean 48k (EARS / Expresso) --downsample--> clean 16k --(teacher 24L, frozen)--> H_clean
                                            \--telephony-degrade--> degraded 16k --(student 24L + LoRA)--> H_student
  loss = MSE(H_student, H_clean)

Only the LoRA adapter trains (~16M params); the 580M base + teacher are frozen.
Manual loop (no Lightning/Hydra) with a direct audio loader — same shape as the
verified dry-run. Runs at 16 kHz (the FE is SR-agnostic; the 48k source just makes
the 16k clean reference genuinely clean rather than already-telephony).
"""
from __future__ import annotations

# Many-core pod: cap thread pools BEFORE numpy/torch import (else N workers ×
# ~CPU threads oversubscribe and starve the GPU). See neucodec-44k CLAUDE "CPU threads".
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
import transformers
from transformers import AutoFeatureExtractor, Wav2Vec2BertModel
from peft import LoraConfig, inject_adapter_in_model
from degradations import Degrader  # realistic call-centre degradation (ported from neucodec-44k-se)

SSL_MODEL = "facebook/w2v-bert-2.0"
LAYERS = 24            # full w2v-BERT 2.0 (vs the 8-layer Sidon default)
SR = 16000            # FE works at 16 kHz
AUDIO_EXTS = (".wav", ".flac", ".mp3", ".ogg", ".opus", ".m4a")

# Realistic call-centre degradation (Degrader, ported from neucodec-44k-se, tuned vs
# real emgs call samples): 80% telephony mimic (telephone HP -> random <6 kHz narrowband
# ceiling -> GSM/G.711-mulaw -> 16-40 kbps MP3 -> line noise) + 20% generic chain.
DEGRADE_CFG = {
    "enable": True, "mode": "mix", "telephony_share": 0.8,
    "prob_reverb": 0.3, "prob_noise": 0.5, "prob_band_limit": 0.5,
    "prob_clip": 0.5, "prob_codec": 0.5, "prob_packet_loss": 0.3,
    "band_limit_srs": [4000, 6000, 8000, 11025, 12000], "codec_qscale": [1, 10],
    "reverb_backend": "synthetic", "reverb_rt60": [0.1, 0.7],
    "noise_filelist": None, "noise_snr": [-5, 20],
    "telephony": {
        "band_hz": [2800, 4200], "hp_hz": [200, 350],
        "codecs": ["gsm", "mulaw", "none"], "codec_weights": [0.6, 0.25, 0.15],
        "mp3_kbps": [16, 24, 32, 40], "snr_db": [8, 28], "noise_prob": 0.85,
        "sr_choices": [8000, 11025, 12000, 16000],
    },
}


def log(m: str) -> None:
    print(m, flush=True)


def _winit(_):
    """Per-worker init: single-thread (avoid oversubscription) and decorrelate RNGs.
    With persistent_workers the worker is seeded once here, then its RNG advances
    across epochs, so each window still gets a fresh random crop + degradation."""
    torch.set_num_threads(1)
    info = torch.utils.data.get_worker_info()
    if info is not None:
        s = int(info.seed % (2 ** 31))
        random.seed(s); np.random.seed(s); torch.manual_seed(s)


def _infinite(loader):
    """Re-iterate a map-style DataLoader across epochs (re-shuffles each pass) so a
    fixed --steps budget can span multiple epochs."""
    while True:
        for batch in loader:
            yield batch


class CleanTelephonyPairs(Dataset):
    """Map-style window sampler: one item = one `win`-second window as a
    (clean 16k, telephony-degraded 16k) pair. The index holds one slot per `win`
    seconds of every file (longer files -> more slots), so a full pass over the
    dataset (one epoch) covers ~all the audio once. Window starts are random per
    access, so re-epochs see different crops + fresh degradation."""
    def __init__(self, files, win_s):
        self.win = win_s
        self.degrader = Degrader(DEGRADE_CFG)
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
        win = int(self.win * SR)
        start = random.randint(0, max(0, frames - wlen))
        try:
            x, _ = sf.read(path, start=start, frames=wlen if frames > wlen else -1,
                           dtype="float32", always_2d=False)
        except Exception:  # noqa: BLE001
            return np.zeros(win, "float32"), np.zeros(win, "float32")
        if x.ndim > 1:
            x = x[:, 0]
        if sr0 != SR:
            x = torchaudio.functional.resample(torch.from_numpy(x).float().view(1, -1), sr0, SR).view(-1).numpy()
        m = np.abs(x).max()
        if m <= 1e-6:
            return np.zeros(win, "float32"), np.zeros(win, "float32")
        x = (x / m * 0.95).astype("float32")
        clean = np.zeros(win, "float32")
        clean[:min(win, len(x))] = x[:win]
        degraded = self.degrader.degrade(clean, SR)[:win]
        if len(degraded) < win:
            degraded = np.pad(degraded, (0, win - len(degraded)))
        return clean, degraded


def build_student():
    m = Wav2Vec2BertModel.from_pretrained(SSL_MODEL, num_hidden_layers=LAYERS, layerdrop=0.0)
    m = inject_adapter_in_model(
        LoraConfig(lora_alpha=16, lora_dropout=0.1, r=64, bias="lora_only",
                   target_modules=["output_dense"]), m)
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="/data/clean48k")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--win", type=float, default=12.0)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=2000)
    ap.add_argument("--save-every", type=int, default=2000)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--out", default="/Sidon/fe_callcentre")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--wandb-project", default="sidon")
    ap.add_argument("--wandb-name", default="fe-callcentre-d24")
    a = ap.parse_args()
    dev = torch.device(a.device if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("medium")
    torch.set_num_threads(1)
    os.makedirs(a.out, exist_ok=True)

    files = [p for p in glob.glob(os.path.join(a.data_root, "**", "*"), recursive=True)
             if p.lower().endswith(AUDIO_EXTS)]
    if not files:
        raise SystemExit(f"no audio under {a.data_root}")
    ds = CleanTelephonyPairs(files, a.win)
    if len(ds) == 0:
        raise SystemExit(f"no usable audio under {a.data_root}")
    log(f"[data] {len(files)} files ~{ds.total_sec/3600:.1f}h -> {len(ds)} windows/epoch ({a.win}s)")

    # wandb (only if a key is present; resumes the same run on relaunch)
    wb = None
    if os.environ.get("WANDB_API_KEY") and os.environ.get("WANDB_MODE", "online") != "disabled":
        try:
            import wandb
            wb = wandb.init(project=a.wandb_project, name=a.wandb_name,
                            id=a.wandb_name, resume="allow",
                            config={"layers": LAYERS, "steps": a.steps, "batch": a.batch,
                                    "win_s": a.win, "lr": a.lr, "warmup": a.warmup,
                                    "n_files": len(files), "windows_per_epoch": len(ds),
                                    "sr": SR, "lora_r": 64})
            log(f"[wandb] run: {wb.url}")
        except Exception as e:  # noqa: BLE001
            log(f"[wandb] disabled ({e})")

    proc = AutoFeatureExtractor.from_pretrained(SSL_MODEL)

    def collate(samples):
        cl = [torch.nn.functional.pad(torch.from_numpy(c), (40, 40)).numpy() for c, _ in samples]
        dg = [torch.nn.functional.pad(torch.from_numpy(d), (40, 40)).numpy() for _, d in samples]
        sc = proc(cl, sampling_rate=SR, return_tensors="pt", padding=True)
        sd = proc(dg, sampling_rate=SR, return_tensors="pt", padding=True)
        return sc, sd

    loader = DataLoader(ds, batch_size=a.batch, shuffle=True, drop_last=True,
                        num_workers=a.num_workers, collate_fn=collate, pin_memory=True,
                        worker_init_fn=_winit, persistent_workers=a.num_workers > 0)
    steps_per_epoch = max(1, len(loader))

    log(f"[model] loading teacher + student (w2v-BERT {LAYERS}L) …")
    teacher = Wav2Vec2BertModel.from_pretrained(SSL_MODEL, num_hidden_layers=LAYERS).to(dev).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    student = build_student().to(dev).train()
    n_tr = sum(p.numel() for p in student.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in student.parameters())
    log(f"[model] student trainable {n_tr/1e6:.2f}M / {n_all/1e6:.1f}M (LoRA only; base+teacher frozen)")

    opt = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad],
                            lr=a.lr, weight_decay=0.01)
    sched = transformers.get_constant_schedule_with_warmup(opt, num_warmup_steps=a.warmup)
    mse = torch.nn.MSELoss()

    # resume
    last = os.path.join(a.out, "last.pt")
    step0 = 0
    if os.path.exists(last):
        ck = torch.load(last, map_location=dev)
        student.load_state_dict(ck["student"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        step0 = ck["step"]
        log(f"[resume] from step {step0}")

    log(f"[train] steps={a.steps} batch={a.batch} win={a.win}s lr={a.lr} warmup={a.warmup} "
        f"steps/epoch={steps_per_epoch}")
    step = step0
    t0 = time.time()
    run = 0.0
    for sc, sd in _infinite(loader):
        sc = {k: v.to(dev) for k, v in sc.items()}
        sd = {k: v.to(dev) for k, v in sd.items()}
        with torch.no_grad():
            h_clean = teacher(**sc).last_hidden_state
        h_stud = student(**sd).last_hidden_state
        tlen = min(h_clean.shape[1], h_stud.shape[1])
        loss = mse(h_stud[:, :tlen], h_clean[:, :tlen].detach())
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in student.parameters() if p.requires_grad], 1.0)
        opt.step(); sched.step()
        step += 1
        run += loss.item()
        if step % a.log_every == 0:
            avg = run / a.log_every
            ep = step / steps_per_epoch
            log(f"[step {step}/{a.steps}] ep={ep:.2f} mse={avg:.5f} lr={sched.get_last_lr()[0]:.2e} "
                f"{(time.time()-t0)/max(1,step-step0):.2f}s/step")
            if wb is not None:
                wb.log({"mse": avg, "lr": sched.get_last_lr()[0], "epoch": ep}, step=step)
            run = 0.0
        if step % a.save_every == 0 or step >= a.steps:
            torch.save({"step": step, "student": student.state_dict(),
                        "opt": opt.state_dict(), "sched": sched.state_dict()}, last)
            log(f"[ckpt] saved {last} @ step {step}")
        if step >= a.steps:
            break
    if wb is not None:
        wb.finish()
    log("[done] FE call-centre finetune complete.")


if __name__ == "__main__":
    main()
