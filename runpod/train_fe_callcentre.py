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
from torch.utils.data import DataLoader, IterableDataset
import transformers
from transformers import AutoFeatureExtractor, Wav2Vec2BertModel
from peft import LoraConfig, inject_adapter_in_model

SSL_MODEL = "facebook/w2v-bert-2.0"
LAYERS = 24            # full w2v-BERT 2.0 (vs the 8-layer Sidon default)
SR = 16000            # FE works at 16 kHz
AUDIO_EXTS = (".wav", ".flac", ".mp3", ".ogg", ".opus", ".m4a")


def log(m: str) -> None:
    print(m, flush=True)


def telephony_degrade(x: np.ndarray, rng: random.Random) -> np.ndarray:
    """Simulate call-centre/telephony degradation on a 16 kHz clean signal."""
    t = torch.from_numpy(x).float().view(1, -1)
    n = t.shape[-1]
    # 1) narrowband (defining telephony trait): 16k -> 8k -> 16k
    t = torchaudio.functional.resample(torchaudio.functional.resample(t, SR, 8000), 8000, SR)
    t = t[:, :n] if t.shape[-1] >= n else torch.nn.functional.pad(t, (0, n - t.shape[-1]))
    # 2) G.711 mu-law companding
    if rng.random() < 0.7:
        t = t.clamp(-1, 1)
        t = torchaudio.functional.mu_law_decoding(torchaudio.functional.mu_law_encoding(t, 256), 256)
    # 3) lossy codec (mp3) as a proxy for VoIP codecs
    if rng.random() < 0.4:
        try:
            eff = torchaudio.io.AudioEffector(
                format="mp3", codec_config=torchaudio.io.CodecConfig(qscale=9))
            y = eff.apply(t.view(-1, 1), SR).view(1, -1)
            t = y[:, :n] if y.shape[-1] >= n else torch.nn.functional.pad(y, (0, n - y.shape[-1]))
        except Exception:
            pass
    # 4) packet loss: zero random 20-150 ms chunks
    if rng.random() < 0.5:
        dur = n / SR
        for _ in range(max(1, int(dur * 3 / 10))):
            d = rng.uniform(0.02, 0.15)
            s = rng.uniform(0, max(0.0, dur - d))
            t[:, int(s * SR):int((s + d) * SR)] = 0
    # 5) additive noise at random SNR
    if rng.random() < 0.6:
        snr = rng.uniform(5, 25)
        p = t.pow(2).mean() + 1e-9
        t = t + torch.randn_like(t) * torch.sqrt(p / (10 ** (snr / 10)))
    m = t.abs().max()
    if m > 1e-6:
        t = t / m * 0.95
    return t.view(-1).numpy().astype("float32")


class CleanTelephonyPairs(IterableDataset):
    def __init__(self, files, win_s, seed=0):
        self.files = files
        self.win = win_s
        self.seed = seed

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        rng = random.Random(self.seed * 1000 + (info.id if info else 0) + 1)
        win = int(self.win * SR)
        while True:
            path = rng.choice(self.files)
            try:
                fi = sf.info(path)
                sr0, frames = fi.samplerate, fi.frames
                wlen = int(self.win * sr0)
                start = rng.randint(0, max(0, frames - wlen))
                x, _ = sf.read(path, start=start, frames=wlen if frames > wlen else -1,
                               dtype="float32", always_2d=False)
            except Exception:
                continue
            if x.ndim > 1:
                x = x[:, 0]
            if len(x) < int(0.5 * sr0):
                continue
            if sr0 != SR:
                x = torchaudio.functional.resample(torch.from_numpy(x).float().view(1, -1), sr0, SR).view(-1).numpy()
            m = np.abs(x).max()
            if m > 1e-6:
                x = (x / m * 0.95).astype("float32")
            else:
                continue
            clean = x[:win]
            degraded = telephony_degrade(clean, rng)[:len(clean)]
            if len(degraded) < len(clean):
                degraded = np.pad(degraded, (0, len(clean) - len(degraded)))
            yield clean, degraded


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
    log(f"[data] {len(files)} clean audio files under {a.data_root}")

    # wandb (only if a key is present; resumes the same run on relaunch)
    wb = None
    if os.environ.get("WANDB_API_KEY") and os.environ.get("WANDB_MODE", "online") != "disabled":
        try:
            import wandb
            wb = wandb.init(project=a.wandb_project, name=a.wandb_name,
                            id=a.wandb_name, resume="allow",
                            config={"layers": LAYERS, "steps": a.steps, "batch": a.batch,
                                    "win_s": a.win, "lr": a.lr, "warmup": a.warmup,
                                    "n_files": len(files), "sr": SR, "lora_r": 64})
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

    loader = DataLoader(CleanTelephonyPairs(files, a.win), batch_size=a.batch,
                        num_workers=a.num_workers, collate_fn=collate, pin_memory=True,
                        worker_init_fn=lambda _: torch.set_num_threads(1),
                        persistent_workers=a.num_workers > 0)

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

    log(f"[train] steps={a.steps} batch={a.batch} win={a.win}s lr={a.lr} warmup={a.warmup}")
    step = step0
    t0 = time.time()
    run = 0.0
    for sc, sd in loader:
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
            log(f"[step {step}/{a.steps}] mse={avg:.5f} lr={sched.get_last_lr()[0]:.2e} "
                f"{(time.time()-t0)/(step-step0):.2f}s/step")
            if wb is not None:
                wb.log({"mse": avg, "lr": sched.get_last_lr()[0]}, step=step)
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
