#!/usr/bin/env python3
"""Download CLEAN 48 kHz speech for the Sidon call-centre FE finetune (teacher pool).

Sources (all genuinely clean studio/anechoic 48k; telephony degradation is simulated
at train time, so we only need clean references):
  - EARS    : facebookresearch anechoic 48k, per-speaker zips on GitHub (subset)
  - Expresso: ylacombe/expresso (read) + nytopop/expresso-conversational (conv), 48k studio

Everything under / (never /workspace). Each source drops a .done sentinel (resumable).
Audio lands as wav under /data/clean48k/<source>/ and the trainer globs it directly.

Usage: python prepare_clean48k.py --out /data/clean48k [--ears-speakers 30]
"""
from __future__ import annotations

import argparse
import io
import os
import subprocess
from pathlib import Path

os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")

EARS_BASE = "https://github.com/facebookresearch/ears_dataset/releases/download/dataset"
HF_DATASETS = {
    "expresso_read": ("ylacombe/expresso", "read", "train"),
    "expresso_conv": ("nytopop/expresso-conversational", "conversational", "train"),
}


def log(m: str) -> None:
    print(m, flush=True)


def hf_login():
    tok = os.environ.get("HF_TOKEN")
    if tok:
        try:
            from huggingface_hub import login
            login(token=tok, add_to_git_credential=False)
            log("[hf] logged in")
        except Exception as e:  # noqa: BLE001
            log(f"[hf] login warning: {e}")
    return tok


def download_ears(out_root: Path, n_speakers: int) -> None:
    d = out_root / "ears"
    if (d / ".done").exists():
        log(f"[skip] ears done ({d})"); return
    d.mkdir(parents=True, exist_ok=True)
    for i in range(1, n_speakers + 1):
        spk = f"p{i:03d}"
        z = d / f"{spk}.zip"
        log(f"[ears] ({i}/{n_speakers}) {spk}")
        if subprocess.run(["curl", "-sSL", f"{EARS_BASE}/{spk}.zip", "-o", str(z)]).returncode != 0 or not z.exists():
            log(f"[ears] WARNING failed {spk}"); continue
        subprocess.run(["unzip", "-o", "-q", str(z), "-d", str(d)])
        z.unlink(missing_ok=True)
    (d / ".done").write_text("ok\n")
    log(f"[done] ears -> {d}")


def export_hf(name: str, out_root: Path, token) -> None:
    repo, config, split = HF_DATASETS[name]
    d = out_root / name
    if (d / ".done").exists():
        log(f"[skip] {name} done ({d})"); return
    d.mkdir(parents=True, exist_ok=True)
    import soundfile as sf
    from datasets import load_dataset, Audio
    log(f"[hf-ds] {repo}:{config}:{split} -> {d}")
    ds = load_dataset(repo, config, split=split, token=token).cast_column("audio", Audio(decode=False))
    n = 0
    for i, row in enumerate(ds):
        a = row.get("audio") or {}
        b = a.get("bytes")
        if not b and a.get("path") and os.path.exists(a["path"]):
            b = open(a["path"], "rb").read()
        if not b:
            continue
        try:
            arr, sr = sf.read(io.BytesIO(b), dtype="float32", always_2d=False)
            if getattr(arr, "ndim", 1) > 1:
                arr = arr.mean(axis=1)
            sf.write(str(d / f"{i:06d}.wav"), arr, int(sr), subtype="PCM_16")
            n += 1
        except Exception as e:  # noqa: BLE001
            log(f"[hf-ds] skip {i}: {e}")
        if n and n % 2000 == 0:
            log(f"[hf-ds] {name}: {n}")
    (d / ".done").write_text("ok\n")
    log(f"[done] {name}: {n} wavs -> {d}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/data/clean48k")
    ap.add_argument("--ears-speakers", type=int, default=30)
    ap.add_argument("--sources", default="ears,expresso_read,expresso_conv")
    a = ap.parse_args()
    out = Path(a.out)
    if "/workspace" in str(out.resolve()):
        raise SystemExit("refusing /workspace — use / ")
    out.mkdir(parents=True, exist_ok=True)
    token = hf_login()
    for s in [x.strip() for x in a.sources.split(",") if x.strip()]:
        if s == "ears":
            download_ears(out, a.ears_speakers)
        elif s in HF_DATASETS:
            export_hf(s, out, token)
        else:
            log(f"[warn] unknown source {s}")
    total = sum(1 for _ in out.glob("**/*.wav"))
    log(f"[prepare_clean48k] done: {total} wav files under {out}")


if __name__ == "__main__":
    main()
