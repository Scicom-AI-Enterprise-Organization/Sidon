#!/usr/bin/env python3
"""Download CLEAN 48 kHz speech for the Sidon call-centre FE finetune (teacher pool).

Sources (all genuinely clean studio/anechoic 48k; telephony degradation is simulated
at train time, so we only need clean references):
  - EARS    : facebookresearch anechoic 48k, per-speaker zips on GitHub (subset)
  - Expresso: ylacombe/expresso (read) + nytopop/expresso-conversational (conv), 48k studio
  - clean_extra : the DNSMOS-filtered ≥44 kHz HF datasets (clean_teacher_datasets.json,
                  bak ≥ baseline) — top-N cleanest, capped per dataset. See noisefilter/.

Everything under / (never /workspace). Each source drops a .done sentinel (resumable).
Audio lands as wav under /data/clean48k/<source>/ and the trainer globs it directly.

Usage:
  python prepare_clean48k.py --out /data/clean48k [--ears-speakers 30]
  python prepare_clean48k.py --sources clean_extra --clean-topn 60 --clean-max-clips 1500
"""
from __future__ import annotations

import argparse
import io
import json
import os
import re
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


def _open_stream(repo: str, token):
    """Open an HF dataset in streaming mode, discovering config + split robustly."""
    from datasets import load_dataset, get_dataset_config_names, get_dataset_split_names
    try:
        cfgs = get_dataset_config_names(repo, token=token)
    except Exception:  # noqa: BLE001
        cfgs = []
    cfg = cfgs[0] if cfgs else None
    try:
        splits = get_dataset_split_names(repo, cfg, token=token) if cfg else get_dataset_split_names(repo, token=token)
    except Exception:  # noqa: BLE001
        splits = []
    split = "train" if "train" in splits else (splits[0] if splits else "train")
    return load_dataset(repo, cfg, split=split, streaming=True, token=token), cfg, split


def _row_audio(a):
    """Extract (array, sr) from a streamed audio cell — decoded Audio dict OR raw bytes/path."""
    import soundfile as sf
    if isinstance(a, dict):
        if a.get("array") is not None and a.get("sampling_rate"):
            return a["array"], int(a["sampling_rate"])
        if a.get("bytes"):
            return sf.read(io.BytesIO(a["bytes"]), dtype="float32", always_2d=False)
        if a.get("path") and os.path.exists(a["path"]):
            return sf.read(a["path"], dtype="float32", always_2d=False)
    elif isinstance(a, (bytes, bytearray)):
        return sf.read(io.BytesIO(bytes(a)), dtype="float32", always_2d=False)
    return None, None


def export_hf_clean(entry: dict, out_root: Path, token, max_clips: int, min_sr: int = 44000) -> int:
    """Stream one clean HF dataset and write up to max_clips mono wavs (>= min_sr)."""
    import soundfile as sf
    import numpy as np
    repo = entry["id"]
    col = entry.get("audio_col") or "audio"
    safe = re.sub(r"[^A-Za-z0-9._-]", "__", repo)
    d = out_root / "extra" / safe
    if (d / ".done").exists():
        log(f"[skip] {repo} done ({d})"); return 0
    d.mkdir(parents=True, exist_ok=True)
    try:
        from datasets import Audio
        ds, cfg, split = _open_stream(repo, token)
        feats = getattr(ds, "features", None) or {}
        if col not in feats:  # fall back to any Audio-typed column
            col = next((k for k, v in feats.items() if type(v).__name__ == "Audio"), col)
        # decode=False -> raw bytes (we decode with soundfile); avoids the torchcodec
        # dependency datasets>=5 needs for auto-decoding Audio features.
        try:
            ds = ds.cast_column(col, Audio(decode=False))
        except Exception:  # noqa: BLE001 — not an Audio feature; _row_audio handles raw
            pass
        log(f"[clean] {repo} (cfg={cfg} split={split} col={col}) -> {d}")
    except Exception as e:  # noqa: BLE001
        log(f"[clean] {repo}: open failed: {type(e).__name__}: {str(e)[:120]}"); return 0
    n = 0
    try:
        for i, row in enumerate(ds):
            if n >= max_clips:
                break
            try:
                arr, sr = _row_audio(row.get(col))
            except Exception:  # noqa: BLE001
                continue
            if arr is None or not sr or sr < min_sr:
                continue
            arr = np.asarray(arr, dtype="float32")
            if arr.ndim > 1:
                arr = arr.mean(axis=1)
            if len(arr) < int(0.5 * sr) or not np.isfinite(arr).all():
                continue
            try:
                sf.write(str(d / f"{i:07d}.wav"), arr, int(sr), subtype="PCM_16")
                n += 1
            except Exception:  # noqa: BLE001
                continue
            if n % 1000 == 0:
                log(f"[clean] {repo}: {n}")
    except Exception as e:  # noqa: BLE001
        log(f"[clean] {repo}: stream error after {n}: {type(e).__name__}: {str(e)[:120]}")
    (d / ".done").write_text(f"{n}\n")
    log(f"[done] {repo}: {n} wavs -> {d}")
    return n


def download_clean_extra(out_root: Path, token, manifest: str, topn: int, max_clips: int, min_sr: int) -> None:
    p = Path(manifest)
    if not p.is_absolute():
        p = Path(__file__).resolve().parent / manifest
    if not p.exists():
        log(f"[clean_extra] manifest not found: {p}"); return
    rows = json.load(open(p)).get("datasets", [])
    rows = sorted(rows, key=lambda r: -(r.get("bak") or 0))
    if topn > 0:
        rows = rows[:topn]
    log(f"[clean_extra] {len(rows)} datasets (top {topn or 'all'} by bak), <= {max_clips} clips each, >= {min_sr} Hz")
    total = 0
    for j, e in enumerate(rows, 1):
        log(f"[clean_extra] ({j}/{len(rows)}) bak={e.get('bak')} {e['id']}")
        total += export_hf_clean(e, out_root, token, max_clips, min_sr)
    log(f"[clean_extra] total {total} wavs from {len(rows)} datasets")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/data/clean48k")
    ap.add_argument("--ears-speakers", type=int, default=30)
    ap.add_argument("--sources", default="ears,expresso_read,expresso_conv")
    ap.add_argument("--clean-manifest", default="clean_teacher_datasets.json")
    ap.add_argument("--clean-topn", type=int, default=60, help="how many cleanest datasets (0=all)")
    ap.add_argument("--clean-max-clips", type=int, default=1500, help="cap per dataset")
    ap.add_argument("--clean-min-sr", type=int, default=44000)
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
        elif s == "clean_extra":
            download_clean_extra(out, token, a.clean_manifest, a.clean_topn, a.clean_max_clips, a.clean_min_sr)
        else:
            log(f"[warn] unknown source {s}")
    total = sum(1 for _ in out.glob("**/*.wav"))
    log(f"[prepare_clean48k] done: {total} wav files under {out}")


if __name__ == "__main__":
    main()
    # HF streaming/torch spawn background threads whose teardown can race the
    # interpreter finalizer (PyGILState_Release fatal); work is done + logs flushed,
    # so exit hard to skip that noisy (non-zero) shutdown.
    os._exit(0)
