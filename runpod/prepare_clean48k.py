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


def _open_stream(repo: str, token, config=None, split=None):
    """Open an HF dataset in streaming mode. Uses explicit config/split when given,
    else discovers them; falls back to discovery if an explicit choice fails."""
    from datasets import load_dataset, get_dataset_config_names, get_dataset_split_names

    def _discover():
        try:
            cfgs = get_dataset_config_names(repo, token=token)
        except Exception:  # noqa: BLE001
            cfgs = []
        c = config if config is not None else (cfgs[0] if cfgs else None)
        try:
            sp = get_dataset_split_names(repo, c, token=token) if c else get_dataset_split_names(repo, token=token)
        except Exception:  # noqa: BLE001
            sp = []
        s = split if split is not None else ("train" if "train" in sp else (sp[0] if sp else "train"))
        return c, s

    c, s = _discover()
    try:
        return load_dataset(repo, c, split=s, streaming=True, token=token), c, s
    except Exception:  # noqa: BLE001 — explicit config/split bad -> rediscover from scratch
        config = split = None
        c, s = _discover()
        return load_dataset(repo, c, split=s, streaming=True, token=token), c, s


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
    tag = entry.get("tag") or safe
    d = out_root / entry.get("subroot", "extra") / tag
    if (d / ".done").exists():
        log(f"[skip] {repo} done ({d})"); return 0
    d.mkdir(parents=True, exist_ok=True)
    try:
        from datasets import Audio
        ds, cfg, split = _open_stream(repo, token, entry.get("config"), entry.get("split"))
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


def download_clean_extra(out_root: Path, token, manifest: str, topn: int, max_clips: int,
                         min_sr: int, shard: str = "0/1") -> None:
    p = Path(manifest)
    if not p.is_absolute():
        p = Path(__file__).resolve().parent / manifest
    if not p.exists():
        log(f"[clean_extra] manifest not found: {p}"); return
    rows = json.load(open(p)).get("datasets", [])
    rows = sorted(rows, key=lambda r: -(r.get("bak") or 0))
    if topn > 0:
        rows = rows[:topn]
    # shard "i/N": this process takes the strided slice rows[i::N] (per-dataset dirs are
    # disjoint, so several sharded processes run in parallel without conflict).
    si, ns = (int(x) for x in shard.split("/"))
    rows = rows[si::ns]
    log(f"[clean_extra] shard {si}/{ns}: {len(rows)} datasets (of top {topn or 'all'} by bak), "
        f"<= {max_clips} clips each, >= {min_sr} Hz")
    total = 0
    for j, e in enumerate(rows, 1):
        log(f"[clean_extra] ({j}/{len(rows)}) bak={e.get('bak')} {e['id']}")
        total += export_hf_clean(e, out_root, token, max_clips, min_sr)
    log(f"[clean_extra] total {total} wavs from {len(rows)} datasets")


# Sidon-paper clean teacher corpora (TRUSTED clean -> no DNSMOS filter). Several are
# 24 kHz (libritts-r/fleurs-r/jvs) — fine for the 16 kHz FE; upsampled for the 48k
# decoder (band-limited >12 kHz, accepted per the paper). VCTK excluded (not clean
# enough, per user). Configs/splits are best-effort; _open_stream rediscovers on a
# bad choice. Capped per entry to keep the corpus balanced + bounded.
SIDON_TEACHERS = [
    {"id": "mythicinfinity/libritts_r", "config": "clean", "split": "train.clean.100", "tag": "libritts_r_clean100", "max_clips": 4000},  # 24k en
    {"id": "mythicinfinity/libritts_r", "config": "clean", "split": "train.clean.360", "tag": "libritts_r_clean360", "max_clips": 4000},  # 24k en
    {"id": "PrincePK/jvs_ver1", "tag": "jvs", "max_clips": 4000},                                  # 24k ja
    {"id": "japanese-asr/ja_asr.jsut_basic5000", "config": "default", "tag": "jsut", "max_clips": 5000},  # ja
    {"id": "google/fleurs-r", "config": "en_us", "tag": "fleurs_r_en", "max_clips": 2500},         # 24k en
    {"id": "google/fleurs-r", "config": "ja_jp", "tag": "fleurs_r_ja", "max_clips": 2000},
    {"id": "google/fleurs-r", "config": "ms_my", "tag": "fleurs_r_ms", "max_clips": 2500},         # Malay!
    {"id": "google/fleurs-r", "config": "cmn_hans_cn", "tag": "fleurs_r_zh", "max_clips": 2000},
    {"id": "google/fleurs-r", "config": "ta_in", "tag": "fleurs_r_ta", "max_clips": 1500},         # Tamil
    {"id": "google/fleurs-r", "config": "yo_ng", "tag": "fleurs_r_yo", "max_clips": 1500},
]


def download_sidon_teachers(out_root: Path, token, min_sr: int) -> None:
    log(f"[sidon] {len(SIDON_TEACHERS)} teacher corpora (trusted clean, no DNSMOS; min_sr {min_sr})")
    total = 0
    for e in SIDON_TEACHERS:
        e = dict(e); e["subroot"] = "sidon"
        total += export_hf_clean(e, out_root, token, e.get("max_clips", 4000), min_sr=min_sr)
    log(f"[sidon] total {total} wavs under {out_root}/sidon")


# BibleTTS (OpenSLR SLR129): genuine 48 kHz/24-bit FLAC studio. Per-language tarballs;
# open download (no token). Subset to keep disk bounded.
BIBLETTS_BASE = "https://www.openslr.org/resources/129"
BIBLETTS_LANGS = ["hausa", "yoruba"]


def download_bibletts(out_root: Path, langs) -> None:
    import tarfile
    d = out_root / "sidon" / "bibletts"
    if (d / ".done").exists():
        log(f"[skip] bibletts done ({d})"); return
    d.mkdir(parents=True, exist_ok=True)
    for lang in langs:
        url = f"{BIBLETTS_BASE}/{lang}.tar.gz"
        tgz = d / f"{lang}.tar.gz"
        log(f"[bibletts] {url}")
        if subprocess.run(["curl", "-sSL", url, "-o", str(tgz)]).returncode != 0 or not tgz.exists():
            log(f"[bibletts] WARNING download failed {lang}"); continue
        try:
            with tarfile.open(str(tgz)) as t:
                t.extractall(str(d))
        except Exception as e:  # noqa: BLE001
            log(f"[bibletts] extract fail {lang}: {e}")
        tgz.unlink(missing_ok=True)
    (d / ".done").write_text("ok\n")
    log(f"[done] bibletts -> {d}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/data/clean48k")
    ap.add_argument("--ears-speakers", type=int, default=30)
    ap.add_argument("--sources", default="ears,expresso_read,expresso_conv")
    ap.add_argument("--clean-manifest", default="clean_teacher_datasets.json")
    ap.add_argument("--clean-topn", type=int, default=60, help="how many cleanest datasets (0=all)")
    ap.add_argument("--clean-max-clips", type=int, default=1500, help="cap per dataset")
    ap.add_argument("--clean-min-sr", type=int, default=44000)
    ap.add_argument("--clean-shard", default="0/1", help="i/N strided slice for parallel downloads")
    ap.add_argument("--sidon-min-sr", type=int, default=16000,
                    help="min SR for sidon_teachers (24k corpora ok for the 16k FE)")
    ap.add_argument("--bibletts-langs", default=",".join(BIBLETTS_LANGS))
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
            download_clean_extra(out, token, a.clean_manifest, a.clean_topn, a.clean_max_clips,
                                 a.clean_min_sr, a.clean_shard)
        elif s == "sidon_teachers":
            download_sidon_teachers(out, token, a.sidon_min_sr)
        elif s == "bibletts":
            download_bibletts(out, [x.strip() for x in a.bibletts_langs.split(",") if x.strip()])
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
