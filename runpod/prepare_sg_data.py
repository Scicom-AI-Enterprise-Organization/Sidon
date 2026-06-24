#!/usr/bin/env python3
"""Prepare the Singaporean-podcast subset as WebDataset shards for Sidon.

Runs ON THE POD. Everything under "/" (never /workspace). Mirrors the dataset
choice of ../neucodec-44k (malaysia-ai/singaporean-podcast-youtube) but adapts
it to what Sidon's online `WebDatasetDataModule` expects: raw *clean* audio
packaged as WebDataset `.tar` shards, with the audio stored under a field whose
name contains "audio" (the loader does `[k for k in keys if "audio" in k][0]`).

Pipeline:
  1. snapshot_download the split zip (sg-podcast.zip + .z01..z06) with HF_TOKEN,
     hf_transfer + hf_xet for speed.
  2. 7zz-extract to <root>/sg, then delete the archives.
  3. Take a CAPPED subset of the mp3s (this is a dry run — we don't need all
     1255 h), segment each into fixed-length 48 kHz mono FLAC chunks with one
     ffmpeg call per file, and pack the chunks into shards under
     <root>/sg_wds/{train,valid}. Audio field name = "audio.flac".
  4. Synthesize a small white-noise WebDataset under <root>/noise_wds so the
     online degradation pipeline (`add_non_parametric_noise`) has a noise source
     without depending on an external noise corpus.

Usage:
  python prepare_sg_data.py --data-root /data \
      [--max-files 60] [--valid-files 4] [--chunk-seconds 30] [--skip-download]
"""
from __future__ import annotations

import argparse
import glob
import io
import os
import shutil
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")

SG_REPO = "malaysia-ai/singaporean-podcast-youtube"
SG_PATTERNS = ["sg-podcast.z*", "sg-podcast.zip"]
SG_MAIN = "sg-podcast.zip"
TARGET_SR = 48_000


def log(m: str) -> None:
    print(m, flush=True)


def disk_free_gb(path: str) -> float:
    p = os.path.abspath(path)
    while p and not os.path.exists(p):
        p = os.path.dirname(p)
    return shutil.disk_usage(p or "/").free / 1e9


def find_7z() -> str:
    for cand in ("7zz", "7z", "7za"):
        if shutil.which(cand):
            return cand
    sys.exit("no 7-Zip binary found (need 7zz/7z) — run bootstrap.sh first")


def hf_login() -> str | None:
    tok = os.environ.get("HF_TOKEN")
    if tok:
        try:
            from huggingface_hub import login
            login(token=tok, add_to_git_credential=False)
            log("[hf] logged in with HF_TOKEN")
        except Exception as e:  # noqa: BLE001
            log(f"[hf] login warning: {e}")
    else:
        log("[hf] no HF_TOKEN set — downloading anonymously (slower / may rate-limit)")
    return tok


def download_and_extract(data_root: Path, token: str | None) -> Path:
    out_root = data_root / "sg"
    done = out_root / ".done"
    if done.exists():
        log(f"[skip] sg already extracted ({out_root})")
        return out_root
    archive_dir = data_root / "_archives" / "sg"
    archive_dir.mkdir(parents=True, exist_ok=True)
    from huggingface_hub import snapshot_download
    log(f"[dl] {SG_REPO} {SG_PATTERNS} -> {archive_dir}  free={disk_free_gb(str(data_root)):.0f}GB")
    snapshot_download(
        repo_id=SG_REPO, repo_type="dataset", allow_patterns=SG_PATTERNS,
        local_dir=str(archive_dir), token=token, max_workers=16,
    )
    sevenz = find_7z()
    main = archive_dir / SG_MAIN
    out_root.mkdir(parents=True, exist_ok=True)
    log(f"[7z] {sevenz} x {main.name} -> {out_root}  free={disk_free_gb(str(out_root)):.0f}GB")
    rc = subprocess.run([sevenz, "x", str(main), f"-o{out_root}", "-y", "-mmt8"]).returncode
    if rc != 0:
        sys.exit(f"[7z] extraction failed rc={rc} for {main}")
    shutil.rmtree(archive_dir, ignore_errors=True)
    done.write_text("ok\n")
    log(f"[done] sg extracted -> {out_root}  free={disk_free_gb(str(data_root)):.0f}GB")
    return out_root


def decode_to_wav(src: Path, tmp_dir: Path) -> Path | None:
    """Decode one mp3 -> a single proper 48k mono 16-bit WAV.

    Crucially NOT ffmpeg's `-f segment` muxer: that streams output and never
    backpatches the RIFF/data chunk sizes, so libsndfile can't seek the result
    (`psf_fseek() failed`) when the loader reads shard bytes from a BytesIO.
    A single-file output gets correct sizes; we then slice it in-process and
    re-encode each chunk via soundfile (guaranteed seekable), mirroring how the
    synthetic noise WAVs — which decode fine — are written.
    """
    out = tmp_dir / "full.wav"
    cmd = [
        "ffmpeg", "-v", "error", "-y", "-i", str(src),
        "-ar", str(TARGET_SR), "-ac", "1", "-c:a", "pcm_s16le", str(out),
    ]
    rc = subprocess.run(cmd).returncode
    if rc != 0 or not out.exists():
        log(f"[seg] WARNING ffmpeg rc={rc} on {src.name}")
        return None
    return out


def pack_split(files: list[Path], out_dir: Path, chunk_seconds: int,
               min_seconds: float, shard_maxcount: int) -> int:
    import soundfile as sf
    import webdataset as wds

    out_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    chunk_samples = int(chunk_seconds * TARGET_SR)
    min_samples = int(min_seconds * TARGET_SR)
    shard_pattern = str(out_dir / "sg-%06d.tar")
    with wds.ShardWriter(shard_pattern, maxcount=shard_maxcount) as sink:
        for i, src in enumerate(files, 1):
            with tempfile.TemporaryDirectory() as td:
                full = decode_to_wav(src, Path(td))
                if full is None:
                    continue
                try:
                    data, sr = sf.read(str(full), dtype="int16")  # mono -> 1-D
                except Exception as e:  # noqa: BLE001
                    log(f"[pack] skip {src.name}: {e}")
                    continue
                if data.ndim > 1:
                    data = data[:, 0]
                for start in range(0, len(data), chunk_samples):
                    chunk = data[start:start + chunk_samples]
                    if len(chunk) < min_samples:
                        continue
                    buf = io.BytesIO()
                    sf.write(buf, chunk, sr, format="WAV", subtype="PCM_16")
                    sink.write({"__key__": uuid.uuid4().hex, "audio.wav": buf.getvalue()})
                    written += 1
            log(f"[pack] ({i}/{len(files)}) {src.name} -> {written} chunks so far")
    log(f"[pack] {out_dir}: {written} chunks")
    return written


def make_noise(out_dir: Path, n: int, sr: int, shard_maxcount: int) -> None:
    import numpy as np
    import soundfile as sf
    import webdataset as wds

    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1234)
    shard_pattern = str(out_dir / "noise-%06d.tar")
    with wds.ShardWriter(shard_pattern, maxcount=shard_maxcount) as sink:
        for i in range(n):
            dur = float(rng.uniform(1.0, 3.0))
            x = rng.standard_normal(int(dur * sr)).astype("float32")
            # mild colouring: low-pass some, high-pass others, so SNR mixing varies
            if i % 3 == 0:
                x = np.cumsum(x)  # brown-ish
            elif i % 3 == 1:
                x = np.diff(x, prepend=0.0)  # blue-ish
            x = x / (np.abs(x).max() + 1e-7) * 0.5
            buf = io.BytesIO()
            sf.write(buf, x, sr, format="WAV", subtype="PCM_16")
            sink.write({"__key__": f"noise-{i:06d}", "audio.wav": buf.getvalue()})
    log(f"[noise] {out_dir}: {n} synthetic noise clips")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default="/data")
    ap.add_argument("--max-files", type=int, default=60,
                    help="cap on # of source podcasts to pack (dry run; <=0 = all)")
    ap.add_argument("--valid-files", type=int, default=4)
    ap.add_argument("--chunk-seconds", type=int, default=30)
    ap.add_argument("--min-seconds", type=float, default=2.0)
    ap.add_argument("--shard-maxcount", type=int, default=1000)
    ap.add_argument("--noise-clips", type=int, default=80)
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--download-only", action="store_true",
                    help="just download+extract sg into <root>/sg (for the direct mp3 loader); skip packing")
    a = ap.parse_args()

    data_root = Path(a.data_root)
    data_root.mkdir(parents=True, exist_ok=True)
    if "/workspace" in str(data_root.resolve()):
        sys.exit("refusing to use /workspace (network volume) — pass --data-root under /")

    token = hf_login()
    if not a.skip_download:
        sg_root = download_and_extract(data_root, token)
    else:
        sg_root = data_root / "sg"

    if a.download_only:
        n = len(glob.glob(str(sg_root / "**" / "*.mp3"), recursive=True))
        log(f"[download-only] sg extracted at {sg_root} ({n} mp3 files) — skipping packing")
        return

    mp3s = sorted(Path(p) for p in glob.glob(str(sg_root / "**" / "*.mp3"), recursive=True))
    if not mp3s:
        sys.exit(f"no mp3 files found under {sg_root}")
    log(f"[scan] {len(mp3s)} mp3 files under {sg_root}")
    if a.max_files > 0:
        mp3s = mp3s[: a.max_files]
        log(f"[scan] capped to {len(mp3s)} files for the dry run")

    valid = mp3s[-a.valid_files:] if a.valid_files > 0 else []
    train = mp3s[: len(mp3s) - len(valid)]
    log(f"[split] train={len(train)} files  valid={len(valid)} files")

    wds_root = data_root / "sg_wds"
    pack_split(train, wds_root / "train", a.chunk_seconds, a.min_seconds, a.shard_maxcount)
    pack_split(valid, wds_root / "valid", a.chunk_seconds, a.min_seconds, a.shard_maxcount)
    make_noise(data_root / "noise_wds", a.noise_clips, TARGET_SR, a.shard_maxcount)

    log("\n[prepare_sg_data] complete.")
    log(f"  train shards: {wds_root / 'train'}")
    log(f"  valid shards: {wds_root / 'valid'}")
    log(f"  noise shards: {data_root / 'noise_wds'}")


if __name__ == "__main__":
    main()
