#!/usr/bin/env python3
"""Add a Malaysia-AI podcast to the Sidon clean teacher pool, STRICTLY filtered.

For ONE podcast repo (split-zip archive): download it, SELECTIVELY extract ~max_hours
of mp3s (size budget — never extract the full 70-126 GB), delete the zip, then for each
mp3: ffmpeg-decode -> 48 kHz mono -> chunk (chunk_s, non-overlapping) -> DNSMOS `bak`
-> keep chunks with bak >= thr as 48 kHz wav under out_dir (mp3 deleted as processed,
so disk stays bounded). Same studio-clean bar as EARS/Expresso/clean_extra, so noisy /
music / overlapping-speech podcast chunks are dropped.

Venv needs: huggingface_hub, soundfile, soxr, numpy, speechmos, onnxruntime
(dnsmos_metric.clip_dnsmos). Needs `7z` (p7zip-full) + `ffmpeg` on PATH.

Usage:
  python prepare_podcast_clean.py --repo malaysia-ai/singaporean-podcast-youtube \
     --main sg-podcast.zip --patterns "sg-podcast.z*,sg-podcast.zip" \
     --out /data/clean48k/podcast_sg --max-hours 150 --bak-thr 3.644 --workers 16
"""
import argparse
import glob
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dnsmos_metric import clip_dnsmos

SR = 48000


def log(m):
    print(m, flush=True)


def find_7z():
    for c in ("7z", "7za", "7zr"):
        if shutil.which(c):
            return c
    raise SystemExit("7z not found — apt-get install -y p7zip-full")


def decode_whole(src, sr=SR):
    """ffmpeg-decode an mp3 -> 48 kHz mono float32 (one temp wav per file)."""
    fd, tmp = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        rc = subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src),
                             "-ar", str(sr), "-ac", "1", "-c:a", "pcm_s16le", tmp]).returncode
        if rc != 0:
            return None
        x, _ = sf.read(tmp, dtype="float32", always_2d=False)
        return x if getattr(x, "ndim", 1) == 1 else x.mean(axis=1)
    except Exception:  # noqa: BLE001
        return None
    finally:
        try:
            os.unlink(tmp)
        except Exception:  # noqa: BLE001
            pass


def process_file(args):
    src, out_dir, chunk_s, thr, max_chunks = args
    x = decode_whole(src)
    try:
        os.unlink(src)  # disk mgmt: drop the mp3 once decoded
    except Exception:  # noqa: BLE001
        pass
    cs = int(chunk_s * SR)
    if x is None or len(x) < cs:
        return (0, 0)
    base = os.path.splitext(os.path.basename(src))[0]
    n = len(x) // cs
    kept = 0
    for k in range(min(n, max_chunks)):
        seg = x[k * cs:(k + 1) * cs]
        if float(np.abs(seg).max()) < 1e-4:
            continue
        r = clip_dnsmos(seg, SR)              # resamples to 16k internally
        if r is None or r["bak"] < thr:
            continue
        try:
            sf.write(os.path.join(out_dir, f"{base}_{k:04d}.wav"), seg, SR, subtype="PCM_16")
            kept += 1
        except Exception:  # noqa: BLE001
            continue
    return (kept, n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--main", required=True)        # e.g. sg-podcast.zip
    ap.add_argument("--patterns", required=True)    # e.g. "sg-podcast.z*,sg-podcast.zip"
    ap.add_argument("--out", required=True)         # /data/clean48k/podcast_sg
    ap.add_argument("--work", default="/data/_pod")
    ap.add_argument("--max-hours", type=float, default=150.0)
    ap.add_argument("--chunk-s", type=float, default=15.0)
    ap.add_argument("--bak-thr", type=float, default=3.644)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--max-chunks-per-file", type=int, default=240)
    ap.add_argument("--upload-repo", default="", help="HF dataset repo to tar+upload the clean chunks into")
    ap.add_argument("--upload-name", default="", help="tar filename in the repo (default <out-basename>.tar)")
    ap.add_argument("--free-after-upload", type=int, default=1, help="rm local chunks after upload (bound disk)")
    a = ap.parse_args()
    tok = os.environ.get("HF_TOKEN")
    os.makedirs(a.out, exist_ok=True)
    if os.path.exists(os.path.join(a.out, ".done")):
        log(f"[skip] {a.repo} already done ({a.out})"); return
    arch = os.path.join(a.work, "arch")
    ext = os.path.join(a.work, "ext")
    shutil.rmtree(a.work, ignore_errors=True)
    os.makedirs(arch, exist_ok=True); os.makedirs(ext, exist_ok=True)

    # 1. download the split-zip parts
    from huggingface_hub import snapshot_download
    pats = [p.strip() for p in a.patterns.split(",")]
    log(f"[dl] {a.repo} {pats} -> {arch}")
    snapshot_download(repo_id=a.repo, repo_type="dataset", allow_patterns=pats,
                      local_dir=arch, token=tok, max_workers=16)
    sevenz = find_7z()
    main = os.path.join(arch, a.main)

    # 2. extract until ~max_hours of mp3 is on disk, then STOP (7z writes whole files
    # in archive order, so we just kill it once enough is extracted). This bounds disk
    # (the full archive is 70-126 GB) and avoids the unicode-listfile selective-extract
    # bug. The zip is deleted immediately after to free space.
    def _dirbytes(d):
        t = 0
        for r, _, fs in os.walk(d):
            for f in fs:
                try:
                    t += os.path.getsize(os.path.join(r, f))
                except Exception:  # noqa: BLE001
                    pass
        return t

    budget = int(a.max_hours * 60 * 1.3 * 1e6)   # mp3 ~1 MB/min, *1.3 safety
    log(f"[7z] extracting up to ~{budget/1e9:.1f} GB (~{a.max_hours}h) from {a.main} ...")
    p7 = subprocess.Popen([sevenz, "x", main, f"-o{ext}", "-y", "-mmt8"],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    while p7.poll() is None:
        time.sleep(5)
        if _dirbytes(ext) >= budget:
            p7.terminate()
            try:
                p7.wait(timeout=30)
            except Exception:  # noqa: BLE001
                p7.kill()
            break
    shutil.rmtree(arch, ignore_errors=True)      # free the 70-126 GB zip now

    # 3. collect extracted mp3s (already ~budget; the last one may be partial -> ffmpeg
    # just skips undecodable ones)
    mp3s = [p for p in glob.glob(os.path.join(ext, "**", "*"), recursive=True)
            if p.lower().endswith(".mp3")]
    log(f"[sel] extracted {len(mp3s)} mp3s (~{_dirbytes(ext)/1e9:.1f} GB)")
    if not mp3s:
        raise SystemExit("no mp3 extracted")

    # 4. chunk + DNSMOS-filter each mp3 (parallel; mp3 deleted as processed)
    log(f"[proc] {len(mp3s)} mp3s -> chunk {a.chunk_s}s + DNSMOS bak>={a.bak_thr}")
    t0, done, kept_tot, seen_tot = time.time(), 0, 0, 0
    CHUNK = 64
    for ci in range(0, len(mp3s), CHUNK):
        batch = mp3s[ci:ci + CHUNK]
        try:
            with ProcessPoolExecutor(max_workers=a.workers, max_tasks_per_child=8) as ex:
                futs = [ex.submit(process_file, (p, a.out, a.chunk_s, a.bak_thr, a.max_chunks_per_file))
                        for p in batch]
                for fut in as_completed(futs):
                    kept, seen = fut.result()
                    kept_tot += kept; seen_tot += seen; done += 1
                    if done % 50 == 0 or done == len(mp3s):
                        log(f"[proc] {done}/{len(mp3s)} files | kept {kept_tot} chunks "
                            f"({100*kept_tot/max(1,seen_tot):.0f}% of {seen_tot}) "
                            f"~{kept_tot*a.chunk_s/3600:.1f}h clean | {(time.time()-t0)/max(1,done):.2f}s/file")
        except Exception as e:  # noqa: BLE001
            log(f"[proc] batch {ci//CHUNK} pool error: {type(e).__name__} — continuing")
    shutil.rmtree(a.work, ignore_errors=True)
    open(os.path.join(a.out, ".done"), "w").write(f"{kept_tot}\n")
    log(f"[done] {a.repo}: kept {kept_tot} clean chunks (~{kept_tot*a.chunk_s/3600:.1f}h) -> {a.out}")

    # 5. tar + upload the clean chunks to HF so they persist (CPU pod is disposable),
    # then free local space so the next podcast fits the 160 GB disk.
    if a.upload_repo and kept_tot > 0:
        from huggingface_hub import HfApi
        tarname = a.upload_name or (os.path.basename(a.out.rstrip("/")) + ".tar")
        tarp = os.path.join(os.path.dirname(a.out.rstrip("/")), tarname)
        log(f"[upload] tar {a.out} -> {tarp}")
        subprocess.run(["tar", "-cf", tarp, "-C", os.path.dirname(a.out.rstrip("/")),
                        os.path.basename(a.out.rstrip("/"))])
        api = HfApi(token=os.environ.get("HF_TOKEN"))
        api.create_repo(repo_id=a.upload_repo, repo_type="dataset", exist_ok=True)
        api.upload_file(path_or_fileobj=tarp, path_in_repo=tarname,
                        repo_id=a.upload_repo, repo_type="dataset",
                        commit_message=f"{tarname}: {kept_tot} clean chunks (~{kept_tot*a.chunk_s/3600:.1f}h) from {a.repo}")
        log(f"[upload] -> {a.upload_repo}/{tarname}")
        try:
            os.unlink(tarp)
        except Exception:  # noqa: BLE001
            pass
        if a.free_after_upload:
            shutil.rmtree(a.out, ignore_errors=True)
            log(f"[upload] freed local {a.out}")


if __name__ == "__main__":
    main()
    os._exit(0)
