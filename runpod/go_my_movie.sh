#!/usr/bin/env bash
# MY podcast + Malaysian movie only (SG already done+uploaded), at more workers for
# throughput. Same strict DNSMOS prep + 5GB-zip upload to HF. xet download.
set -u
cd /Sidon
set -a; [ -f /Sidon/.env ] && source /Sidon/.env; set +a
export HF_HOME=/hf_cache HF_XET_HIGH_PERFORMANCE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export NUMBA_CACHE_DIR=/data/.numba_cache; mkdir -p "$NUMBA_CACHE_DIR"
PY=/Sidon/.venv_dnsmos/bin/python
MAX_HOURS=${MAX_HOURS:-150}; BAK_THR=${BAK_THR:-3.644}; WORKERS=${WORKERS:-28}
UP_REPO=${UP_REPO:-Scicom-intl/sidon-callcentre-podcast}

"$PY" runpod/prepare_podcast_clean.py --repo malaysia-ai/malaysian-podcast-youtube \
    --main malaysian-podcast.zip --patterns "malaysian-podcast.z*,malaysian-podcast.zip" \
    --out /data/clean48k/podcast_my --work /data/_pod_my \
    --max-hours "$MAX_HOURS" --bak-thr "$BAK_THR" --workers "$WORKERS" \
    --upload-repo "$UP_REPO" --upload-name podcast_my

"$PY" runpod/prepare_podcast_clean.py --repo malaysia-ai/malaysian-movie-youtube \
    --patterns "part-*.zip" --standalone 1 \
    --out /data/clean48k/movie_my --work /data/_pod_movie \
    --max-hours "$MAX_HOURS" --bak-thr "$BAK_THR" --workers "$WORKERS" \
    --upload-repo "$UP_REPO" --upload-name movie_my

echo "MY_MOVIE_ALL_DONE"
