#!/usr/bin/env bash
# Build the strict-clean podcast teacher subsets (SG then MY, sequential to bound
# disk) into /data/clean48k/podcast_{sg,my}/. Uses the DNSMOS venv. Resumable
# (each --out gets a .done). ~150h source each, kept = bak>=3.644 chunks.
set -u
cd /Sidon
set -a; [ -f /Sidon/.env ] && source /Sidon/.env; set +a
export HF_HOME=/hf_cache HF_XET_HIGH_PERFORMANCE=1
PY=/Sidon/.venv_dnsmos/bin/python
MAX_HOURS=${MAX_HOURS:-150}; BAK_THR=${BAK_THR:-3.644}; WORKERS=${WORKERS:-16}

"$PY" runpod/prepare_podcast_clean.py --repo malaysia-ai/singaporean-podcast-youtube \
    --main sg-podcast.zip --patterns "sg-podcast.z*,sg-podcast.zip" \
    --out /data/clean48k/podcast_sg --work /data/_pod_sg \
    --max-hours "$MAX_HOURS" --bak-thr "$BAK_THR" --workers "$WORKERS"

"$PY" runpod/prepare_podcast_clean.py --repo malaysia-ai/malaysian-podcast-youtube \
    --main malaysian-podcast.zip --patterns "malaysian-podcast.z*,malaysian-podcast.zip" \
    --out /data/clean48k/podcast_my --work /data/_pod_my \
    --max-hours "$MAX_HOURS" --bak-thr "$BAK_THR" --workers "$WORKERS"

echo "PODCAST_ALL_DONE"
