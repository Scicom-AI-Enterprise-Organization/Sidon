#!/usr/bin/env bash
# Build the strict-clean podcast teacher subsets (SG then MY, sequential to bound
# disk) into /data/clean48k/podcast_{sg,my}/. Uses the DNSMOS venv. Resumable
# (each --out gets a .done). ~150h source each, kept = bak>=3.644 chunks.
set -u
cd /Sidon
set -a; [ -f /Sidon/.env ] && source /Sidon/.env; set +a
export HF_HOME=/hf_cache HF_HUB_DISABLE_XET=1   # xet can leave broken pointers on upload
PY=/Sidon/.venv_dnsmos/bin/python
MAX_HOURS=${MAX_HOURS:-150}; BAK_THR=${BAK_THR:-3.644}; WORKERS=${WORKERS:-16}
UP_REPO=${UP_REPO:-Scicom-intl/sidon-callcentre-podcast}   # clean chunks persist here (tar per podcast)

# SG then MY; each tar+uploads its clean chunks to HF and frees local disk so the
# next podcast's 70-126 GB archive fits the 160 GB CPU-pod disk.
"$PY" runpod/prepare_podcast_clean.py --repo malaysia-ai/singaporean-podcast-youtube \
    --main sg-podcast.zip --patterns "sg-podcast.z*,sg-podcast.zip" \
    --out /data/clean48k/podcast_sg --work /data/_pod_sg \
    --max-hours "$MAX_HOURS" --bak-thr "$BAK_THR" --workers "$WORKERS" \
    --upload-repo "$UP_REPO" --upload-name podcast_sg.tar

"$PY" runpod/prepare_podcast_clean.py --repo malaysia-ai/malaysian-podcast-youtube \
    --main malaysian-podcast.zip --patterns "malaysian-podcast.z*,malaysian-podcast.zip" \
    --out /data/clean48k/podcast_my --work /data/_pod_my \
    --max-hours "$MAX_HOURS" --bak-thr "$BAK_THR" --workers "$WORKERS" \
    --upload-repo "$UP_REPO" --upload-name podcast_my.tar

echo "PODCAST_ALL_DONE"
