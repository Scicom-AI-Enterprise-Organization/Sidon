#!/usr/bin/env bash
# Drive the Sidon dry-run ON THE POD: ensure the sg-podcast mp3s are present,
# then a short finetune that CONTINUES FROM sidon-v0.1's frozen feature extractor.
#
# Why this shape (verified): the released sidon-v0.1 weights ship as *frozen
# TorchScript* (state_dict empty — weights inlined as graph constants), so they
# cannot be loaded back into the trainable Sidon nn.Modules (build_ckpt_from_hf.py
# proves this and refuses to write random-init checkpoints). The feasible,
# faithful continuation — and what stage-3 finetuning does conceptually — is to
# keep v0.1's feature extractor FROZEN and train a fresh DAC decoder + GAN on top
# of its features. That is runpod/dryrun_v01_direct.py: a self-contained manual
# loop with a custom dataloader that reads random windows straight from the mp3s
# (ffmpeg -ss/-t) — no WebDataset, no shard packing. Everything under / .
#
# Tunables via env (defaults in brackets):
#   MAX_FILES [60]  STEPS [20]  BATCH [2]  WIN [10]  NUM_WORKERS [2]
set -euo pipefail

REPO=/Sidon
VENV=$REPO/.venv
cd "$REPO"

set -a; [ -f "$REPO/.env" ] && source "$REPO/.env"; set +a
export HF_HOME=/hf_cache
export HF_XET_HIGH_PERFORMANCE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
export HYDRA_FULL_ERROR=1
# CRITICAL on this 224-vCPU box (see ../neucodec-44k CLAUDE.md "CPU threads"):
# without these each dataloader worker spawns a ~224-thread pool -> oversubscribe
# -> GPU starves to ~1%. dryrun_v01_direct.py also sets these before importing
# numpy/torch and caps per-worker threads, but we set them here too.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

MAX_FILES=${MAX_FILES:-60}
STEPS=${STEPS:-20}
BATCH=${BATCH:-2}
WIN=${WIN:-10}
NUM_WORKERS=${NUM_WORKERS:-2}
CKPT_DIR=$REPO/ckpt_v0.1

mkdir -p /data /hf_cache "$REPO/runs" "$CKPT_DIR"

# ---- 1. data: download + extract sg mp3s (no packing; direct loader) --------
if [ ! -f /data/sg/.done ]; then
    echo "===== [run] downloading + extracting sg-podcast mp3s ====="
    "$VENV/bin/python" runpod/prepare_sg_data.py --data-root /data --download-only
else
    echo "===== [run] /data/sg already extracted — skipping download ====="
fi

# ---- 2. dry-run finetune: frozen v0.1 FE + fresh trainable DAC decoder ------
echo "===== [run] dry-run finetune from v0.1 FE (steps=$STEPS, batch=$BATCH, win=${WIN}s) ====="
exec "$VENV/bin/python" runpod/dryrun_v01_direct.py \
    --steps "$STEPS" \
    --batch "$BATCH" \
    --win "$WIN" \
    --max-files "$MAX_FILES" \
    --num-workers "$NUM_WORKERS" \
    --data-root /data/sg \
    --save "$CKPT_DIR/dryrun_decoder.pt"
