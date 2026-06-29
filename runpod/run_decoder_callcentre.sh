#!/usr/bin/env bash
# Drive the Sidon call-centre DECODER stage ON THE POD: train a DAC decoder + GAN
# to reconstruct clean 48 kHz from the (frozen) call-centre FE's features. Reuses
# the clean data + FE checkpoint already on the pod. Resumable.
#
# Tunables (env): STEPS [50000] BATCH [4] WIN [8] LR [1e-4] NUM_WORKERS [8]
set -euo pipefail
REPO=/Sidon
VENV=$REPO/.venv
cd "$REPO"
set -a; [ -f "$REPO/.env" ] && source "$REPO/.env"; set +a
export HF_HOME=/hf_cache HF_XET_HIGH_PERFORMANCE=1 TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"   # for sidon.model.losses (DACLoss/GANLoss)
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

STEPS=${STEPS:-50000}; BATCH=${BATCH:-2}; ACCUM=${ACCUM:-4}; WIN=${WIN:-8}; LR=${LR:-1e-4}; NUM_WORKERS=${NUM_WORKERS:-8}
DEC_CHANNELS=${DEC_CHANNELS:-1536}; WANDB_NAME=${WANDB_NAME:-decoder-callcentre}
mkdir -p "$REPO/decoder_callcentre"

[ -f /Sidon/fe_callcentre/last.pt ] || { echo "missing FE checkpoint /Sidon/fe_callcentre/last.pt"; exit 1; }
[ -d /data/clean48k ] || { echo "missing /data/clean48k"; exit 1; }

echo "===== [run] decoder call-centre stage (steps=$STEPS batch=$BATCH win=${WIN}s) ====="
exec "$VENV/bin/python" runpod/train_decoder_callcentre.py \
    --data-root /data/clean48k --fe-ckpt /Sidon/fe_callcentre/last.pt \
    --steps "$STEPS" --batch "$BATCH" --accum "$ACCUM" --win "$WIN" --lr "$LR" \
    --dec-channels "$DEC_CHANNELS" --wandb-name "$WANDB_NAME" \
    --num-workers "$NUM_WORKERS" --out "$REPO/decoder_callcentre"
