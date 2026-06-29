#!/usr/bin/env bash
# Drive the Sidon call-centre FE finetune ON THE POD: download clean 48k speech,
# then distill the 24-layer w2v-BERT + LoRA student (telephony-degraded) toward the
# frozen teacher (clean). Everything under / . Resumable.
#
# Tunables (env): STEPS [20000] BATCH [12] WIN [12] LR [2e-5] NUM_WORKERS [8]
#                 EARS_SPEAKERS [30] SOURCES [ears,expresso_read,expresso_conv]
set -euo pipefail
REPO=/Sidon
VENV=$REPO/.venv
cd "$REPO"
set -a; [ -f "$REPO/.env" ] && source "$REPO/.env"; set +a
export HF_HOME=/hf_cache
export HF_XET_HIGH_PERFORMANCE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

STEPS=${STEPS:-20000}
BATCH=${BATCH:-12}
WIN=${WIN:-12}
LR=${LR:-2e-5}
NUM_WORKERS=${NUM_WORKERS:-8}
EARS_SPEAKERS=${EARS_SPEAKERS:-30}
SOURCES=${SOURCES:-ears,expresso_read,expresso_conv}
mkdir -p /data /hf_cache "$REPO/fe_callcentre"

# ---- 1. clean data ----------------------------------------------------------
if [ ! -d /data/clean48k ] || [ -z "$(ls -A /data/clean48k 2>/dev/null)" ]; then
    echo "===== [run] downloading clean 48k speech ($SOURCES) ====="
    "$VENV/bin/python" runpod/prepare_clean48k.py --out /data/clean48k \
        --ears-speakers "$EARS_SPEAKERS" --sources "$SOURCES"
else
    echo "===== [run] /data/clean48k present — skipping data prep ====="
fi

# ---- 2. FE finetune ---------------------------------------------------------
echo "===== [run] FE call-centre finetune (steps=$STEPS batch=$BATCH win=${WIN}s) ====="
exec "$VENV/bin/python" runpod/train_fe_callcentre.py \
    --data-root /data/clean48k \
    --steps "$STEPS" --batch "$BATCH" --win "$WIN" --lr "$LR" \
    --num-workers "$NUM_WORKERS" --out "$REPO/fe_callcentre"
