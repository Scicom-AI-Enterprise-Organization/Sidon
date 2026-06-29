#!/usr/bin/env bash
# Server-side one-shot: stop the (now-superseded) decoder and FRESH-train the stage-1
# FE with the realistic Degrader + the expanded teacher corpus (/data/clean48k incl.
# clean_extra). Data is already on disk, so we skip prepare and call the trainer
# directly. Fresh output dir (keeps the old fe_callcentre/last.pt intact).
set -u
pkill -9 -f train_decoder_callcentre 2>/dev/null   # decoder built on OLD FE -> superseded
pkill -9 -f run_decoder 2>/dev/null
pkill -9 -f train_fe_callcentre 2>/dev/null
sleep 3
cd /Sidon
set -a; [ -f /Sidon/.env ] && source /Sidon/.env; set +a
export HF_HOME=/hf_cache HF_XET_HIGH_PERFORMANCE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
# More workers than before: the realistic Degrader (GSM/MP3 encode + soxr/scipy) is
# heavier per clip than the old telephony_degrade.
exec /Sidon/.venv/bin/python runpod/train_fe_callcentre.py \
    --data-root /data/clean48k \
    --steps ${STEPS:-30000} --batch ${BATCH:-12} --win ${WIN:-12} \
    --lr ${LR:-2e-5} --warmup ${WARMUP:-2000} --num-workers ${NUM_WORKERS:-16} \
    --out /Sidon/fe_callcentre_realdeg \
    --wandb-name ${WANDB_NAME:-fe-callcentre-realdeg}
