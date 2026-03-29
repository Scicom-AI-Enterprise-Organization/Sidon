#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda:0}"
NUM_RUNS="${NUM_RUNS:-10}"  # 10 runs + 1 warmup

GENESES_CKPT="$ROOT_DIR/sidon/xvbpkrd1"
DIALOGUE_CKPT="$ROOT_DIR/sidon/yq7tva0l/checkpoints/epoch=3-step=556080.ckpt"

echo "=== Geneses 30 steps ==="
"$PYTHON_BIN" "$ROOT_DIR/eval_rtf.py" \
  --geneses-checkpoint "$GENESES_CKPT" \
  --num-steps 30 --num-runs "$NUM_RUNS" --device "$DEVICE"

echo "=== Geneses 100 steps ==="
"$PYTHON_BIN" "$ROOT_DIR/eval_rtf.py" \
  --geneses-checkpoint "$GENESES_CKPT" \
  --num-steps 100 --num-runs "$NUM_RUNS" --device "$DEVICE"

echo "=== DialogueSidon vae32 30 steps ==="
"$PYTHON_BIN" "$ROOT_DIR/eval_rtf.py" \
  --dialogue-checkpoint "$DIALOGUE_CKPT" \
  --num-steps 30 --num-runs "$NUM_RUNS" --device "$DEVICE"
