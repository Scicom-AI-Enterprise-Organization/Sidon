#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
INFER_PY="${INFER_PY:-$ROOT_DIR/infer_geneses.py}"

CHECKPOINT="${CHECKPOINT:-$ROOT_DIR/sidon/xvbpkrd1}"
INPUT_DIR="${INPUT_DIR:-/home/qch10240fz/nakata/github.com/sidon_eval/test_set_opendialog/}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/qch10240fz/nakata/github.com/sidon_eval/results_opendialog/geneses}"
DEVICE="${DEVICE:-cuda:0}"
NUM_STEPS="${NUM_STEPS:-100}"
CHUNK_SECONDS="${CHUNK_SECONDS:-20}"
OVERLAP_SECONDS="${OVERLAP_SECONDS:-5}"

if [[ ! -f "$INFER_PY" ]]; then
  echo "Inference script not found: $INFER_PY"
  exit 1
fi

if [[ ! -e "$CHECKPOINT" ]]; then
  echo "Checkpoint path not found: $CHECKPOINT"
  echo "Set CHECKPOINT to a .ckpt file or checkpoint directory."
  exit 1
fi

if [[ ! -d "$INPUT_DIR" ]]; then
  echo "Input directory not found: $INPUT_DIR"
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

"$PYTHON_BIN" "$INFER_PY" \
  --checkpoint "$CHECKPOINT" \
  --input-dir "$INPUT_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --device "$DEVICE" \
  --num-steps "$NUM_STEPS" \
  --chunk-seconds "$CHUNK_SECONDS" \
  --overlap-seconds "$OVERLAP_SECONDS"
