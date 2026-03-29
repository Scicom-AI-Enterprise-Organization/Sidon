#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
INFER_PY="${INFER_PY:-$ROOT_DIR/infer.py}"

INPUT_DIR="${INPUT_DIR:-/home/qch10240fz/nakata/github.com/sidon_eval/test_set_callfriend}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/qch10240fz/nakata/github.com/sidon_eval/results_callfriend}"
DEVICE="${DEVICE:-cuda:0}"
NUM_STEPS="${NUM_STEPS:-30}"
SIZES_CSV="${SIZES_CSV:-xsmall,small}"

declare -A CHECKPOINT_BY_SIZE=(
  [xsmall]="td6b8q85"
  [small]="jeamrwm0"
)

IFS=',' read -r -a SIZES <<< "$SIZES_CSV"

for size in "${SIZES[@]}"; do
  size="${size// /}"
  checkpoint_id="${CHECKPOINT_BY_SIZE[$size]:-}"

  if [[ -z "$checkpoint_id" ]]; then
    echo "Skipping unknown size: $size"
    continue
  fi

  checkpoint_path="$ROOT_DIR/sidon/$checkpoint_id"
  if [[ ! -d "$checkpoint_path" && ! -f "$checkpoint_path" ]]; then
    echo "Checkpoint path not found for size $size: $checkpoint_path"
    continue
  fi

  output_dir="$OUTPUT_ROOT/size_${size}"
  mkdir -p "$output_dir"

  echo "Running size=$size checkpoint_id=$checkpoint_id"
  "$PYTHON_BIN" "$INFER_PY" \
    --checkpoint "$checkpoint_path" \
    --input-dir "$INPUT_DIR" \
    --output-dir "$output_dir" \
    --device "$DEVICE" \
    --num-steps "$NUM_STEPS"
done
