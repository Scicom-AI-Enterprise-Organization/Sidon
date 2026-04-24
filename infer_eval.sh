#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"

EVAL_ROOT="${EVAL_ROOT:-/home/qch10240fz/nakata/github.com/sidon_eval}"

DATASETS=(
  "test_set_callfriend:results_callfriend"
  "test_set_switchboard:results_switchboard"
)

for entry in "${DATASETS[@]}"; do
  input_dir="$EVAL_ROOT/${entry%%:*}"
  output_root="$EVAL_ROOT/${entry##*:}"

  echo "=== Dataset: $input_dir ==="

  PYTHON_BIN="$PYTHON_BIN" \
    INPUT_DIR="$input_dir" \
    OUTPUT_ROOT="$output_root" \
    bash "$ROOT_DIR/infer.sh"

  PYTHON_BIN="$PYTHON_BIN" \
    INPUT_DIR="$input_dir" \
    OUTPUT_ROOT="$output_root" \
    bash "$ROOT_DIR/infer_size_ablation.sh"

  PYTHON_BIN="$PYTHON_BIN" \
    INPUT_DIR="$input_dir" \
    OUTPUT_DIR="$output_root/geneses" \
    bash "$ROOT_DIR/infer_geneses.sh"
done
