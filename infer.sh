#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
INFER_PY="${INFER_PY:-$ROOT_DIR/infer.py}"

INPUT_DIR="${INPUT_DIR:-/home/qch10240fz/nakata/github.com/sidon_eval/test_set_opendialog/}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/qch10240fz/nakata/github.com/sidon_eval/results_opendialog}"
DEVICE="${DEVICE:-cuda:0}"
NUM_STEPS="${NUM_STEPS:-30}"
LATENT_SIZES_CSV="${LATENT_SIZES_CSV:-8,16,32,64,128}"

declare -A CHECKPOINT_BY_LATENT=(
  # IDs from the original infer.sh comments
  [128]="4jb721l4"
  [32]="yxdtp61x"
  [16]="yj6ti4ff"
  [8]="u6d8ph6t"
  [64]="daklmvvm"
)

IFS=',' read -r -a LATENT_SIZES <<< "$LATENT_SIZES_CSV"

for latent_size in "${LATENT_SIZES[@]}"; do
  latent_size="${latent_size// /}"
  checkpoint_id="${CHECKPOINT_BY_LATENT[$latent_size]:-}"

  if [[ -z "$checkpoint_id" ]]; then
    echo "Skipping unknown latent size: $latent_size"
    continue
  fi

  checkpoint_path="$ROOT_DIR/sidon/$checkpoint_id"
  if [[ ! -d "$checkpoint_path" && ! -f "$checkpoint_path" ]]; then
    echo "Checkpoint path not found for latent size $latent_size: $checkpoint_path"
    continue
  fi

  output_dir="$OUTPUT_ROOT/vae_${latent_size}"
  mkdir -p "$output_dir"

  echo "Running latent_size=$latent_size checkpoint_id=$checkpoint_id"
  "$PYTHON_BIN" "$INFER_PY" \
    --checkpoint "$checkpoint_path" \
    --input-dir "$INPUT_DIR" \
    --output-dir "$output_dir" \
    --device "$DEVICE" \
    --num-steps "$NUM_STEPS"
done
