#!/bin/bash
#PBS -W group_list=gj18
#PBS -j oe
set -euo pipefail

cd "${PBS_O_WORKDIR:-$(pwd)}"

if [ -f .venv/bin/activate ]; then
  # shellcheck disable=SC1091
  . .venv/bin/activate
fi

module load gcc >/dev/null 2>&1 || true
module load cuda >/dev/null 2>&1 || true

export CUDA_VISIBLE_DEVICES=0
export HF_HOME="/work/gj18/e43001/tmp/${PBS_JOBID}/huggingface"


split=train
data_language="${LANGUAGES}"

DEFAULT_OUTPUT_ROOT="/work/gj18/e43001/yodas2_sidon"
OUTPUT_ROOT="${OUTPUT_ROOT:-$DEFAULT_OUTPUT_ROOT}"
pattern_template="${OUTPUT_PATTERN_TEMPLATE:-${OUTPUT_PATTERN:-$OUTPUT_ROOT/$data_language/{split}-%05d.tar.gz}}"

SAMPLES_PER_SHARD="${SAMPLES_PER_SHARD:-5000}"
BATCH_SIZE="${BATCH_SIZE:-1}"
LOADER_WORKERS="${LOADER_WORKERS:-8}"
DECODER_PATH="${DECODER_PATH:-}"
DEVICE="${DEVICE:-cuda}"
CHUNK_SECONDS="${CHUNK_SECONDS:-20.0}"
TARGET_SAMPLE_RATE="${TARGET_SAMPLE_RATE:-24000}"
HF_CACHE_DIR="${HF_CACHE_DIR:-${HF_HOME}}"
LIMIT="${LIMIT:-}"
SKIP_ERRORS="${SKIP_ERRORS:-true}"
DRY_RUN="${DRY_RUN:-false}"
HF_TOKEN="${HF_TOKEN:-}"
AUDIO_COLUMN="${AUDIO_COLUMN:-audio}"
KEY_COLUMN="${KEY_COLUMN:-id}"
TEXT_COLUMN="${TEXT_COLUMN:-text}"
HF_DATASET="${HF_DATASET:-/work/gj18/e43001/.cache/huggingface/hub/datasets--espnet--yodas2/snapshots/c9674490249665d658f527e2684848377108d82c/yodas2.py}"
HF_NAME_OVERRIDE="${HF_NAME:-}"
S3_UPLOAD_EXTRA_ARGS="${S3_UPLOAD_EXTRA_ARGS:-}"
REMOVE_HF_CACHE="${REMOVE_HF_CACHE:-true}"


language_dir="$OUTPUT_ROOT/$data_language"
completion_marker="$language_dir/completed.txt"
if [ -f "$completion_marker" ]; then
  echo "Completion marker found for $data_language; skipping cleanse."
  exit 0
fi

mkdir -p "$language_dir"

output_pattern="$OUTPUT_ROOT/$data_language/$split-%05d.tar.gz"
mkdir -p "$(dirname "$output_pattern")"
cmd=(
  python scripts/examples/cleanse_mls.py
  --output-pattern "$output_pattern"
  --samples-per-shard "$SAMPLES_PER_SHARD"
  --batch-size "$BATCH_SIZE"
  --loader-workers "$LOADER_WORKERS"
  --device "$DEVICE"
  --chunk-seconds "$CHUNK_SECONDS"
  --target-sample-rate "$TARGET_SAMPLE_RATE"
  --hf-cache-dir "$HF_CACHE_DIR"
  --hf-dataset "$HF_DATASET"
  --hf-split "$split"
  --hf-name "${HF_NAME_OVERRIDE:-$data_language}"
  --audio-column "$AUDIO_COLUMN"
  --key-column "$KEY_COLUMN"
  --text-column "$TEXT_COLUMN"
  --num-shards 1
  --shard-index 0
  --check-safe-load true
)

printf 'Launching cleanser for %s %s:\n  %s\n' "$data_language" "$split" "${cmd[*]}"
"${cmd[@]}"


if [ -n "$S3_UPLOAD_URI" ]; then
  if ! command -v aws >/dev/null 2>&1; then
    echo "aws CLI is required for S3 uploads but was not found in PATH." >&2
    exit 1
  fi
  if [ -d "$language_dir" ]; then
    echo "Uploading cleansed shards from '$language_dir' to '$S3_UPLOAD_URI'..."
    aws s3 sync "$language_dir" "$S3_UPLOAD_URI" $S3_UPLOAD_EXTRA_ARGS
    echo "Removing local cleansed data at '$language_dir' after upload..."
    case "$language_dir" in
      ''|/|/.|..)
        echo "Refusing to remove directory '$language_dir'." >&2
        ;;
      *)
        rm -rf -- "$language_dir"
        ;;
    esac
  else
    echo "Language directory '$language_dir' does not exist, skipping S3 upload." >&2
  fi
else
  echo "S3_UPLOAD_URI not set; skipping upload." >&2
fi

if [ "$REMOVE_HF_CACHE" = true ] && [ -n "$HF_CACHE_DIR" ] && [ -d "$HF_CACHE_DIR" ]; then
  case "$HF_CACHE_DIR" in
    ''|/|/.|..)
      echo "Refusing to remove HF cache directory '$HF_CACHE_DIR'." >&2
      ;;
    *)
      echo "Removing Hugging Face cache at '$HF_CACHE_DIR'..."
      rm -rf -- "$HF_CACHE_DIR"
      ;;
  esac
fi
