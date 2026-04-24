#!/bin/bash
#PBS -l rt_QG=1
#PBS -l walltime=00:10:00
#PBS -j oe
#PBS -k oed
#PBS -W group_list=qgah50068
set -euo pipefail

cd "${PBS_O_WORKDIR:-$(pwd)}"
source /etc/profile.d/modules.sh
module load hpc_sdk/24.9 nvhpc-hpcx-cuda12/24.9

if [ -f .venv/bin/activate ]; then
  . .venv/bin/activate
fi

INPUT_VIDEO="${INPUT_VIDEO:-debate.mp4}"
CHECKPOINT="${CHECKPOINT:-sidon/yxdtp61x}"
OUTPUT_WAV="${OUTPUT_WAV:-output_separated.wav}"
DEVICE="${DEVICE:-cuda:0}"
NUM_STEPS="${NUM_STEPS:-30}"
CHUNK_SECONDS="${CHUNK_SECONDS:-20}"
OVERLAP_SECONDS="${OVERLAP_SECONDS:-5}"

.venv/bin/python infer.py \
  --checkpoint "$CHECKPOINT" \
  --input-video "$INPUT_VIDEO" \
  --output-wav "$OUTPUT_WAV" \
  --device "$DEVICE" \
  --num-steps "$NUM_STEPS" \
  --chunk-seconds "$CHUNK_SECONDS" \
  --overlap-seconds "$OVERLAP_SECONDS"
