#!/bin/sh
#PBS -q rt_HF
#PBS -l select=1
#PBS -l walltime=72:00:00
#PBS -P gag51394
#PBS -j oe
#PBS -k oed

set -euo pipefail

cd "${PBS_O_WORKDIR:-$(pwd)}"

if [ -f .venv/bin/activate ]; then
  # shellcheck disable=SC1091
  . .venv/bin/activate
fi
module load hpcx-mt/2.20
export PATH="/home/acc12576tt/miniconda3/bin/:$PATH"
export LD_LIBRARY_PATH="/home/acc12576tt/miniconda3/lib/:$LD_LIBRARY_PATH"
export MAIN_ADDR="$(hostname)"
export MAIN_PORT=$((10000 + RANDOM % 20000))
export HYDRA_FULL_ERROR=1

uv run python src/sidon/train.py \
  data=preprocessed \
  data.datamodule.batch_size=32 \
  model=sidon_feature_predictor \
  train=default \
  train.trainer.precision=bf16-mixed \
  "train.ckpt_path='/home/acc12576tt/github.com/Sidon/sidon/t2h90k5j/checkpoints/epoch=13-step=388305.ckpt'" \
  hydra.run.dir=/groups/gag51394/users/nakata/sidon_runs/${PBS_JOBID}
