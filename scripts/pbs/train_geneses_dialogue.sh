#!/bin/bash
#PBS -l rt_QF=2
#PBS -l walltime=48:00:00
#PBS -j oe
#PBS -k oed
#PBS -W group_list=qgah50068
set -euo pipefail

cd "${PBS_O_WORKDIR:-$(pwd)}"
source /etc/profile.d/modules.sh

module load hpc_sdk/24.9 nvhpc-hpcx-cuda12/24.9
export PATH="/work/gj18/e43001/miniconda3/bin/:$PATH"
export LD_LIBRARY_PATH="/work/gj18/e43001/miniconda3/lib/"
export MAIN_ADDR="$(hostname)"
export MAIN_PORT=$((10000 + RANDOM % 20000))

if [ -f .venv/bin/activate ]; then
  # shellcheck disable=SC1091
  . .venv/bin/activate
fi

num_gpus=$(nvidia-smi -L | wc -l)
num_nodes=$(sort -u "$PBS_NODEFILE" | wc -l)
num_procs=$((num_nodes * num_gpus))

BATCH_SIZE=${BATCH_SIZE:-8}
VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-2}
TRAIN_NUM_WORKERS=${TRAIN_NUM_WORKERS:-8}
VAL_NUM_WORKERS=${VAL_NUM_WORKERS:-4}
PRECISION=${PRECISION:-bf16-mixed}
export WANDB_NAME=${WANDB_NAME:-"geneses_dialogue_${PBS_JOBID}"}

unset OMPI_MCA_mca_base_env_list
mpirun \
  -x PATH \
  -x LD_LIBRARY_PATH \
  -x MAIN_ADDR \
  -x MAIN_PORT \
  -x WANDB_NAME \
  -bind-to none \
  -np "$num_procs" -map-by ppr:"$num_gpus":node -hostfile "$PBS_NODEFILE" \
  .venv/bin/python src/sidon/train.py \
  data=dialogue_preprocessed \
  data.datamodule.batch_size=${BATCH_SIZE} \
  data.datamodule.val_batch_size=${VAL_BATCH_SIZE} \
  data.datamodule.train_num_workers=${TRAIN_NUM_WORKERS} \
  data.datamodule.val_num_workers=${VAL_NUM_WORKERS} \
  model=geneses_dialogue \
  train=default \
  train.trainer.precision=${PRECISION} \
  train.trainer.gradient_clip_val=null \
  hydra.run.dir=./sidon_runs/${PBS_JOBID} \
  +train.trainer.num_nodes=$num_nodes +train.trainer.devices=$num_gpus
