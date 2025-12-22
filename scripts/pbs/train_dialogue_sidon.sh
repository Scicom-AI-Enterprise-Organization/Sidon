#!/bin/bash
#PBS -q regular-g
#PBS -l select=16
#PBS -W group_list=gj18
#PBS -j oe
#PBS -k oed
_module_raw () {
        eval "$(/usr/bin/tclsh '/usr/share/Modules/libexec/modulecmd.tcl' zsh "$@")"
        _mlstatus=$? 
        return $_mlstatus
}

module () {
        local _mlredir=1 
        if [ -n "${MODULES_REDIRECT_OUTPUT+x}" ]
        then
                if [ "$MODULES_REDIRECT_OUTPUT" = '0' ]
                then
                        _mlredir=0 
                elif [ "$MODULES_REDIRECT_OUTPUT" = '1' ]
                then
                        _mlredir=1 
                fi
        fi
        case " $@ " in
                (*' --no-redirect '*) _mlredir=0  ;;
                (*' --redirect '*) _mlredir=1  ;;
        esac
        if [ $_mlredir -eq 0 ]
        then
                _module_raw "$@"
        else
                _module_raw "$@" 2>&1
        fi
}
set -euo pipefail
module unload nvidia
module load gcc

cd "${PBS_O_WORKDIR:-$(pwd)}"

export PATH="/work/gj18/e43001/miniconda3/bin/:$PATH"
export LD_LIBRARY_PATH="/work/gj18/e43001/miniconda3/lib/"
export MAIN_ADDR="$(hostname)"
export MAIN_PORT=$((10000 + RANDOM % 20000))
export HYDRA_FULL_ERROR=1
# limit per-process thread pools so dataloader workers don't oversubscribe the node
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-1}
if [ -f .venv/bin/activate ]; then
  # shellcheck disable=SC1091
  . .venv/bin/activate
fi
num_gpus=$(nvidia-smi -L | wc -l)
num_nodes=$(sort -u $PBS_NODEFILE | wc -l)
#multiply num_nodes by num_procs to get total number of processes
num_procs=$((num_nodes*num_gpus))

unset OMPI_MCA_mca_base_env_list
mpirun  \
    -x PATH \
    -x LD_LIBRARY_PATH \
    -x MAIN_ADDR \
    -x MAIN_PORT \
    -x HYDRA_FULL_ERROR \
    -bind-to none \
     -np $num_procs -map-by ppr:$num_gpus:node -hostfile $PBS_NODEFILE \
    .venv/bin/python src/sidon/train.py \
  data=dialogue_preprocessed \
  data.datamodule.batch_size=4 \
  model=dialogue_sidon \
  train=default \
  train.trainer.gradient_clip_val=null \
  train.trainer.precision=bf16-mixed \
  hydra.run.dir=./sidon_runs/${PBS_JOBID} \
  +train.trainer.num_nodes=$num_nodes +train.trainer.devices=$num_gpus \
 'train.ckpt_path="/work/gj18/e43001/github.com/Sidon/sidon/7bsqorip/checkpoints/epoch=0-step=300000.ckpt"'
