#!/bin/bash
#PBS -q debug-g
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
num_nodes=$(sort -u "$PBS_NODEFILE" | wc -l)
# multiply num_nodes by num_procs to get total number of processes
num_procs=$((num_nodes*num_gpus))

profile_output_base=${NSYS_PROFILE_OUTPUT:-"./sidon_runs/${PBS_JOBID}/nsys/train_e2e"}
mkdir -p "$(dirname "$profile_output_base")"
# unique output per rank to cover all nodes (defaults to host + mpi rank)
profile_output_pattern=${NSYS_PROFILE_PATTERN:-"${profile_output_base}_%h_rank%q{OMPI_COMM_WORLD_RANK}"}
profile_trace=${NSYS_TRACE:-cuda,nvtx,osrt}
profile_sample=${NSYS_SAMPLE:-none}
profile_cpuctxsw=${NSYS_CPUCTXSW:-none}

nsys_cmd=(/work/opt/local/aarch64/cores/nvidia/24.9/Linux_aarch64/24.9/compilers/bin/nsys profile --force-overwrite=true --trace="${profile_trace}" --sample="${profile_sample}" --cpuctxsw="${profile_cpuctxsw}" --output "${profile_output_pattern}")
if [ -n "${NSYS_PROFILE_ARGS:-}" ]; then
  # shellcheck disable=SC2206
  extra_args=(${NSYS_PROFILE_ARGS})
  nsys_cmd+=("${extra_args[@]}")
fi

unset OMPI_MCA_mca_base_env_list
mpirun \
    -x PATH \
    -x LD_LIBRARY_PATH \
    -x MAIN_ADDR \
    -x MAIN_PORT \
    -x HYDRA_FULL_ERROR \
    -bind-to none \
    -np "$num_procs" -map-by ppr:"$num_gpus":node -hostfile "$PBS_NODEFILE" \
    "${nsys_cmd[@]}" \
    .venv/bin/python src/sidon/train.py \
  data=dialogue_preprocessed \
  data.datamodule.batch_size=16 \
  model=dialogue_sidon_feature_predictor \
  train=default \
  train.trainer.max_steps=50 \
  train.trainer.gradient_clip_val=null \
  train.trainer.precision=bf16-mixed \
  hydra.run.dir=./sidon_runs/${PBS_JOBID} \
  +train.trainer.num_nodes="$num_nodes" +train.trainer.devices="$num_gpus"
