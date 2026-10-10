#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source .venv-neuron/bin/activate
export LD_LIBRARY_PATH="$PWD/.python/cpython-3.12.14-linux-x86_64-gnu/lib:${LD_LIBRARY_PATH:-}"
export NEURON_RT_VISIBLE_CORES=0,1,2,3
export NEURON_LOGICAL_NC_CONFIG=2 NEURON_SKIP_EFA_AFFINITY=1 PJRT_DEVICE=NEURON
export NEURON_CC_FLAGS='--model-type=transformer --retry_failed_compilation'
export NEURON_PROCESS_TAG=gptoss-training
exec torchrun --standalone --nproc_per_node=4 "$@"
