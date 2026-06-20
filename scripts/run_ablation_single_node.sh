#!/usr/bin/env bash
# Portable launcher for single-node multi-GPU ablation training.
# Optional SLURM usage:
#   sbatch --export=ALL scripts/run_ablation_single_node.sh configs/attention_pool_extra3_full_train2/fold0.yaml

set -euo pipefail

CONFIG_FILE="${1:-}"
if [[ -z "${CONFIG_FILE}" ]]; then
  echo "Usage: scripts/run_ablation_single_node.sh configs/<experiment>/foldX.yaml [-- extra trainer args]"
  exit 1
fi
shift || true

VIDEO_FOLDER="${VIDEO_FOLDER:-./Data/LusBeninVideos}"
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
WORKDIR="${WORKDIR:-$(cd "$(dirname "$0")/.." && pwd)}"

cd "$WORKDIR"
mkdir -p ./outputs/logs ./ablation_results/logs

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:512}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-$((10000 + RANDOM % 50000))}"

CMD=(torchrun --standalone --nproc_per_node "${GPUS_PER_NODE}" --max_restarts 0
  ultr_ai/train/train_ablation_distributed.py
  --config "${CONFIG_FILE}"
  --video_folder "${VIDEO_FOLDER}")

if (($# > 0)); then
  CMD+=("$@")
fi

echo "Running: ${CMD[*]}"
"${CMD[@]}"
