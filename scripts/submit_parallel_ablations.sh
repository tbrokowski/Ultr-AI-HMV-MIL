#!/usr/bin/env bash
# Submit all paper release ablation families (5 folds each).

set -euo pipefail

CONFIG_BASE_DIR="configs"
SLURM_SCRIPT="scripts/run_ablation_single_node.sh"
SUBMISSION_LOG_DIR="./ablation_results"
mkdir -p "$SUBMISSION_LOG_DIR"

declare -A ABLATIONS=(
  ["hmv_mil"]="attention_pool_extra3_full_train2"
  ["uniform"]="uniform_extra3"
  ["mean_pool"]="mean_pool_extra3"
  ["no_pathology"]="singletask_extra3"
  ["no_pretraining"]="attention_pool_noInitWeights"
  ["k1"]="attention_pool_extra4_k1"
  ["k8"]="attention_pool_extra4_k8"
  ["no_mil"]="attention_pool_cxr"
  ["levit"]="LeViT-Attention"
  ["3dcnn"]="3dcnn"
  ["cnnlstm"]="cnnlstm"
  ["inception3d"]="inception3d"
  ["vivit"]="vivit"
  ["r2plus1d"]="r2plus1d"
)

FOLDS=(0 1 2 3 4)
SUBMIT_MODE="${1:-local}"
MAX_PARALLEL_JOBS="${MAX_PARALLEL_JOBS:-4}"

submit_one() {
  local config_path="$1"
  if [[ "$SUBMIT_MODE" == "slurm" ]] && command -v sbatch >/dev/null 2>&1; then
    sbatch "$SLURM_SCRIPT" "$config_path"
  else
    bash "$SLURM_SCRIPT" "$config_path"
  fi
}

active=0
for _key in "${!ABLATIONS[@]}"; do
  family="${ABLATIONS[$_key]}"
  for fold in "${FOLDS[@]}"; do
    config_path="${CONFIG_BASE_DIR}/${family}/fold${fold}.yaml"
    if [[ ! -f "$config_path" ]]; then
      echo "Missing config: $config_path" >&2
      continue
    fi
    submit_one "$config_path"
    active=$((active + 1))
    if [[ "$SUBMIT_MODE" != "slurm" && "$active" -ge "$MAX_PARALLEL_JOBS" ]]; then
      wait
      active=0
    fi
  done
done
wait

echo "Submitted ${#ABLATIONS[@]} families x ${#FOLDS[@]} folds (mode=${SUBMIT_MODE})"
