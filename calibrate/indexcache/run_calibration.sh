#!/bin/bash
# ==============================================================
# IndexCache single-GPU offline calibration script
#
# Run from this directory:
#   bash run_calibration.sh
#
# Update the variables below to change the model, dataset, output directory,
# or calibration settings.
# ==============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- GPU configuration ----
export CUDA_VISIBLE_DEVICES="0"
export TOKENIZERS_PARALLELISM=false

# ---- Path configuration ----
model_dir="Qwen/Qwen3-4B"
data_path="${SCRIPT_DIR}/../data/openr1_1000.jsonl"
output_dir="${SCRIPT_DIR}/output-qwen3_4b_openr1"
calibration_script="${SCRIPT_DIR}/offline_calibration.py"

# ---- Calibration hyperparameters ----
num_samples=32
max_seq_len=32768
k=512
retention_ratio=0.1111111111111111  # Keep 4 of Qwen3-4B's 36 layers as F layers
search_mode="loss"

# Search strategy:
#   beam_size=1  -> greedy search (fastest)
#   beam_size=2  -> beam search with 2 candidates
#   beam_size=4  -> beam search with 4 candidates
beam_size=1

# Fraction of the sequence used for loss evaluation:
#   loss_last_frac=1.0 -> use all tokens
#   loss_last_frac=0.1 -> use the last 10% of tokens
loss_last_frac=1.0
seed=42

if [[ ! -f "${calibration_script}" ]]; then
    echo "[Error] Calibration script not found: ${calibration_script}" >&2
    exit 1
fi

if [[ ! -e "${data_path}" ]]; then
    echo "[Error] Calibration data not found: ${data_path}" >&2
    echo "       Please update data_path in this script." >&2
    exit 1
fi

mkdir -p "${output_dir}"
TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"
shell_log="${output_dir}/calibration_${TIMESTAMP}.log"

echo "============================================"
echo "  IndexCache single-GPU offline calibration"
echo "============================================"
echo "GPU:              ${CUDA_VISIBLE_DEVICES}"
echo "Model:            ${model_dir}"
echo "Data:             ${data_path}"
echo "Output:           ${output_dir}"
echo "Samples:          ${num_samples}"
echo "Max sequence len: ${max_seq_len}"
echo "Top-k:            ${k}"
echo "Retention ratio:  ${retention_ratio}"
echo "Search mode:      ${search_mode}"
echo "Beam size:        ${beam_size}"
echo "Loss last frac:   ${loss_last_frac}"
echo "Seed:             ${seed}"
echo "Log:              ${shell_log}"
echo "============================================"

# offline_calibration.py falls back to single-GPU mode when RANK is unset.
python "${calibration_script}" \
    --model_name_or_path "${model_dir}" \
    --data_path "${data_path}" \
    --output_dir "${output_dir}" \
    --num_samples "${num_samples}" \
    --max_seq_len "${max_seq_len}" \
    --k "${k}" \
    --retention_ratio "${retention_ratio}" \
    --search_mode "${search_mode}" \
    --beam_size "${beam_size}" \
    --loss_last_frac "${loss_last_frac}" \
    --seed "${seed}" \
    --run_timestamp "${TIMESTAMP}" \
    2>&1 | tee "${shell_log}"

echo ""
echo "Calibration completed!"
echo "  Config: ${output_dir}/indexcache_config.json"
echo "  Log:    ${shell_log}"
