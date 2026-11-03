export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

model_path="Qwen/Qwen3-4B"

# ==========================================
# 1. Prism method (proposal + refined cross-layer index sharing)
# ==========================================
token_budget=2048
attention_method="prism" # indexcache or dense 

# Prism knobs (map to candidate_factor / proposal_layers in PrismQwen3Config)
export PRISM_CANDIDATE_FACTOR=2.0
export PRISM_PROPOSAL_LAYERS=0,7,8,13,15,17,18,26,35

output_path=./results/${attention_method}
mkdir -p $output_path

export PROFILE_FILE="${output_path}/profile.txt"

python pred_threshold_decode.py \
    --model $model_path \
    --e \
    --token_budget $token_budget \
    --attention_method $attention_method \
    --use_batch_exist \
    --use_fused_kernel \
    --output_path $output_path > $output_path/out.log 2>&1


