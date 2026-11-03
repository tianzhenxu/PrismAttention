model_dir="Qwen/Qwen3-4B"
attention_method="prism"      # prism / indexcache / dense / fa2 / sdpa
max_tokens=32768
num_gpus=8
limit=-1
output_dir="./result_${attention_method}"
# tasks="aime24,aime25,math,gpqa"
tasks="math"

token_budget="2048"
start_layer=0

# Prism-only knobs (map to candidate_factor / proposal_layers in PrismQwen3Config).
# Only used when attention_method is "prism" or "indexcache"; ignored otherwise.
export PRISM_CANDIDATE_FACTOR=2.0
export PRISM_PROPOSAL_LAYERS=0,7,8,13,14,18,20,21,28

python parallel_run_hf.py \
      --model_dir "$model_dir" \
      --tasks "$tasks" \
      --output_dir "$output_dir" \
      --attention_method "$attention_method" \
      --token_budget "$token_budget" \
      --num_gpus "$num_gpus" \
      --limit "$limit" \
      --max_tokens "$max_tokens" \
      --start_layer "$start_layer"
