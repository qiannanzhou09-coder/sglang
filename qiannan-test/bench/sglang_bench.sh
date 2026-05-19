export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
lm_eval --model local-completions \
  --model_args model=default,base_url=http://localhost:8000/v1/completions,tokenizer=/inspire/hdd/global_public/public_models/Qwen/Qwen3.5-122B-A10B-FP8 \
  --tasks gsm8k \
  --batch_size auto