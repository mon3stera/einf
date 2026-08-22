sudo ncu \
  --set basic \
  --kernel-name-base function \
  --kernel-name 'regex:.*flash_attention_forward_kernel.*' \
  --launch-skip 10 \
  --launch-count 1 \
  --export /tmp/einf-flash-chunk-basic \
  --force-overwrite \
  python benchmarks/profile_flash_attention.py \
    --q-len 128 \
    --kv-len 32768 \
    --warmup 10
