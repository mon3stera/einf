mkdir -p benchmark-results

sudo -E env \
  PYTHONPATH="$PWD/src" \
  PATH="$HOME/work/einf-venv/bin:$PATH" \
  "$(command -v ncu)" \
  --target-processes all \
  --kernel-name-base demangled \
  --kernel-name 'regex:.*flash_attention_kernel.*' \
  --launch-skip 10 \
  --launch-count 1 \
  --replay-mode kernel \
  --set full \
  --force-overwrite \
  -o "$PWD/benchmark-results/ncu-cute-fa-basic" \
  "$HOME/work/einf-venv/bin/python" -c '
import math
import torch

from einf.executors.torch.dsl import cute_flash_attention

torch.manual_seed(0)

q_len = 4096
kv_len = 4096
num_attention_heads = 14
num_kv_heads = 2
head_dim = 64

Q = torch.randn(
    q_len, num_attention_heads, head_dim,
    device="cuda", dtype=torch.bfloat16,
)
K = torch.randn(
    kv_len, num_kv_heads, head_dim,
    device="cuda", dtype=torch.bfloat16,
)
V = torch.randn_like(K)

start_pos = kv_len - q_len
scale = 1.0 / math.sqrt(head_dim)

for _ in range(10):
    cute_flash_attention(Q, K, V, start_pos, scale)

torch.cuda.synchronize()

cute_flash_attention(Q, K, V, start_pos, scale)
torch.cuda.synchronize()
'