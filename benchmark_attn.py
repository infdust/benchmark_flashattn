import torch
import time
import flash_attn_2_cuda as flash_attn

seqlen = 128 * 4
warmup = 100
repeat = 1000
heads = 32
kv_heads = 32
head_size = 128

qkv = torch.empty((seqlen, heads + kv_heads + kv_heads, head_size), device='cuda', dtype=torch.float16)
q, k, v = qkv.split([heads, kv_heads, kv_heads], dim=-2)
seqlens = torch.tensor([0, seqlen], device='cuda', dtype=torch.int32)

for _ in range(warmup):
    flash_attn.varlen_fwd(
        q,
        k,
        v,
        None,
        seqlens,
        seqlens,
        None,
        None,
        None,
        None,
        seqlen,
        seqlen,
        0.0,
        head_size**-0.5,
        False,
        True,
        -1,
        -1,
        0.0,
        False,
        None,
    )
torch.cuda.synchronize()
start_time = time.time()
for _ in range(repeat):
    flash_attn.varlen_fwd(
        q,
        k,
        v,
        None,
        seqlens,
        seqlens,
        None,
        None,
        None,
        None,
        seqlen,
        seqlen,
        0.0,
        head_size**-0.5,
        False,
        True,
        -1,
        -1,
        0.0,
        False,
        None,
    )
torch.cuda.synchronize()
end_time = time.time()

duration = (end_time - start_time) * 1e6 / repeat
print(f"Operator duration: {duration:.2f} us")
