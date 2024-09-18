import torch
import time

try:
    import vllm_flash_attn as flash_attn
except ModuleNotFoundError:
    try:
        import flash_attn
    except ModuleNotFoundError:
        print("Flash attention not found!")
        exit(1)

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
    flash_attn.flash_attn_varlen_func(
        q=q,
        k=k,
        v=v,
        cu_seqlens_q=seqlens,
        cu_seqlens_k=seqlens,
        max_seqlen_q=seqlen,
        max_seqlen_k=seqlen,
        softmax_scale=head_size**-0.5,
        causal=True,
        window_size=(-1, -1),
        alibi_slopes=None,
    )
torch.cuda.synchronize()
start_time = time.time()
for _ in range(repeat):
    flash_attn.flash_attn_varlen_func(
        q=q,
        k=k,
        v=v,
        cu_seqlens_q=seqlens,
        cu_seqlens_k=seqlens,
        max_seqlen_q=seqlen,
        max_seqlen_k=seqlen,
        softmax_scale=head_size**-0.5,
        causal=True,
        window_size=(-1, -1),
        alibi_slopes=None,
    )
torch.cuda.synchronize()
end_time = time.time()

duration = (end_time - start_time) * 1e6 / repeat
print(f"Operator duration: {duration:.2f} us")
