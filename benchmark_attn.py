import torch
import flash_attn_2_cuda as flash_attn
use_nvtx = False
try:
    import torch.cuda.nvtx as nvtx
    use_nvtx = True
except ModuleNotFoundError:
    pass
use_roctx = False
try:
    roctx_lib = ctypes.CDLL("libroctx64.so")
    use_roctx = True
except OSError:
    pass
if use_roctx:
    # roctxRangePush(const char*)
    roctx_lib.roctxRangePushA.argtypes = [ctypes.c_char_p]
    roctx_lib.roctxRangePushA.restype = ctypes.c_int
    def roctx_range_push(name):
        roctx_lib.roctxRangePushA(name.encode('utf-8'))
    # roctxRangePop()
    roctx_lib.roctxRangePop.restype = ctypes.c_int
    def roctx_range_pop():
        roctx_lib.roctxRangePop()

seqlen = 128 * 4
warmup = 100
repeat = 1000
heads = 32
kv_heads = 32
head_size = 128

qkv = torch.empty((seqlen, heads + kv_heads + kv_heads, head_size), device='cuda', dtype=torch.float16)
q, k, v = qkv.split([heads, kv_heads, kv_heads], dim=-2)
seqlens = torch.tensor([0, seqlen], device='cuda', dtype=torch.int32)

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

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
if use_nvtx:
    nvtx.range_push("benchmark")
if use_roctx:
    roctx_range_push("benchmark")
start.record()
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
end.record()
torch.cuda.synchronize()
if use_nvtx:
    nvtx.range_pop()
if use_roctx:
    roctx_range_pop()

duration = start.elapsed_time(end) * 1e3 / repeat
print(f"Operator duration: {duration:.2f} us")
