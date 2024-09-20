import argparse
import torch
import flash_attn_2_cuda as flash_attn
import torch.cuda.nvtx as nvtx

def run_flash_attn(q, k, v, acc_seq_lens, max_seq_len, scale):
    flash_attn.varlen_fwd(
        q,
        k,
        v,
        None,
        acc_seq_lens,
        acc_seq_lens,
        None,
        None,
        None,
        None,
        max_seq_len,
        max_seq_len,
        0.0,
        scale,
        False,
        True,
        -1,
        -1,
        0.0,
        False,
        None,
    )
    
def run(acc_seq_lens, max_seq_len, total_seq_len, q_heads, kv_heads, head_size, warmup, repeat):
    qkv = torch.empty((total_seq_len, q_heads + kv_heads + kv_heads, head_size), device='cuda', dtype=torch.float16)
    q, k, v = qkv.split([q_heads, kv_heads, kv_heads], dim=-2)
    seqlens = torch.tensor(acc_seq_lens, device='cuda', dtype=torch.int32)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    for _ in range(warmup):
        run_flash_attn(q, k, v, acc_seq_lens, max_seq_len, head_size**-0.5)

    torch.cuda.synchronize()
    nvtx.range_push("benchmark")
    start.record()
    for _ in range(repeat):
        run_flash_attn(q, k, v, acc_seq_lens, max_seq_len, head_size**-0.5)
    end.record()
    torch.cuda.synchronize()
    nvtx.range_pop()
    return start.elapsed_time(end) * 1e3 / repeat

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--q-heads', type=int, default=32)
    parser.add_argument('--kv-heads', type=int, default=32)
    parser.add_argument('--head-size', type=int, default=128)
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--repeat', type=int, default=100)
    parser.add_argument('--seq-lens', type=str, default='128')
    args = parser.parse_args()

    seq_lens = [int(s) for s in args.seq_lens.split(',')] if args.seq_lens else []
    assert len(seq_lens) > 0
    max_seq_len = 0
    total_seq_len = 0
    acc_seq_lens = []
    for seq_len in seq_lens:
        if seq_len > max_seq_len:
            max_seq_len = seq_len
        total_seq_len += seq_len
        acc_seq_lens.append(total_seq_len)
    duration = run(acc_seq_lens, max_seq_len, total_seq_len, args.q_heads, args.kv_heads, args.head_size, args.warmup, args.repeat)
    print(f"Operator duration: {duration:.2f} us")
