import argparse
import os
import pandas as pd
import subprocess
import sys

script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
tmp_dir = os.path.join(script_dir, 'tmp')
script_impl_path = os.path.join(script_dir, 'benchmark_attn_impl.py')
output_path = os.path.join(tmp_dir, 'output.csv')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--q-heads', type=int, default=32)
    parser.add_argument('--kv-heads', type=int, default=32)
    parser.add_argument('--head-size', type=int, default=128)
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--repeat', type=int, default=100)
    parser.add_argument('--seq-lens', type=str, default='128')
    parser.add_argument('--profiler', type=str, default='none')
    args = parser.parse_args()

    seq_lens = [int(s) for s in args.seq_lens.split(',')] if args.seq_lens else []
    assert len(seq_lens) > 0
    flops = 0
    bytes = 0
    for seq_len in seq_lens:
        # q * k (causal masked)
        flops += 2 * (seq_len * (seq_len + 1) / 2) * (args.q_heads * args.head_size)
        bytes += 2 * (seq_len * args.q_heads * args.head_size + args.q_heads * args.head_size * seq_len)
        # qk * v (causal masked)
        flops += 2 * (seq_len * (seq_len + 1) / 2) * (args.q_heads * args.head_size)
        bytes += 2 * (args.q_heads * args.head_size * seq_len + seq_len * args.q_heads * args.head_size)
    
    if not os.path.exists(tmp_dir):
        try:
            subprocess.run(['mkdir', tmp_dir])
        except Exception as e:
            print("Error executing command:", e)
    profiler = args.profiler
    benchmark_impl = f"""python3 {script_impl_path} \
    --q-heads {args.q_heads} \
    --kv-heads {args.kv_heads} \
    --head-size {args.head_size} \
    --warmup {args.warmup} \
    --repeat {args.repeat} \
    --seq-lens {args.seq_lens}"""

    try:
        if profiler == 'none':
            proc = subprocess.Popen(['bash'], cwd=tmp_dir, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            stdout, stderr = proc.communicate(benchmark_impl)
            if proc.returncode != 0:
                raise subprocess.CalledProcessError(proc.returncode, commands, output=stdout, stderr=stderr)
            time_us = float(stdout)

        elif profiler == 'nsys':
            proc = subprocess.Popen(['bash'], cwd=tmp_dir, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            commands = f"""
                nsys start -c nvtx -f true -o output --stats=true
                nsys launch -p benchmark -e NSYS_NVTX_PROFILER_REGISTER_ONLY=0 {benchmark_impl}
                nsys stats --force-export=true -f csv --force-overwrite=true -o output -r cuda_gpu_kern_sum output.nsys-rep
                mv output_cuda_gpu_kern_sum.csv {output_path}
            """
            stdout, stderr = proc.communicate(commands)
            if proc.returncode != 0:
                raise subprocess.CalledProcessError(proc.returncode, commands, output=stdout, stderr=stderr)
            df = pd.read_csv(output_path)
            time_us = df["Avg (ns)"][0] / 1e3

        elif profiler == 'ncu':
            proc = subprocess.Popen(['bash'], cwd=tmp_dir, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            commands = f"""
                ncu -f --nvtx --print-summary per-nvtx --kernel-id ::flash_fwd_kernel: --csv --metrics gpu__time_duration.avg -o output {benchmark_impl}
                ncu -i output.ncu-rep --print-summary per-nvtx --csv > {output_path}
            """
            stdout, stderr = proc.communicate(commands)
            if proc.returncode != 0:
                raise subprocess.CalledProcessError(proc.returncode, commands, output=stdout, stderr=stderr)
            df = pd.read_csv(output_path)
            time_us = df["Average"][0]

        elif profiler == 'rocprof':
            proc = subprocess.Popen(['bash'], cwd=tmp_dir, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            commands = f"""
                echo -e "pmc:\nkernel: mha Attention" > input.txt
                rocprof -i input.txt --hsa-trace -o {output_path} {benchmark_impl}
            """
            stdout, stderr = proc.communicate(commands)
            if proc.returncode != 0:
                raise subprocess.CalledProcessError(proc.returncode, commands, output=stdout, stderr=stderr)
            df = pd.read_csv(output_path)
            time_us = df["DurationNs"][args.warmup:-1].mean() / 1e3

        else:
            print(f"Unrecognized profiler: {profiler}")
            raise

    except subprocess.CalledProcessError as e:
        print(f"Command '{e.cmd}' failed with return code {e.returncode}")
        print(f"Output: {e.output}")
        print(f"Error: {e.stderr}")
        raise
    print(f"arguments: seq_lens={args.seq_lens}, heads={args.q_heads}, kvheads={args.kv_heads}, head_size={args.head_size}")
    print(f"duration: {time_us:.2f} us")
    print(f"flops: {flops/time_us/1e6:.2f} Tflops")
    print(f"bandwidth: {bytes/time_us/1e3:.2f} GB/s")
