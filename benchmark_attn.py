import argparse
import subprocess
import sys
import os

script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
tmp_dir = os.path.join(script_dir, 'tmp')
script_impl_path = os.path.join(script_dir, 'benchmark_attn_impl.py')

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
            with subprocess.Popen(['bash'], cwd=tmp_dir, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) as proc:
                stdout, stderr = proc.communicate(benchmark_impl)
            print(stdout)
            print(stderr)

        elif profiler == 'nsys':
            with subprocess.Popen(['bash'], cwd=tmp_dir, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) as proc:
                commands = f"""
                    nsys start -c nvtx -f true -o rep --stats=true
                    nsys launch -p benchmark -e NSYS_NVTX_PROFILER_REGISTER_ONLY=0 {benchmark_impl}
                    nsys stats --force-export=true -f csv --force-overwrite=true -o rep -r cuda_gpu_kern_sum rep.nsys-rep
                    mv rep_cuda_gpu_kern_sum.csv rep.csv
                """
                stdout, stderr = proc.communicate(commands)
            print(stdout)
            print(stderr)

        elif profiler == 'ncu':
            with subprocess.Popen(['bash'], cwd=tmp_dir, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) as proc:
                commands = f"""
                    ncu -f --nvtx --print-summary per-nvtx --kernel-id ::flash_fwd_kernel: --csv --metrics gpu__time_duration.avg -o rep {benchmark_impl}
                    ncu -i rep.ncu-rep --print-summary per-nvtx --csv > rep.csv
                """
                stdout, stderr = proc.communicate(commands)
            print(stdout)
            print(stderr)

        elif profiler == 'rocprof':
            with subprocess.Popen(['bash'], cwd=tmp_dir, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) as proc:
                commands = f"""
                    echo -e "pmc:\nkernel: mha Attention" > input.txt
                    rocprof -i input.txt --hsa-trace -o rep.csv {benchmark_impl}
                """
                stdout, stderr = proc.communicate(commands)
            print(stdout)
            print(stderr)
    except Exception as e:
        print("Error:", e)
    # print(f"duration: {duration:.2f} us")
    # print(f"flops: {flops:.2f} Tflops")
    # print(f"bytes: {bytes:.2f} GB")
