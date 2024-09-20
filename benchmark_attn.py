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
    impl = f"python3 {script_impl_path} --q-heads {args.q_heads} --kv-heads {args.kv_heads} --head-size {args.head_size} --warmup {args.warmup} --repeat {args.repeat} --seq-lens {args.seq_lens}"

    try:
        if profiler == 'none':
            with subprocess.Popen(['bash'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) as proc:
            stdout, stderr = proc.communicate(impl)
            print(stdout)
            print(stderr)

        elif profiler == 'nsys':
            with subprocess.Popen(['bash'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) as proc:
                commands = f"""
                    nsys start --capture-range=nvtx --force-overwrite=true --output=rep --stats=true
                    nsys launch --nvtx-capture=benchmark --env-var=NSYS_NVYX_PROFILER_REGISTER_ONLY=0 {impl}
                    nsys stats --force-export=true --format=csv --force-overwrite=true --output=rep --report=cuda_gpu_kernel_sum rep.nsys-rep
                """
            stdout, stderr = proc.communicate(commands)
            print(stdout)
            print(stderr)
    except Exception as e:
        print("Error:", e)
    # print(f"duration: {duration:.2f} us")
    # print(f"flops: {flops:.2f} Tflops")
    # print(f"bytes: {bytes:.2f} GB")
