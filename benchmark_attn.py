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
    if !os.path.exists(tmp_dir):
        try:
            subprocess.run(['mkdir', tmp_dir])
        except Exception as e:
            print("Error executing command:", e)

    if args.profiler == 'none':
        args = [arg for arg in sys.argv[1:] if arg != '--profiler' and arg != 'none']
        command = ["cd", tmp_dir, '&&', 'python3', script_impl_path] + args
        try:
            print(subprocess.run(command))
        except Exception as e:
            print("Error executing command:", e)
    # print(f"duration: {duration:.2f} us")
    # print(f"flops: {flops:.2f} Tflops")
    # print(f"bytes: {bytes:.2f} GB")
