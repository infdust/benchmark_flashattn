# prepare cuda

Reference docker image: nvidia/cuda:12.4.1-devel-ubuntu22.04  
You may need to install Python3.9 and other tools by yourself  
```bash
pip install flash_attn==2.6.3 pandas
```
You don't have to use the provided docker image, but the torch version must be 2.4.0+.  

# prepare rocm

Reference docker image: rocm/pytorch:rocm6.1.2_ubuntu20.04_py3.9_pytorch_staging  
```bash
pip install torch==2.4.0 flash_attn==2.6.3 pandas
```

You might fail to install flash-attention by pip. Try to install from source instead:  
```bash
git clone https://github.com/Dao-AILab/flash-attention
cd flash-attention
git checkout v2.6.3
GPU_ARGS="gfx90a" pip install -v .
```
**Note: it might take hours to build flash-attention form source**

# run benchmark

```bash
python3 benchmark_attn.py [options]
```

Here's the avalible options:
| Argument          | Description                           |
|-------------------|---------------------------------------|
| --q-heads(=32)    | Num of heads for Q tensors
| --kv-heads(=32)   | Num of heads for KV tensors
| --head-size(=128) | Head size for QKV tensors
| --warmup(=10)     | Num of iterations in warmup
| --repeat(=100)    | Num of iterations in benchmarking
| --seq-lens(=128)  | Length of each sequences. Can be one or more integers seprated by ','
| --profiler(=none) | Method for recording kernel execution time.<br>Available options: none(use cuda event), ncu, nsys, rocprof.<br>Requires corresponding tools installed except for 'none'.
