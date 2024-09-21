# Prepare cuda

Reference docker image: nvidia/cuda:12.4.1-devel-ubuntu22.04  
You may need to install Python3.9 and other tools by yourself  
```bash
pip install flash_attn==2.6.3 pandas
```
You don't have to use the provided docker image, but the torch version must be 2.4.0+.  

# Prepare rocm

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

# Run benchmark

```bash
python3 benchmark_attn.py [options]
```

Here're the avalible options:  
| Argument          | Description |                          
|-------------------|-------------|
| --q-heads(=32)    | Num of heads for Q tensors
| --kv-heads(=32)   | Num of heads for KV tensors
| --head-size(=128) | Head size for QKV tensors
| --warmup(=10)     | Num of iterations in warmup
| --repeat(=100)    | Num of iterations in benchmarking
| --seq-lens(=128)  | Length of each sequences. Can be one or more integers seprated by ','
| --profiler(=none) | Method for recording kernel execution time.<br>Available options: none(use cuda event), ncu, nsys, rocprof.<br>Requires corresponding tools installed except for 'none'.  

# Result

The following results are based on the Referenced environment(docker image), running on MI210(rocm, 181Tflops, 1.64TB/s) and RTX4090(cuda, 165Tflops, 1.01TB/s).  
Five series are tested using four performance testing methods on two platforms:  
| Series  | Platform | profiler |
|---------|----------|----------|
| cuda    | cuda     | none     |
| ncu     | cuda     | ncu      |
| nsys    | cuda     | nsys     |
| rocm    | rocm     | none     |
| rocprof | rocm     | rocprof  |

The following charts show how performance varies with seq_len. Overall, rocm-version of flash-attention has a significant gap compared to cuda.

### Single sequence

The parameter '--seq-lens' is set to: '128', '256', '512', ... . This means a single sequence of variable length is passed in at a time.   

<img src="https://github.com/infdust/benchmark_flashattn/blob/main/assets/duration.png" alt="duration" width="600"/>
<img src="https://github.com/infdust/benchmark_flashattn/blob/main/assets/performance.png" alt="performance" width="600"/>
<img src="https://github.com/infdust/benchmark_flashattn/blob/main/assets/bandwidth.png" alt="bandwidth" width="600"/>

### Grouped sequences

The parameter '--seq-lens' is set to: '128', '128,128', '128,128,128,128', ... . This means a different number of sequences, each of length 128, are passed in at a time.  

<img src="https://github.com/infdust/benchmark_flashattn/blob/main/assets/duration_grouped.png" alt="duration_grouped" width="600"/>
<img src="https://github.com/infdust/benchmark_flashattn/blob/main/assets/performance_grouped.png" alt="performance_grouped" width="600"/>
<img src="https://github.com/infdust/benchmark_flashattn/blob/main/assets/bandwidth_grouped.png" alt="bandwidth_grouped" width="600"/>


