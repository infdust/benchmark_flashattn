# prepare cuda

Reference docker image: nvidia/cuda:12.4.1-devel-ubuntu22.04  
You may need to install Python3.9 and other tools by yourself  
```bash
pip install flash_attn==2.6.3
```

# prepare rocm

Reference docker image: rocm/pytorch:rocm6.1.2_ubuntu20.04_py3.9_pytorch_staging  
```bash
pip install torch==2.4.0 flash_attn==2.6.3
```

You might fail to install flash-attention by pip. Try to install from source instead:
```bash
git clone https://github.com/Dao-AILab/flash-attention
cd flash-attention
git checkout v2.6.3
GPU_ARGS="gfx90a" pip install -v .
```

# run benchmark

```bash
python3 benchmark_attn.py
```
