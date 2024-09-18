# run on cuda  

Reference docker image: nvidia/cuda:12.4.1-devel-ubuntu22.04  
You may need to install Python3.9 and other tools by yourself  

```bash
pip install -r requirements.txt
python3 benchmark_attn.py
```


# run on rocm  

Reference docker image: rocm/pytorch:rocm6.1.2_ubuntu20.04_py3.9_pytorch_staging  

```bash
pip install -r requirements-rocm.txt
python3 benchmark_attn.py
```
