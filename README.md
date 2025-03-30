PyTorch extension for Fast FFNs, à la [arxiv.org/abs/2308.14711](https://arxiv.org/abs/2308.14711). Includes a profiling script.

```python
import torch
import torch.nn as nn

input_width = 8
output_width = 8
depth = 2
batch_size = 1
seq_length = 5

fff = FFF(
    input_width=input_width,
    output_width=output_width, 
    depth=depth,
    activation=nn.GELU()
)

x = torch.randn(batch_size, seq_length, input_width)

fff.forward(x)
```

### Setup

To build:
```sh
pip install .
```

To test:
```sh
python test/david_fff.py

# To build and test
pip install . && python test/david_fff.py

# To run profiler (for flamegraph)
python profile_fff.py
```

You should see this output from the speedtest (we can see for `depth=12`, our implementation is 20x faster on the CPU):
```
OMP_NUM_THREADS: 1
MKL_NUM_THREADS: 1
Torch threads: 1
Benchmarking latency...

Input shape: torch.Size([4, 8191, 768]), input_width: 768, output_width: 768, depth: 12

C++ FFF (CPU): 633.927 ms per iteration
Average CPU core utilization: 97.1%
Number of CPU cores: 255
Active cores (CPUs with >10% usage): 253

PyTorch FFF (CPU): 3123.723 ms per iteration
PyTorch MLP (CPU): 12224.502 ms per iteration
CPU Speedup PT FFF vs C++ FFF: 4.93x
CPU Speedup PT MLP vs C++ FFF: 19.28x
PyTorch FFF (CUDA): 11.041 ms per iteration
PyTorch MLP (CUDA): 55.062 ms per iteration
CUDA PT FFF vs CPU C++ FFF: 0.004x
CUDA PT FFF vs CUDA PT MLP: 4.987x
```