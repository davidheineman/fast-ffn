import os
import torch
from torch import Tensor
import extension_cpp
import psutil

# Set OpenMP/MKL threads for max parallel compute
n_threads = 1
os.environ["OMP_NUM_THREADS"] = str(n_threads)
os.environ["MKL_NUM_THREADS"] = str(n_threads)
torch.set_num_threads(n_threads)

# Add diagnostic prints
print(f"OMP_NUM_THREADS: {os.environ.get('OMP_NUM_THREADS')}")
print(f"MKL_NUM_THREADS: {os.environ.get('MKL_NUM_THREADS')}")
print(f"Torch threads: {torch.get_num_threads()}")

torch.manual_seed(42)


def sample_inputs(device, *, requires_grad=False):
    def make_tensor(*size):
        return torch.randn(size, device=device, requires_grad=requires_grad)

    input_width = 768
    output_width = 768
    depth = 12
    n_nodes = (1 << (depth + 1)) - 1 # 2**12 - 1

    # batch_size = 1
    batch_size = 4
    # batch_size = 16

    return [
        [
            # make_tensor(16, n_nodes, input_width),  # x
            make_tensor(batch_size, n_nodes, input_width),  # x (batch_size, mlp_dim, hidden_dim)
            input_width,  # input_width
            output_width,  # output_width
            depth,  # depth
            make_tensor(n_nodes, input_width),  # weights_in
            make_tensor(output_width, n_nodes)  # weights_out
        ]
    ]


def reference_mlp(
    x: Tensor,
    input_width: int,
    output_width: int,
    depth: int,
    weights_in: Tensor,
    weights_out: Tensor
) -> Tensor:
    batch_size, seq_length, _ = x.shape
    x_flat = x.view(-1, input_width)
    hidden = x_flat @ weights_in.T
    hidden = torch.nn.GELU()(hidden)
    output = hidden @ weights_out.T
    output = output.view(batch_size, seq_length, output_width)
    return output


def reference_fff(
    x: Tensor, 
    input_width: int, 
    output_width: int, 
    depth: int, 
    weights_in: Tensor, 
    weights_out: Tensor
) -> Tensor:
    batch_size, seq_length, _ = x.shape

    # Initialize tensors
    current_nodes = torch.zeros((batch_size, seq_length), dtype=torch.long, device=x.device)
    all_nodes = torch.zeros((batch_size, seq_length, depth + 1), dtype=torch.long, device=x.device)
    all_logits = torch.empty((batch_size, seq_length, depth + 1), dtype=torch.float, device=x.device)

    # Iterate through depth
    for i in range(depth + 1):
        all_nodes[:, :, i] = current_nodes
        plane_coeffs = weights_in[current_nodes.flatten()].view(-1, input_width)
        plane_coeff_score = (x.view(-1, 1, input_width) @ plane_coeffs.unsqueeze(-1)).view(batch_size, seq_length)
        current_nodes = current_nodes * 2 + (plane_coeff_score >= 0).long() + 1
        all_logits[:, :, i] = plane_coeff_score

    # Select weights and compute MLP layers
    selected_weights_out = weights_out.T[all_nodes.flatten()].view(batch_size, seq_length, depth + 1, output_width)
    mlp1 = torch.nn.GELU()(all_logits)
    mlp2 = torch.einsum("bsl,bslw->bsw", mlp1, selected_weights_out)

    return mlp2


def test_correctness():
    device = 'cpu'
    samples = sample_inputs(device)

    for args in samples:
        x, input_width, output_width, depth, weights_in, weights_out = args
        print(x.shape)
        result = extension_cpp.ops.fff(x, input_width, output_width, depth, weights_in, weights_out)
        print(result)
        expected = reference_fff(x, input_width, output_width, depth, weights_in, weights_out)
        print(expected)
        torch.testing.assert_close(result, expected, rtol=1e-2, atol=1e-2)


def test_speed(n_trials):
    print("Benchmarking latency...")
    import time

    def monitor_cpu_usage():
        return psutil.cpu_percent(interval=0.1, percpu=True)
    
    # Get device list
    device_list = ['cpu']
    if torch.cuda.is_available():
        device_list.append('cuda')

    # Pre-generate all samples
    cpu_samples = [sample_inputs('cpu')[0] for _ in range(n_trials)]
    cuda_samples = [sample_inputs('cuda')[0] for _ in range(n_trials)] if 'cuda' in device_list else []

    # Get initial sample to print dimensions
    x, input_width, output_width, depth, weights_in, weights_out = cpu_samples[0]
    print(f"\nInput shape: {x.shape}, input_width: {input_width}, output_width: {output_width}, depth: {depth}")
    
    # Warmup C++ implementation (CPU only)
    for args in cpu_samples:
        _ = extension_cpp.ops.fff(*args)
    
    # Time C++ implementation (CPU only)
    start_time = time.perf_counter()
    cpu_usage_samples = []
    for args in cpu_samples:
        cpu_usage_before = monitor_cpu_usage()
        _ = extension_cpp.ops.fff(*args)
        cpu_usage_samples.append(monitor_cpu_usage())
    cpp_time = (time.perf_counter() - start_time) * 1000 / n_trials
    
    # Print CPU usage statistics
    avg_cpu_usage = [sum(core)/len(cpu_usage_samples) for core in zip(*cpu_usage_samples)]
    print(f"\nC++ FFF (CPU): {cpp_time:.3f} ms per iteration")
    print(f"Average CPU core utilization: {sum(avg_cpu_usage)/len(avg_cpu_usage):.1f}%")
    print(f"Number of CPU cores: {psutil.cpu_count()}")
    print(f"Active cores (CPUs with >10% usage): {sum(1 for x in avg_cpu_usage if x > 10)}")
    print()

    # Store CPU reference time for later comparison
    cpu_ref_time = None

    # Benchmark reference implementation on each device
    for device in device_list:
        samples = cuda_samples if device == 'cuda' else cpu_samples
        
        # Warmup reference implementation
        for args in samples:
            _ = reference_fff(*args)
            _ = reference_mlp(*args)
        
        if device == 'cuda':
            # CUDA timing
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            
            start.record()
            for args in samples:
                _ = reference_fff(*args)
            end.record()
            torch.cuda.synchronize()
            ref_time = start.elapsed_time(end) / n_trials

            # Time MLP implementation
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            
            start.record()
            for args in samples:
                _ = reference_mlp(*args)
            end.record()
            torch.cuda.synchronize()
            mlp_time = start.elapsed_time(end) / n_trials
        else:
            # CPU timing
            start_time = time.perf_counter()
            for args in samples:
                _ = reference_fff(*args)
            ref_time = (time.perf_counter() - start_time) * 1000 / n_trials
            cpu_ref_time = ref_time

            # Time MLP implementation
            start_time = time.perf_counter()
            for args in samples:
                _ = reference_mlp(*args)
            mlp_time = (time.perf_counter() - start_time) * 1000 / n_trials

        print(f"PyTorch FFF ({device.upper()}): {ref_time:.3f} ms per iteration")
        print(f"PyTorch MLP ({device.upper()}): {mlp_time:.3f} ms per iteration")
        if device == 'cpu':
            print(f"CPU Speedup PT FFF vs C++ FFF: {ref_time/cpp_time:.2f}x")
            print(f"CPU Speedup PT MLP vs C++ FFF: {mlp_time/cpp_time:.2f}x")
        elif device == 'cuda':
            print(f"CUDA PT FFF vs CPU C++ FFF: {ref_time/cpu_ref_time:.3f}x")
            print(f"CUDA PT FFF vs CUDA PT MLP: {mlp_time/ref_time:.3f}x")


if __name__ == '__main__': 
    # test_correctness()
    test_speed(n_trials=1)
