import torch
from torch import Tensor
import numpy as np

torch.manual_seed(42)


def sample_inputs(device, *, requires_grad=False):
    def make_tensor(*size):
        return torch.randn(size, device=device, requires_grad=requires_grad)

    input_width = 768
    output_width = 768
    depth = 12 # 2**12 - 1
    n_nodes = (1 << (depth + 1)) - 1 

    return [
        [
            make_tensor(16, n_nodes, input_width),  # x
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


def next_power_of_two(n: int) -> int:
    p = 1
    while p < n:
        p *= 2
    return p


def fft_dot(A: Tensor, B: Tensor, L: int) -> Tensor:
    N_fft = next_power_of_two(L)
    A_fft = torch.fft.rfft(A, n=N_fft, dim=-1)
    B_fft = torch.fft.rfft(B, n=N_fft, dim=-1)
    return (torch.matmul(A_fft, torch.conj(B_fft).T) / N_fft).real


def fourier_mlp(
    x: Tensor,
    input_width: int,
    output_width: int,
    depth: int,
    weights_in: Tensor,
    weights_out: Tensor
) -> Tensor:
    batch, n_nodes, _ = x.shape
    x_flat = x.view(-1, input_width)
    hidden = fft_dot(x_flat, weights_in, input_width)
    hidden = torch.nn.GELU()(hidden)
    output = fft_dot(hidden, weights_out, n_nodes)
    return output.view(batch, n_nodes, output_width)


def count_ops(x: Tensor, weights_in: Tensor, weights_out: Tensor, is_fourier: bool) -> dict:
    batch, n_nodes, input_width = x.shape
    output_width = weights_out.shape[0]
    
    if is_fourier:
        # FFT ops (approximate)
        n_fft_in = next_power_of_two(input_width)
        n_fft_nodes = next_power_of_two(n_nodes)
        fft_ops = 2 * (
            batch * n_nodes * n_fft_in * np.log2(n_fft_in) +  # First FFT
            batch * n_nodes * n_fft_nodes * np.log2(n_fft_nodes)   # Second FFT
        )
        return {
            'multiply_adds': fft_ops,
            'activations': batch * n_nodes,  # GELU activations
        }
    else:
        return {
            'multiply_adds': (
                batch * n_nodes * input_width * 1 +  # First matrix multiply
                batch * n_nodes * n_nodes * output_width  # Second matrix multiply
            ),
            'activations': batch * n_nodes,  # GELU activations
        }


def count_memory(x: Tensor, weights_in: Tensor, weights_out: Tensor, is_fourier: bool) -> dict:
    batch, n_nodes, input_width = x.shape
    output_width = weights_out.shape[0]
    bytes_per_float = 4  # assuming float32
    
    if is_fourier:
        n_fft_in = next_power_of_two(input_width)
        n_fft_nodes = next_power_of_two(n_nodes)
        return {
            'input': batch * n_nodes * input_width * bytes_per_float,
            'weights': (n_nodes * input_width + output_width * n_nodes) * bytes_per_float,
            'fft_space': batch * n_nodes * (n_fft_in + n_fft_nodes) * bytes_per_float * 2,  # *2 for complex numbers
            'output': batch * n_nodes * output_width * bytes_per_float
        }
    else:
        return {
            'input': batch * n_nodes * input_width * bytes_per_float,
            'weights': (n_nodes * input_width + output_width * n_nodes) * bytes_per_float,
            'hidden': batch * n_nodes * n_nodes * bytes_per_float,
            'output': batch * n_nodes * output_width * bytes_per_float
        }


def test_speed(n_trials):
    print("Benchmarking latency and memory...")
    import time
    
    def get_fresh_samples(device):
        return sample_inputs(device)[0]

    device_list = []
    if torch.cuda.is_available():
        device_list.append('cuda')
    
    x, input_width, output_width, depth, weights_in, weights_out = get_fresh_samples('cpu')
    print(f"\nInput shape: {x.shape}, input_width: {input_width}, output_width: {output_width}, depth: {depth}")
    
    # Calculate theoretical ops and memory
    ops_fourier = count_ops(x, weights_in, weights_out, is_fourier=True)
    ops_reference = count_ops(x, weights_in, weights_out, is_fourier=False)
    mem_fourier = count_memory(x, weights_in, weights_out, is_fourier=True)
    mem_reference = count_memory(x, weights_in, weights_out, is_fourier=False)
    
    print("\nTheoretical Operation Counts:")
    print(f"Reference Implementation: {int(ops_reference['multiply_adds']):,} multiply-adds, {ops_reference['activations']:,} activations")
    print(f"Fourier Implementation:   {int(ops_fourier['multiply_adds']):,} multiply-adds, {ops_fourier['activations']:,} activations")
    
    print("\nTheoretical Memory Usage:")
    total_fourier = sum(mem_fourier.values()) / (1024**3)  # Convert to GB
    total_reference = sum(mem_reference.values()) / (1024**3)  # Convert to GB
    print(f"Reference Implementation: {total_reference:.2f} GB")
    print(f"Fourier Implementation:   {total_fourier:.2f} GB")
    print(f"Theoretical Memory Ratio: {total_reference/total_fourier:.2f}x")
    
    for device in device_list:
        print(f"\nDevice: {device.upper()}")
        
        # Memory tracking setup
        if device == 'cuda':
            torch.cuda.reset_peak_memory_stats()
        
        # Warmup both implementations
        for _ in range(10):
            args = get_fresh_samples(device)
            _ = fourier_mlp(*args)
            _ = reference_mlp(*args)
        
        if device == 'cuda':
            # CUDA timing for Fourier implementation
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            
            start.record()
            for _ in range(n_trials):
                args = get_fresh_samples(device)
                _ = fourier_mlp(*args)
            end.record()
            torch.cuda.synchronize()
            fourier_time = start.elapsed_time(end) / n_trials
            
            # CUDA timing for reference implementation
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            
            start.record()
            for _ in range(n_trials):
                args = get_fresh_samples(device)
                _ = reference_mlp(*args)
            end.record()
            torch.cuda.synchronize()
            ref_time = start.elapsed_time(end) / n_trials
            
            # Add memory statistics for CUDA
            fourier_memory = torch.cuda.max_memory_allocated() / 1024**2  # Convert to MB
            torch.cuda.reset_peak_memory_stats()
            
            # Run reference implementation to measure its memory
            args = get_fresh_samples(device)
            _ = reference_mlp(*args)
            ref_memory = torch.cuda.max_memory_allocated() / 1024**3  # Convert to GB
            
            print(f"Reference Implementation peak memory: {ref_memory:.1f} GB")
            print(f"Fourier Implementation peak memory:   {fourier_memory/1024:.1f} GB")
            print(f"Memory ratio: {ref_memory/fourier_memory:.2f}x")

        else:
            # CPU timing for Fourier implementation
            start_time = time.perf_counter()
            for _ in range(n_trials):
                args = get_fresh_samples(device)
                _ = fourier_mlp(*args)
            fourier_time = (time.perf_counter() - start_time) * 1000 / n_trials
            
            # CPU timing for reference implementation
            start_time = time.perf_counter()
            for _ in range(n_trials):
                args = get_fresh_samples(device)
                _ = reference_mlp(*args)
            ref_time = (time.perf_counter() - start_time) * 1000 / n_trials

        print(f"Reference Implementation: {ref_time:.3f} ms per iteration") 
        print(f"Fourier Implementation: {fourier_time:.3f} ms per iteration")
        print(f"Speedup: {ref_time/fourier_time:.2f}x")


if __name__ == '__main__': 
    test_speed(n_trials=5)
