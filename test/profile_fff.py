import os
import torch
from david_fff import sample_inputs, reference_fff, reference_mlp
import extension_cpp
import subprocess
from pathlib import Path

output_dir = Path("profiling_results")
output_dir.mkdir(exist_ok=True)

def run_profiling(duration, output_path):
    """
    Run profiling for the specified duration and generate a flamegraph
    """
    # Create a separate process that runs the workload
    pid = os.fork()
    
    if pid == 0:  # Child process
        # Run the workload continuously
        while True:
            args = sample_inputs('cpu')[0]
            extension_cpp.ops.fff(*args)
            # reference_fff(*args)
            reference_mlp(*args)
    else:
        try:
            # Run py-spy on the child process
            cmd = [
                "py-spy", "record",
                "--pid", str(pid),
                "--output", output_path,
                "--format", "flamegraph",
                "--duration", str(duration),
                "--native",  # Include native C++ calls
                "--subprocesses",  # Profile subprocesses too
            ]
            
            subprocess.run(cmd, check=True)
        finally:
            # Make sure to cleanup the child process
            os.kill(pid, 9)

def profile_single_run():
    """
    Profile a single run with cProfile for detailed stats
    """
    import cProfile
    import pstats
    from pstats import SortKey
    
    args = sample_inputs('cpu')[0]
    
    # Profile C++ implementation
    profiler = cProfile.Profile()
    profiler.enable()
    extension_cpp.ops.fff(*args)
    profiler.disable()
    
    # Print detailed stats
    stats = pstats.Stats(profiler)
    stats.sort_stats(SortKey.CUMULATIVE)
    stats.print_stats(50)  # Print top 50 entries
    
    # Also save to file
    stats.dump_stats(str(output_dir / "cpp_profile.prof"))
    
    # Profile PyTorch implementation
    profiler = cProfile.Profile()
    profiler.enable()
    reference_fff(*args)
    profiler.disable()
    
    stats = pstats.Stats(profiler)
    stats.sort_stats(SortKey.CUMULATIVE)
    stats.print_stats(50)
    stats.dump_stats(str(output_dir / "pytorch_profile.prof"))

def profile_with_torch():
    """
    Profile using PyTorch's built-in profiler
    """
    from torch.profiler import profile, record_function, ProfilerActivity
    
    args = sample_inputs('cpu')[0]
    
    with profile(
        activities=[
            ProfilerActivity.CPU,
            ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        with record_function("cpp_fff"):
            extension_cpp.ops.fff(*args)
        with record_function("pytorch_fff"):
            reference_fff(*args)
        with record_function("pytorch_mlp"):
            reference_mlp(*args)
    
    # Print table with results
    print(prof.key_averages().table(
        sort_by="cpu_time_total", 
        row_limit=20
    ))
    
    # Export chrome trace
    prof.export_chrome_trace(str(output_dir / "trace.json"))

if __name__ == "__main__":
    # Create output directory
    print("Running flamegraph profiling...")
    run_profiling(duration=100, output_path=str(output_dir / "flamegraph.svg"))
    
    print("\nRunning cProfile analysis...")
    profile_single_run()
    
    print("\nRunning PyTorch profiler...")
    profile_with_torch() 