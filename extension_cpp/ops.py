import torch
from torch import Tensor

__all__ = ["fff"]


def fff(x: Tensor, input_width: int, output_width: int, depth: int, weights_in: Tensor, weights_out: Tensor) -> Tensor:
    return torch.ops.extension_cpp.fff(x, input_width, output_width, depth, weights_in, weights_out)


# Registers a FakeTensor kernel (aka "meta kernel", "abstract impl")
# that describes what the properties of the output Tensor are given
# the properties of the input Tensor. The FakeTensor kernel is necessary
# for the op to work performantly with torch.compile.
@torch.library.register_fake("extension_cpp::fff")
def _(x: Tensor, input_width: int, output_width: int, depth: int, weights_in: Tensor, weights_out: Tensor):
    torch._check(x.dtype == torch.float, f"Expected x.dtype == torch.float, got {x.dtype}")
    torch._check(weights_in.dtype == torch.float, f"Expected weights_in.dtype == torch.float, got {weights_in.dtype}")
    torch._check(weights_out.dtype == torch.float, f"Expected weights_out.dtype == torch.float, got {weights_out.dtype}")
    torch._check(x.device == weights_in.device == weights_out.device, 
                 f"Expected tensors to be on the same device: {x.device}, {weights_in.device}, {weights_out.device}")
    torch._check(weights_in.shape[1] == input_width, 
                 f"Expected weights_in.shape[1] == input_width, got {weights_in.shape[1]} vs {input_width}")
    torch._check(weights_out.shape[1] == output_width, 
                 f"Expected weights_out.shape[1] == output_width, got {weights_out.shape[1]} vs {output_width}")
    torch._check(weights_in.shape[0] == weights_out.shape[0] == depth, 
                 f"Expected weights_in.shape[0] == weights_out.shape[0] == depth, got {weights_in.shape[0]} vs {weights_out.shape[0]} vs {depth}")
    
    # Placeholder for the return shape
    return torch.empty((x.shape[0], output_width), dtype=x.dtype, device=x.device)

