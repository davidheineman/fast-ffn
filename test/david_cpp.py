import torch
import extension_cpp


def reference_muladd(a, b, c):
    return a * b + c


def sample_inputs(device, *, requires_grad=False):
    def make_tensor(*size):
        return torch.randn(size, device=device, requires_grad=requires_grad)

    def make_nondiff_tensor(*size):
        return torch.randn(size, device=device, requires_grad=False)

    return [
        [make_tensor(3), make_tensor(3), 1],
        [make_tensor(20), make_tensor(20), 3.14],
        [make_tensor(20), make_nondiff_tensor(20), -123],
        [make_nondiff_tensor(2, 3), make_tensor(2, 3), -0.3],
    ]

def main():
    device = 'cpu'

    samples = sample_inputs(device)
    for args in samples:
        result = extension_cpp.ops.mymuladd(*args)
        expected = reference_muladd(*args)
        torch.testing.assert_close(result, expected)

    print(result)


if __name__ == '__main__': main()
