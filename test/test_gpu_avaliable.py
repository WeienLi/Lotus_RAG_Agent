import torch


def test_cuda():
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Version: {torch.version.cuda}")
    if not torch.cuda.is_available():
        print("CUDA is not available. Please check your CUDA installation.")
        return

    print(f"Number of available CUDA devices: {torch.cuda.device_count()}")
    print(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")

    x = torch.rand(3, 3).cuda()
    print("Tensor on GPU:")
    print(x)

    y = torch.rand(3, 3).cuda()
    z = x + y
    print("Result of tensor addition on GPU:")
    print(z)

    print(f"Memory Allocated: {torch.cuda.memory_allocated(0)} bytes")
    print(f"Memory Cached: {torch.cuda.memory_reserved(0)} bytes")


if __name__ == "__main__":
    test_cuda()
