import torch
import time
import sys

print(f"Python: {sys.version}")
print(f"PyTorch: {torch.__version__}")

device = torch.device("cuda")
try:
    print(f"Using Device: {torch.cuda.get_device_name(0)}")
    props = torch.cuda.get_device_properties(0)
    print(f"Device Properties: {props}")
except Exception as e:
    print(f"Error getting device info: {e}")

print("\n--- TEST 1: Tensor Creation ---")
try:
    a = torch.randn(1024, 1024).to(device)
    b = torch.randn(1024, 1024).to(device)
    print("Tensors created on GPU.")
except Exception as e:
    print(f"FAILED: {e}")
    sys.exit(1)

print("\n--- TEST 2: Matrix Multiplication (No MIOpen) ---")
try:
    start = time.time()
    c = torch.matmul(a, b)
    torch.cuda.synchronize() # Wait for completion
    end = time.time()
    print(f"Matmul successful! Time: {end - start:.4f}s")
    print(f"Result shape: {c.shape}")
except Exception as e:
    print(f"FAILED: {e}")
    sys.exit(1)

print("\n✅ GPU is working for Basic Math (Linear/Dense layers)!")
