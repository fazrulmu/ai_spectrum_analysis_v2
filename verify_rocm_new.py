import torch
import sys
import platform
import os

print(f"Python Version: {sys.version}")
print(f"Platform: {platform.system()} {platform.release()}")
print(f"HSA_OVERRIDE_GFX_VERSION: {os.environ.get('HSA_OVERRIDE_GFX_VERSION')}")

try:
    print(f"PyTorch Version: {torch.__version__}")
    print(f"ROCm Version: {torch.version.hip}")
except Exception as e:
    print(f"Error importing torch: {e}")
    sys.exit(1)

if torch.cuda.is_available():
    print(f"CUDA Available: Yes")
    device_count = torch.cuda.device_count()
    print(f"Device Count: {device_count}")
    for i in range(device_count):
        print(f"Device {i}: {torch.cuda.get_device_name(i)}")
        try:
            props = torch.cuda.get_device_properties(i)
            print(f"  - Total Memory: {props.total_memory / (1024**3):.2f} GB")
        except Exception as e:
            print(f"  - Error getting properties: {e}")
            
    # Test Tensor Allocation
    print("\nAttempting Tensor Allocation on GPU...")
    try:
        x = torch.rand(1000, 1000).cuda()
        y = torch.rand(1000, 1000).cuda()
        print("✅ Tensor allocation successful.")
        
        print("Attempting Matrix Multiplication...")
        z = torch.matmul(x, y)
        print("✅ Matrix multiplication successful.")
        print(f"Result shape: {z.shape}")
        
    except Exception as e:
        print(f"❌ Tensor Operation Failed: {e}")
        sys.exit(1)

else:
    print("❌ CUDA/ROCm not available.")
    sys.exit(1)
