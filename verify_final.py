import torch
import sys

print("="*60)
print("   ROCm / PyTorch Verification (Ubuntu Container)")
print("="*60)

try:
    print(f"PyTorch Version   : {torch.__version__}")
    print(f"ROCm Version      : {torch.version.hip}")
    print(f"CUDA Available    : {torch.cuda.is_available()}")
    
    device_count = torch.cuda.device_count()
    print(f"GPU Device Count  : {device_count}")

    if device_count > 0:
        device_name = torch.cuda.get_device_name(0)
        print(f"GPU Device Name   : {device_name}")
        
        print("\n[TEST] Performing Tensor Addition on GPU...")
        x = torch.tensor([10.0, 20.0, 30.0]).cuda()
        y = torch.tensor([1.0, 2.0, 3.0]).cuda()
        z = x + y
        
        print(f"   Input X : {x.tolist()}")
        print(f"   Input Y : {y.tolist()}")
        print(f"   Result Z: {z.tolist()}")
        
        expected = [11.0, 22.0, 33.0]
        if z.tolist() == expected:
            print("\n[SUCCESS] ROCm GPU Acceleration is WORKING! \N{ROCKET}")
            sys.exit(0)
        else:
            print("\n[FAILURE] Math calculation wrong!")
            sys.exit(1)
    else:
        print("\n[FAILURE] No GPU detected by PyTorch.")
        print("Tip: Try explicit override: HSA_OVERRIDE_GFX_VERSION=11.0.3")
        sys.exit(1)

except Exception as e:
    print(f"\n[ERROR] Verification crashed: {e}")
    sys.exit(1)
