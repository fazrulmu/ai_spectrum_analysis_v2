import sys
print("Python version:", sys.version)
print("Hello from test_env.py")
try:
    import pandas as pd
    print("Pandas imported successfully")
    import numpy as np
    print("Numpy imported successfully")
    import tensorflow as tf
    print("TensorFlow imported successfully")
except ImportError as e:
    print(f"ImportError: {e}")
except Exception as e:
    print(f"Error: {e}")
