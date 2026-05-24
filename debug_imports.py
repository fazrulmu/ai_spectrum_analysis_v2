import sys
import os
from pathlib import Path

print("Starting debug script...")
try:
    sys.path.insert(0, str(Path(".").resolve()))
    print(f"Sys path: {sys.path}")
    
    print("Importing src.data.data_processing...")
    from src.data.data_processing import parse_jdx, preprocess_spectrum
    print("Import successful!")
    
    print("Importing preprocess_universal...")
    import preprocess_universal
    print("preprocess_universal imported!")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
