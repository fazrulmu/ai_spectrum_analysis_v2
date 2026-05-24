import pandas as pd
from pathlib import Path

ir_path = Path("data/for_train/universal_training_dataset_IR.csv")
uv_path = Path("data/for_train/universal_training_dataset_UV.csv")

try:
    ir_df = pd.read_csv(ir_path, nrows=5)
    print("IR Columns:", ir_df.columns.tolist())
except Exception as e:
    print(f"Error reading IR: {e}")

try:
    uv_df = pd.read_csv(uv_path, nrows=5)
    print("UV Columns:", uv_df.columns.tolist())
except Exception as e:
    print(f"Error reading UV: {e}")
