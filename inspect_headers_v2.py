import pandas as pd
from pathlib import Path

ir_path = Path("data/for_train/universal_training_dataset_IR.csv")
uv_path = Path("data/for_train/universal_training_dataset_UV.csv")
output_file = Path("csv_headers.txt")

with open(output_file, "w") as f:
    try:
        if ir_path.exists():
            ir_df = pd.read_csv(ir_path, nrows=0) # Read only header
            f.write(f"IR Columns: {list(ir_df.columns)}\n")
        else:
            f.write(f"IR File not found: {ir_path}\n")
    except Exception as e:
        f.write(f"Error reading IR: {e}\n")

    try:
        if uv_path.exists():
            uv_df = pd.read_csv(uv_path, nrows=0) # Read only header
            f.write(f"UV Columns: {list(uv_df.columns)}\n")
        else:
            f.write(f"UV File not found: {uv_path}\n")
    except Exception as e:
        f.write(f"Error reading UV: {e}\n")
