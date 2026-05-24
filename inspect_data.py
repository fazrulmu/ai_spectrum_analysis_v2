import pandas as pd

def inspect_csv(file_path):
    print(f"--- Inspecting {file_path} ---")
    try:
        df = pd.read_csv(file_path, nrows=5)
        print("Columns:", list(df.columns))
        print("First 2 rows:")
        print(df.head(2))
        print("Data types:")
        print(df.dtypes)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")

inspect_csv('data/for_train/universal_training_dataset_IR.csv')
inspect_csv('data/for_train/universal_training_dataset_UV.csv')
