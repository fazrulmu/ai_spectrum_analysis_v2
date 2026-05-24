import pandas as pd
import sys

try:
    # Read only the header (0 rows)
    df = pd.read_csv('data/for_train/universal_training_dataset_IR.csv', nrows=0)
    
    with open('header_info.txt', 'w') as f:
        f.write(f"All Columns: {df.columns.tolist()}\n")
        
        # Simulate the filtering logic from train_test.py
        ir_feature_cols = [col for col in df.columns if col.startswith('num_peaks') or col.startswith('fwhm') or col.replace('.', '', 1).isdigit()]
        f.write(f"\nFiltered Feature Cols (First 10): {ir_feature_cols[:10]}\n")
        f.write(f"Filtered Feature Cols (Last 10): {ir_feature_cols[-10:]}\n")
        
    print("Header info written to header_info.txt")
    
except Exception as e:
    with open('header_info.txt', 'w') as f:
        f.write(f"Error: {e}")
    print(e)
