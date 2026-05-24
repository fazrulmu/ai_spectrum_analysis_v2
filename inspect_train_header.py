import pandas as pd
import sys

try:
    file_path = 'data/for_train/universal_training_dataset_IR.csv'
    print(f"Reading {file_path}...")
    
    # Read just the header
    df = pd.read_csv(file_path, nrows=0)
    cols = list(df.columns)
    
    print(f"Total columns: {len(cols)}")
    
    # Check for spectral columns (float-like)
    spectral_cols = []
    metadata_cols = []
    
    for c in cols:
        try:
            float(c)
            spectral_cols.append(c)
        except ValueError:
            metadata_cols.append(c)
            
    print(f"Spectral columns: {len(spectral_cols)}")
    print(f"Metadata columns: {len(metadata_cols)}")
    
    if spectral_cols:
        print(f"First 5 spectral cols: {spectral_cols[:5]}")
        print(f"Last 5 spectral cols: {spectral_cols[-5:]}")
        
    if metadata_cols:
        print(f"Metadata cols: {metadata_cols}")
        
except Exception as e:
    print(f"Error: {e}")
