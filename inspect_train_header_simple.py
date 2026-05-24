import csv
import sys

try:
    file_path = 'data/for_train/universal_training_dataset_IR.csv'
    print(f"Reading {file_path}...", flush=True)
    
    with open(file_path, 'r', newline='') as f:
        reader = csv.reader(f)
        header = next(reader)
        
    print(f"Total columns: {len(header)}", flush=True)
    
    spectral_cols = []
    metadata_cols = []
    
    for c in header:
        try:
            float(c)
            spectral_cols.append(c)
        except ValueError:
            metadata_cols.append(c)
            
    print(f"Spectral columns: {len(spectral_cols)}", flush=True)
    print(f"Metadata columns: {len(metadata_cols)}", flush=True)
    
    if spectral_cols:
        print(f"First 5 spectral cols: {spectral_cols[:5]}", flush=True)
        print(f"Last 5 spectral cols: {spectral_cols[-5:]}", flush=True)
        
    if metadata_cols:
        print(f"Metadata cols: {metadata_cols}", flush=True)

except Exception as e:
    print(f"Error: {e}", flush=True)
