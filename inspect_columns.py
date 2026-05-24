import pandas as pd

ir_path = 'data/for_train/universal_training_dataset_IR.csv'
try:
    ir_df = pd.read_csv(ir_path, nrows=5) # Read only first few rows
    ir_feature_cols = [col for col in ir_df.columns if col.startswith('num_peaks') or col.startswith('fwhm') or col.replace('.', '', 1).isdigit()]
    
    print(f"Total columns in CSV: {len(ir_df.columns)}")
    print(f"Selected feature columns: {len(ir_feature_cols)}")
    print("First 20 selected columns:", ir_feature_cols[:20])
    print("Last 20 selected columns:", ir_feature_cols[-20:])
    
    # Check for unexpected columns
    expected_spectral = 3601
    expected_meta = 2 # num_peaks, fwhm
    total_expected = expected_spectral + expected_meta
    
    if len(ir_feature_cols) > total_expected:
        print("\n--- Unexpected Columns ---")
        # Assuming spectral columns are just numbers, let's see what else is there
        non_spectral = [col for col in ir_feature_cols if not col.replace('.', '', 1).isdigit()]
        print("Non-numeric feature columns:", non_spectral)
        
        # Check if there are numeric columns that are NOT part of the grid
        # This is harder to check without knowing the exact grid, but let's print some "middle" columns
        print("Middle 20 selected columns:", ir_feature_cols[len(ir_feature_cols)//2 - 10 : len(ir_feature_cols)//2 + 10])

except Exception as e:
    print(e)
