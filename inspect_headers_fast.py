import pandas as pd

ir_path = 'data/for_train/universal_training_dataset_IR.csv'
try:
    # Read only the header
    with open(ir_path, 'r') as f:
        header = f.readline().strip().split(',')
    
    print(f"Total columns: {len(header)}")
    
    # Simulate selection logic
    selected = [col for col in header if col.startswith('num_peaks') or col.startswith('fwhm') or col.replace('.', '', 1).isdigit()]
    print(f"Selected columns: {len(selected)}")
    
    # Find what's extra
    # Expected: 3601 (grid) + 2 (meta) = 3603
    # Actual: 3913
    # Diff: 310
    
    # Print first few and last few selected
    print("First 10:", selected[:10])
    print("Last 10:", selected[-10:])
    
    # Check for columns that are NOT in the expected grid range (4000-400)
    # Assuming grid points are integers or floats
    extra_cols = []
    for col in selected:
        if col in ['num_peaks', 'fwhm']: continue
        try:
            val = float(col)
            if val > 4000 or val < 400:
                extra_cols.append(col)
            # Check if it's a non-integer grid point if we expect integers?
            # Config says 1 cm-1 interval, so mostly integers.
        except:
            extra_cols.append(col)
            
    print(f"Columns outside 400-4000 range or non-numeric: {len(extra_cols)}")
    if extra_cols:
        print("Sample extra columns:", extra_cols[:20])

except Exception as e:
    print(e)
