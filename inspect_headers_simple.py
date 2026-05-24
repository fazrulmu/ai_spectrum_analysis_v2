import csv

ir_path = 'data/for_train/universal_training_dataset_IR.csv'
try:
    with open(ir_path, 'r', encoding='utf-8', errors='replace') as f:
        reader = csv.reader(f)
        header = next(reader)
    
    print(f"Total columns: {len(header)}")
    
    selected = [col for col in header if col.startswith('num_peaks') or col.startswith('fwhm') or col.replace('.', '', 1).isdigit()]
    print(f"Selected columns: {len(selected)}")
    
    # Check for columns that are NOT in the expected grid range (4000-400)
    extra_cols = []
    for col in selected:
        if col in ['num_peaks', 'fwhm']: continue
        try:
            val = float(col)
            if val > 4000 or val < 400:
                extra_cols.append(col)
        except:
            extra_cols.append(col)
            
    print(f"Columns outside 400-4000 range or non-numeric: {len(extra_cols)}")
    if extra_cols:
        print("Sample extra columns:", extra_cols[:20])

except Exception as e:
    print(f"Error: {e}")
