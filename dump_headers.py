import pandas as pd

ir_path = 'data/for_train/universal_training_dataset_IR.csv'
try:
    # Read only header
    df = pd.read_csv(ir_path, nrows=0)
    cols = list(df.columns)
    
    with open('headers.txt', 'w') as f:
        f.write(f"Total columns: {len(cols)}\n")
        f.write(f"Columns: {cols}\n")
        
    # Simulate selection
    selected = [col for col in cols if col.startswith('num_peaks') or col.startswith('fwhm') or col.replace('.', '', 1).isdigit()]
    with open('selected_features.txt', 'w') as f:
        f.write(f"Selected count: {len(selected)}\n")
        f.write(f"First 20: {selected[:20]}\n")
        f.write(f"Last 20: {selected[-20:]}\n")
        
        # Find the extra ones
        # Assuming grid is 4000 to 400
        extras = []
        for col in selected:
            if col in ['num_peaks', 'fwhm']: continue
            try:
                val = float(col)
                if val > 4000 or val < 400:
                    extras.append(col)
            except:
                extras.append(col)
        f.write(f"Extras count: {len(extras)}\n")
        f.write(f"Extras sample: {extras[:50]}\n")

except Exception as e:
    with open('error.txt', 'w') as f:
        f.write(str(e))
