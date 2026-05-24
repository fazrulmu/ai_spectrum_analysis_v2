import pandas as pd

try:
    df = pd.read_csv('debug_prediction_input.csv')
    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist()[:20])
    print("Last Columns:", df.columns.tolist()[-20:])
    print("First Row Values (Head):", df.iloc[0].tolist()[:20])
    print("First Row Values (Tail):", df.iloc[0].tolist()[-20:])
    
    # Check spectral columns order
    spectral_cols = [c for c in df.columns if c not in ['num_peaks', 'fwhm', 'lambda_max', 'log_epsilon']]
    if spectral_cols:
        print(f"First spectral col: {spectral_cols[0]}")
        print(f"Last spectral col: {spectral_cols[-1]}")
        
        # Check if values are normalized (should be between 0 and 1 usually, or standardized)
        # Wait, this is INPUT to scaler, so it should be raw-ish (0-1 absorbance usually)
        vals = df[spectral_cols].iloc[0].values
        print(f"Min spectral val: {vals.min()}")
        print(f"Max spectral val: {vals.max()}")

except Exception as e:
    print(e)
