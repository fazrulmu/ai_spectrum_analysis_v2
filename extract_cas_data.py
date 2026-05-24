import pandas as pd
import argparse
import os

def extract_by_cas(file_path, cas_number, output_file=None):
    print(f"Loading {file_path}...")
    try:
        # Read only necessary columns first to filter? No, need all data.
        # Assuming dataset fits in memory.
        df = pd.read_csv(file_path)
        
        if 'cas_number' not in df.columns:
            print(f"Error: 'cas_number' column not found in {file_path}")
            print(f"Available columns: {df.columns.tolist()[:10]}...")
            return

        # Filter
        filtered_df = df[df['cas_number'] == cas_number]
        
        if filtered_df.empty:
            print(f"No records found for CAS: {cas_number}")
            return

        print(f"Found {len(filtered_df)} records for CAS {cas_number}")
        
        # Analyze values (Check for Shift Bug)
        # Identify spectral columns (numeric, not metadata)
        metadata_cols = ['file_id', 'cas_number', 'spectrum_type', 'state', 'matrix_sample', 'abs_matrix', 'state_raw', 'num_peaks', 'fwhm']
        spectral_cols = [c for c in df.columns if c not in metadata_cols and c.replace('.', '', 1).isdigit()]
        
        if spectral_cols:
            vals = filtered_df[spectral_cols].values
            max_val = vals.max()
            min_val = vals.min()
            mean_val = vals.mean()
            
            print(f"\n--- Data Stats for {cas_number} ---")
            print(f"Max Value: {max_val:.4f}")
            print(f"Min Value: {min_val:.4f}")
            print(f"Mean Value: {mean_val:.4f}")
            
            if max_val > 1.5:
                print("\n⚠️  WARNING: Max value > 1.5 detected.")
                print("   This suggests the data might still have the 'Transmittance Shift' bug (+2.0 offset).")
                print("   If this is Absorbance, it should typically be < 1.5.")
            else:
                print("\n✅ Data range looks normal for Absorbance (0-1.5).")

        # Save
        if output_file:
            filtered_df.to_csv(output_file, index=False)
            print(f"\nExtracted data saved to: {output_file}")
        else:
            out_name = f"extracted_{cas_number}.csv"
            filtered_df.to_csv(out_name, index=False)
            print(f"\nExtracted data saved to: {out_name}")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract rows by CAS number")
    parser.add_argument("--file", default="data/for_train/universal_training_dataset_UV.csv", help="Path to CSV file")
    parser.add_argument("--cas", required=True, help="CAS Number to extract (e.g., 75-09-2)")
    
    args = parser.parse_args()
    
    extract_by_cas(args.file, args.cas)
