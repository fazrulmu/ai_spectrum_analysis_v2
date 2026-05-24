# debug_spectrum.py

# debug_spectrum.py (Versi Baru dengan Opsi Tuning)

import yaml
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

# Impor semua fungsi yang relevan
from src.data_processing import parse_jdx, baseline_als, preprocess_spectrum
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

def debug_preprocess_spectrum(data, config, spectrum_type, normalize=True, lam_override=None, p_override=None):
    print("\n--- MEMULAI PROSES PREPROCESSING ---")
    
    # ... (Bagian 1-5 dari parsing dan resampling tidak berubah)
    if not data or 'x' not in data or 'y' not in data or len(data['x']) < 5: print("❌ GAGAL: Data input tidak valid."); return None
    x, y, metadata = data['x'], data['y'], data['metadata']; print(f"1. Data Awal: {len(x)} titik. Rentang X: [{x.min():.2f}, {x.max():.2f}]")
    df = pd.DataFrame({'x': x, 'y': y}).dropna();
    if df.empty: print("❌ GAGAL: DataFrame kosong."); return None
    df_cleaned = df.groupby('x', as_index=False).mean(); df_sorted = df_cleaned.sort_values(by='x').reset_index(drop=True)
    if len(df_sorted) < 5: print("❌ GAGAL: Data terlalu pendek."); return None
    x_clean, y_clean = df_sorted['x'].values, df_sorted['y'].values; print(f"2. Data Setelah Dibersihkan: {len(x_clean)} titik. Rentang X: [{x_clean.min():.2f}, {x_clean.max():.2f}]")
    if spectrum_type == 'ir':
        x_units = metadata.get('xunits', 'N/A').lower(); print(f"3. Unit terdeteksi: X='{x_units}'")
        if 'micrometer' in x_units or 'micron' in x_units: x_clean[x_clean == 0] = 1e-9; x_standard = 10000 / x_clean; print(f"4. Data Setelah Konversi Unit: Rentang X baru: [{x_standard.min():.2f}, {x_standard.max():.2f}] cm⁻¹")
        else: x_standard = x_clean; print("4. Data Setelah Konversi Unit: Tidak ada konversi.")
        if 'transmittance' in metadata.get('yunits', '').lower(): y_standard = 2 - np.log10(np.clip(y_clean, 1e-9, 100))
        else: y_standard = y_clean
        if x_standard[0] < x_standard[-1]: x_final, y_final = x_standard[::-1], y_standard[::-1]
        else: x_final, y_final = x_standard, y_standard
        grid_config = config['preprocessing']['ir_grid']; grid = np.linspace(grid_config['start'], grid_config['stop'], grid_config['num_points'])
    else: # uv
        x_final, y_final = x_clean, y_clean; grid_config = config['preprocessing']['uv_grid']; grid = np.linspace(grid_config['start'], grid_config['stop'], grid_config['num_points'])
    print(f"5. Grid Target: {len(grid)} titik. Rentang: [{grid.min():.2f}, {grid.max():.2f}]")
    f = interp1d(x_final, y_final, bounds_error=False, fill_value=0.0); y_resampled = f(grid)
    print(f"6. Data Setelah Resampling: Rata-rata Y = {np.mean(y_resampled):.4f}, Std Dev Y = {np.std(y_resampled):.4f}")
    if len(y_resampled) > 7: y_smoothed = savgol_filter(y_resampled, window_length=7, polyorder=2)
    else: y_smoothed = y_resampled

    # --- INTI EKSPERIMEN ---
    state = metadata.get('state', '').lower()
    
    # Gunakan parameter dari command line jika ada, jika tidak, gunakan logika adaptif
    lam_to_use = lam_override if lam_override is not None else (1e5 if 'gas' in state else 1e6)
    p_to_use = p_override if p_override is not None else (0.05 if 'gas' in state else 0.01)
    print(f"7. Parameter Baseline Digunakan: lam={lam_to_use}, p={p_to_use}")
    
    baseline = baseline_als(y_smoothed, lam=lam_to_use, p=p_to_use)
    if np.any(np.isnan(baseline)): y_final_processed = y_smoothed
    else: y_final_processed = y_smoothed - baseline
    print(f"8. Data Setelah Koreksi Baseline: Rata-rata Y = {np.mean(y_final_processed):.4f}, Std Dev Y = {np.std(y_final_processed):.4f}")

    if normalize:
        min_val, max_val = np.min(y_final_processed), np.max(y_final_processed)
        if max_val > min_val: y_out = (y_final_processed - min_val) / (max_val - min_val)
        else: y_out = y_final_processed
    else: y_out = y_final_processed
    print(f"9. Data Final: Rata-rata Y = {np.mean(y_out):.4f}, Std Dev Y = {np.std(y_out):.4f}")

    # Plotting untuk visualisasi
    plt.figure(figsize=(15, 7))
    plt.plot(grid, y_out, label=f'Hasil Final (lam={lam_to_use})')
    plt.plot(grid, y_smoothed, label='Sebelum Koreksi Baseline', linestyle='-', alpha=0.5)
    plt.title(f"Hasil Debug Preprocessing (lam={lam_to_use}, p={p_to_use})", fontsize=16)
    plt.xlabel('Wavenumber (cm⁻¹)'); plt.ylabel('Intensity (Processed)'); plt.legend(); plt.grid(True)
    plt.show()
    return True

def plot_uv_spectrum(raw_data, config):
    """
    Processes and plots a UV-Vis spectrum, showing raw absorbance vs. wavelength.
    """
    print("\n--- MEMULAI PLOTTING UV-VIS SPECTRUM ---")

    # 1. Preprocess the spectrum to get absorbance values without normalization
    # The 'log_epsilon' column from preprocess_spectrum contains the absorbance values when normalize=False
    processed_df = preprocess_spectrum(raw_data, config, 'uv', normalize=False)
    if processed_df is None or processed_df.empty:
        print("❌ GAGAL: Preprocessing tidak menghasilkan data.")
        return

    wavelengths = processed_df['wavelength'].values
    absorbance_values = processed_df['log_epsilon'].values

    # 2. Plotting the spectrum
    plt.figure(figsize=(12, 6))
    plt.plot(wavelengths, absorbance_values, label='Raw Absorbance')
    
    plt.title("UV-Vis Spectrum of My Compound")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Absorbance")
    plt.xlim(200, 320) # Focus on the 200-320 nm region
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Debug dan tuning parameter preprocessing.")
    parser.add_argument("--file", required=True, help="Path ke file JDX.")
    parser.add_argument("--type", required=True, choices=["ir", "uv"], help="Tipe spektrum.")
    parser.add_argument("--lam", type=float, help="(Opsional) Ganti nilai lambda untuk baseline_als.")
    parser.add_argument("--p", type=float, help="(Opsional) Ganti nilai p untuk baseline_als.")
    args = parser.parse_args()

    with open('main_config.yaml', 'r') as f: config = yaml.safe_load(f)

    raw_data = parse_jdx(args.file)
    if raw_data:
        if args.type == 'ir':
            debug_preprocess_spectrum(raw_data, config, args.type, lam_override=args.lam, p_override=args.p)
        elif args.type == 'uv':
            plot_uv_spectrum(raw_data, config)
    else:
        print("❌ GAGAL: Parsing file tidak menghasilkan data.")