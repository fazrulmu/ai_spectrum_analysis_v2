
# debug_pipeline.py

import yaml
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

# Impor fungsi-fungsi inti dari proyek
from src.data_processing import (
    parse_jdx,
    poly_baseline,
    morph_baseline,
    baseline_als,
    smooth_signal,
    choose_baseline_and_correct,
    estimate_morph_size,
    hybrid_poly_als_baseline
)

from scipy.interpolate import interp1d
from scipy.signal import savgol_filter, find_peaks


def debug_pipeline(file_path, config, spectrum_type):
    """
    Memproses satu file spektrum langkah demi langkah dan memvisualisasikan setiap tahap.
    """
    print(f"--- Memulai Debug Pipeline untuk: {os.path.basename(file_path)} ---")
    
    # --- TAHAP 1: PARSING ---
    raw_data = parse_jdx(file_path)
    if not raw_data:
        print("❌ GAGAL: Parsing file tidak menghasilkan data.")
        return
    print("✅ Tahap 1: Parsing berhasil.")
    
    # --- TAHAP 2: PRA-PEMROSESAN AWAL (UNIT & KONVERSI) ---
    x, y, metadata = raw_data['x'], raw_data['y'], raw_data['metadata']
    df = pd.DataFrame({'x': x, 'y': y}).dropna()
    df_sorted = df.groupby('x', as_index=False).mean().sort_values(by='x').reset_index(drop=True)
    x_clean, y_clean = df_sorted['x'].values, df_sorted['y'].values
    
    x_units = metadata.get('xunits', '').lower()
    y_units = metadata.get('yunits', '').lower()
    if spectrum_type == 'ir':
        if 'micrometer' in x_units:
            x_standard = 10000 / x_clean
        else:
            x_standard = x_clean
        if 'transmittance' in y_units:
            T = np.clip(y_clean / 100.0, 1e-9, 1.0)  # ubah %T ke fraksi
            y_standard = -np.log10(T)               # absorbansi
        else:
            y_standard = y_clean
        if x_standard[0] < x_standard[-1]:
            x_final, y_final = x_standard[::-1], y_standard[::-1]
        else:
            x_final, y_final = x_standard, y_standard
        grid_config = config['preprocessing']['ir_grid']
        x_label, y_label = 'Wavenumber (cm⁻¹)', 'Absorbance'
    else:  # uv
        if 'transmittance' in y_units:
            T = np.clip(y_clean / 100.0, 1e-9, 1.0)
            y_final = -np.log10(T)
        else:
            y_final = y_clean
        x_final = x_clean
        grid_config = config['preprocessing']['uv_grid']
        x_label, y_label = 'Wavelength (nm)', 'Log Epsilon'
    
    print("✅ Tahap 2: Standarisasi unit berhasil.")

    # --- TAHAP 3: RESAMPLING & SMOOTHING ---
    grid = np.linspace(grid_config['start'], grid_config['stop'], grid_config['num_points'])
    y_resampled = interp1d(x_final, y_final, bounds_error=False, fill_value=0.0)(grid)

# smoothing pertama sebelum baseline
    y_smoothed = smooth_signal(y_resampled, window_length=11, polyorder=3)
    print("✅ Tahap 3: Resampling & Smoothing berhasil.")

    # --- TAHAP 4: KOREKSI BASELINE ---
        # --- TAHAP 4: KOREKSI BASELINE (Otomatis pilih Morph / Fallback) ---
    try:
        size = estimate_morph_size(y_resampled)
        baseline, y_corrected, method_used = choose_baseline_and_correct(
            grid, y_resampled, morph_size=size, poly_order=3
        )
        print(f"✅ Tahap 4: Koreksi Baseline berhasil (Metode = {method_used}, size={size})")

    except Exception as e:
        print(f"⚠️ Tahap 4: Baseline correction gagal ({e}), fallback ke data smoothed.")
        baseline = np.zeros_like(y_smoothed)
        y_corrected = y_smoothed
        method_used = "None"

    # --- TAHAP 5: NORMALISASI & DETEKSI PUNCAK ---
    y_normalized = (y_corrected - y_corrected.min()) / (y_corrected.max() - y_corrected.min() + 1e-9)
    peaks, props = find_peaks(y_normalized, prominence=0.015)
    print(f"✅ Tahap 5: Normalisasi & Deteksi Puncak selesai. Ditemukan {len(peaks)} puncak.")

    # --- PLOTTING ---
    
    
        
        # Plot 3: Baseline hasil metode otomatis
            # --- PLOTTING DALAM 1 PLOT ---
    plt.figure(figsize=(15, 8))
    plt.title(f'Pipeline Debug untuk {os.path.basename(file_path)}', fontsize=18)

    # Spektrum asli (raw setelah konversi unit)
    plt.plot(x_final, y_final, label='Data Mentah (Setelah Konversi Unit)', alpha=0.5)

    # Data setelah resampling & smoothing
    plt.plot(grid, y_smoothed, label='Data Resampled & Smoothed', alpha=0.7)

    # Baseline
    plt.plot(grid, baseline, '--', label=f'Baseline ({method_used})', alpha=0.9)

    # Data setelah koreksi baseline
    plt.plot(grid, y_corrected, label=f'Koreksi Baseline ({method_used})', linewidth=2)

    # Data final normalisasi (opsional, kalau mau tampilkan)
    plt.plot(grid, y_normalized, label='Data Final (Normalisasi)', alpha=0.7)

    # Puncak terdeteksi
    if peaks.size > 0:
        plt.plot(grid[peaks], y_normalized[peaks], 'x', color='red', markersize=8, label=f'{len(peaks)} Puncak')

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if spectrum_type == 'ir':
        plt.xlim(max(grid), min(grid))  # IR biasanya dibalik
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Debug pipeline preprocessing spektrum.")
    parser.add_argument("--file", required=True, help="Path ke file JDX yang akan di-debug.")
    parser.add_argument("--type", required=True, choices=["ir", "uv"], help="Tipe spektrum.")
    args = parser.parse_args()

    with open('main_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
        
    debug_pipeline(args.file, config, args.type)

