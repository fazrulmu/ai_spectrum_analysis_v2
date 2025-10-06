# src/data_analyzer/run_data_audit.py

import os
import yaml
import glob
import re
import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.signal import find_peaks
import argparse

# Impor relatif dari dalam paket 'src'
from ..data_processing import parse_jdx, preprocess_spectrum
from ..auto_labeler import SPECTRAL_RULES, UV_SPECTRAL_RULES

def audit_dataset(config, output_dir):
    """
    Menjalankan audit komprehensif pada seluruh dataset JDX.
    """
    paths = config['paths']
    
    # Inisialisasi kolektor data
    ir_peak_data = []
    uv_peak_data = []
    bad_baseline_files = []

    all_jdx_files = glob.glob(os.path.join(paths['raw_data_dir'], '**', '*.jdx'), recursive=True)
    print(f"Memulai audit pada {len(all_jdx_files)} file spektrum...")

    for file_path in tqdm(all_jdx_files, desc="Auditing Dataset"):
        try:
            raw_data = parse_jdx(file_path)
            if not raw_data: continue

            metadata = raw_data.get('metadata', {})
            filename = os.path.basename(file_path).lower()
            spectrum_type = 'ir' if re.search(r'(ftir|ft-ir|ir)', filename) else ('uv' if re.search(r'uv[\-\s]?vis', filename) else None)
            if not spectrum_type: continue

            # --- Audit Baseline Khusus untuk IR ---
            if spectrum_type == 'ir' and 'transmittance' in metadata.get('yunits', '').lower():
                y_abs = 2 - np.log10(np.clip(raw_data['y'], 1e-9, 100))
                baseline_metric = np.min(y_abs)
                if baseline_metric > 0.15: # Threshold ini bisa disesuaikan
                    bad_baseline_files.append({
                        "file": os.path.basename(file_path),
                        "baseline_min_abs": round(baseline_metric, 3),
                        "resolution": metadata.get('resolution', 'N/A')
                    })

            # --- Lanjutkan ke Analisis Puncak ---
            normalize_for_uv = False if spectrum_type == 'uv' else True
            processed_df = preprocess_spectrum(raw_data, config, spectrum_type, normalize=normalize_for_uv)
            if processed_df is None: continue

            y_col = 'absorbance' if spectrum_type == 'ir' else 'log_epsilon'
            x_col = 'wavenumber' if spectrum_type == 'ir' else 'wavelength'
            y_vals = processed_df[y_col].values
            x_vals = processed_df[x_col].values
            y_vals_norm = (y_vals - y_vals.min()) / (y_vals.max() - y_vals.min() + 1e-9)
            
            peaks, props = find_peaks(y_vals_norm, prominence=0.01, height=0.01, width=1)
            
            if peaks.size > 0:
                rules = SPECTRAL_RULES if spectrum_type == 'ir' else UV_SPECTRAL_RULES
                for i, peak_idx in enumerate(peaks):
                    peak_location = x_vals[peak_idx]
                    assigned_label = 'unassigned'
                    for rule in rules:
                        if rule['range'][0] <= peak_location <= rule['range'][1]:
                            assigned_label = rule['group']; break
                    
                    peak_info = {
                        'group': assigned_label,
                        'peak_location': peak_location,
                        'peak_height': props['peak_heights'][i],
                        'peak_prominence': props['prominences'][i],
                        'peak_width': props['widths'][i],
                        'actual_intensity': y_vals[peak_idx],
                        'file_path': os.path.basename(file_path)
                    }
                    if spectrum_type == 'ir': ir_peak_data.append(peak_info)
                    else: uv_peak_data.append(peak_info)
        except Exception:
            continue
            
    # Simpan hasil ke file CSV
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    if ir_peak_data:
        df_ir = pd.DataFrame(ir_peak_data)
        output_path = os.path.join(output_dir, f"ir_peak_properties_{timestamp}.csv")
        df_ir.to_csv(output_path, index=False)
        print(f"\n✅ Analisis IR selesai. Data disimpan di: {output_path}")

    if uv_peak_data:
        df_uv = pd.DataFrame(uv_peak_data)
        output_path = os.path.join(output_dir, f"uv_peak_properties_{timestamp}.csv")
        df_uv.to_csv(output_path, index=False)
        print(f"\n✅ Analisis UV selesai. Data disimpan di: {output_path}")

    if bad_baseline_files:
        df_baseline = pd.DataFrame(bad_baseline_files)
        output_path = os.path.join(output_dir, f"bad_baseline_files_{timestamp}.csv")
        df_baseline.to_csv(output_path, index=False)
        print(f"\n✅ Analisis baseline selesai. Data disimpan di: {output_path}")

def main(config_path="main_config.yaml", output_dir="src/data_analyzer/audit_results"):
    print("--- 🔬 Memulai Modul Audit Kualitas Data ---")
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"❌ Error: File konfigurasi '{config_path}' tidak ditemukan.")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    audit_dataset(config, output_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Jalankan audit kualitas dataset spektral.")
    parser.add_argument('--config', default='main_config.yaml', help='Path ke file konfigurasi.')
    args = parser.parse_args()
    main(config_path=args.config)