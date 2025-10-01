# run_data_audit.py

import os
import yaml
import glob
import re
import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import seaborn as sns

# Impor fungsi dari proyek Anda
from src.data_processing import parse_jdx, preprocess_spectrum

def audit_dataset(config):
    """
    Menjalankan audit komprehensif pada seluruh dataset JDX.
    Menganalisis properti puncak IR & UV, dan mengidentifikasi file dengan baseline buruk.
    """
    paths = config['paths']
    
    # Inisialisasi kolektor data
    ir_prominences, ir_heights = [], []
    uv_intensities, uv_peak_counts = [], []
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
                # Tandai file jika baseline minimumnya jauh di atas nol
                if baseline_metric > 0.15: # Threshold ini bisa disesuaikan
                    bad_baseline_files.append({
                        "file": os.path.basename(file_path),
                        "baseline_min_abs": baseline_metric,
                        "resolution": metadata.get('resolution', 'N/A')
                    })

            # --- Lanjutkan ke Analisis Puncak ---
            processed_df = preprocess_spectrum(raw_data, config, spectrum_type, normalize=True)
            if processed_df is None: continue

            y_col = 'absorbance' if spectrum_type == 'ir' else 'log_epsilon'
            y_vals = processed_df[y_col].values
            if np.all(np.isnan(y_vals)) or len(y_vals) == 0: continue

            peaks, props = find_peaks(y_vals, prominence=0.01, height=0.01) # Gunakan threshold rendah untuk menangkap semua
            
            if peaks.size > 0:
                if spectrum_type == 'ir':
                    ir_prominences.extend(props['prominences'])
                    ir_heights.extend(props['peak_heights'])
                else: # uv
                    uv_peak_counts.append(len(peaks))
                    uv_intensities.extend(props['peak_heights']) # Untuk UV, height = log_epsilon
        except Exception:
            continue
            
    return {
        "ir_prominences": np.array(ir_prominences),
        "ir_heights": np.array(ir_heights),
        "uv_peak_counts": np.array(uv_peak_counts),
        "uv_intensities": np.array(uv_intensities),
        "bad_baseline_files": bad_baseline_files
    }

def generate_audit_report(audit_results):
    """Mencetak laporan hasil audit ke konsol."""
    print("\n" + "="*60)
    print(" LAPORAN AUDIT KUALITAS DATASET SPEKTRAL")
    print("="*60)

    # Laporan IR
    if audit_results["ir_prominences"].size > 0:
        print("\n--- Analisis Puncak IR ---")
        print(">> Statistik Prominence:")
        print(pd.Series(audit_results["ir_prominences"]).describe(percentiles=[.25, .5, .75, .95]))
        print("\n>> Statistik Height (Normalized):")
        print(pd.Series(audit_results["ir_heights"]).describe(percentiles=[.25, .5, .75, .95, .99]))
    
    # Laporan UV
    if audit_results["uv_peak_counts"].size > 0:
        print("\n\n--- Analisis Puncak UV-Vis ---")
        print(">> Statistik Jumlah Puncak per Spektrum:")
        print(pd.Series(audit_results["uv_peak_counts"]).describe())
        print("\n>> Statistik Intensitas Puncak (log ε):")
        print(pd.Series(audit_results["uv_intensities"]).describe(percentiles=[.25, .5, .75, .95]))

    # Laporan Baseline Bermasalah
    bad_baselines = audit_results["bad_baseline_files"]
    if bad_baselines:
        print("\n\n--- Analisis Lonjakan Baseline (IR) ---")
        print(f"Ditemukan {len(bad_baselines)} file IR dengan potensi baseline yang terangkat/bermasalah:")
        df_baseline = pd.DataFrame(bad_baselines)
        print(df_baseline.to_string(index=False))
    
    print("\n" + "="*60)
    print(" Audit Selesai.")
    print("="*60)


if __name__ == "__main__":
    with open('main_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
        
    audit_results = audit_dataset(config)
    generate_audit_report(audit_results)