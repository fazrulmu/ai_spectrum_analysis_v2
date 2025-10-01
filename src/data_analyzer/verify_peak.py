# verify_peak.py (Versi Final yang Benar)

import os
import yaml
import glob
import argparse
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from tqdm import tqdm
import numpy as np
# Impor fungsi dari proyek Anda
from src.data_processing import parse_jdx, preprocess_spectrum
from src.auto_labeler import SPECTRAL_RULES

def batch_verify_peaks(input_dir, output_dir, rule_name, config):
    rule = next((r for r in SPECTRAL_RULES if r['group'] == rule_name), None)
    if not rule:
        print(f"Error: Aturan untuk '{rule_name}' tidak ditemukan.")
        return

    print(f"Memulai verifikasi massal untuk aturan '{rule_name}'...")
    os.makedirs(output_dir, exist_ok=True)

    all_files = glob.glob(os.path.join(input_dir, '**', '*.jdx'), recursive=True)
    if not all_files:
        print(f"Error: Tidak ada file .jdx ditemukan di '{input_dir}'.")
        return

    for file_path in tqdm(all_files, desc=f"Verifying '{rule_name}'"):
        try:
            raw_data = parse_jdx(file_path)
            if not raw_data: continue

            processed_df = preprocess_spectrum(raw_data, config, 'ir', normalize=True)
            if processed_df is None: continue

            wavenumbers = processed_df['wavenumber'].values
            absorbances = processed_df['absorbance'].values
            if np.all(np.isnan(absorbances)) or len(absorbances) == 0: continue

            all_peaks_indices, properties = find_peaks(absorbances, prominence=0.02, width=1)
            
            relevant_peaks_indices = []
            if all_peaks_indices.size > 0:
                for i, peak_idx in enumerate(all_peaks_indices):
                    wavenumber_at_peak = wavenumbers[peak_idx]
                    prominence_at_peak = properties['prominences'][i]
                    if rule['range'][0] <= wavenumber_at_peak <= rule['range'][1]:
                        if prominence_at_peak >= rule.get('min_prominence', 0):
                            relevant_peaks_indices.append(peak_idx)

            plt.style.use('seaborn-v0_8-whitegrid')
            plt.figure(figsize=(15, 7))
            
            plt.plot(wavenumbers, absorbances, color='black', linewidth=1, label='Spektrum (Diproses)')
            plt.axvspan(rule['range'][0], rule['range'][1], color='yellow', alpha=0.3, label=f"Rentang Aturan '{rule_name}'")
            
            if relevant_peaks_indices:
                plt.plot(wavenumbers[relevant_peaks_indices], absorbances[relevant_peaks_indices], 
                        'x', color='red', markersize=10, label='Puncak Terdeteksi')

            plt.title(f"Verifikasi '{rule_name}' pada {os.path.basename(file_path)}", fontsize=16)
            plt.xlabel("Wavenumber (cm⁻¹)", fontsize=12)
            plt.ylabel("Absorbance (Normalized)", fontsize=12)
            plt.xlim(max(wavenumbers), min(wavenumbers))
            plt.legend()
            
            output_filename = f"{os.path.splitext(os.path.basename(file_path))[0]}_verify_{rule_name}.png"
            plt.savefig(os.path.join(output_dir, output_filename))
            plt.close()
        
        except Exception as e:
            print(f"Gagal memproses file {file_path}: {e}")
            continue

    print(f"\nVerifikasi selesai. Semua plot disimpan di folder: {output_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Verifikasi Deteksi Puncak Massal.")
    parser.add_argument("--input-dir", required=True, help="Path ke direktori root file JDX.")
    parser.add_argument("--output-dir", required=True, help="Path ke direktori untuk menyimpan plot.")
    parser.add_argument("--rule", required=True, help="Nama 'group' dari aturan yang akan diperiksa.")
    args = parser.parse_args()

    with open('main_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    batch_verify_peaks(args.input_dir, args.output_dir, args.rule, config)