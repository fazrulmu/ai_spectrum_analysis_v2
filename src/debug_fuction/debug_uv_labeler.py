# debug_uv_labeler.py (Versi diperbaiki)

import argparse
import yaml
import matplotlib.pyplot as plt
import numpy as np
import os # <-- TAMBAHKAN BARIS INI

# Impor fungsi-fungsi yang relevan
from src.data_processing import parse_jdx, preprocess_spectrum
from src.auto_labeler_uv import autogenerate_chromophores, UV_SPECTRAL_RULES

def debug_single_file(file_path, config):
    print(f"--- Menganalisis File: {file_path} ---")

    # 1. Muat dan proses data TANPA normalisasi
    raw_data = parse_jdx(file_path)
    if not raw_data:
        print("Gagal mem-parsing file.")
        return

    spectrum_df = preprocess_spectrum(raw_data, config, 'uv', normalize=False)
    
    # 2. Coba deteksi kromofor
    detected_groups = autogenerate_chromophores(spectrum_df)

    print("\n--- Hasil Deteksi ---")
    if detected_groups:
        print(f"Kromofor Terdeteksi: {', '.join(detected_groups)}")
    else:
        print("Tidak ada kromofor yang terdeteksi dengan aturan saat ini.")

    # 3. Tampilkan detail perbandingan untuk setiap aturan
    print("\n--- Detail Perbandingan Aturan ---")
    wavelengths = spectrum_df['wavelength'].values
    log_epsilons = spectrum_df['log_epsilon'].values
    
    for rule in UV_SPECTRAL_RULES:
        mask = (wavelengths >= rule['range'][0]) & (wavelengths <= rule['range'][1])
        max_val_in_range = np.max(log_epsilons[mask]) if np.any(mask) else -1
        
        status = "LOLOS" if max_val_in_range > rule['min_log_epsilon'] else "GAGAL"
        
        print(f"Aturan '{rule['group']}' (Rentang {rule['range'][0]}-{rule['range'][1]} nm):")
        print(f"  -> Nilai Maks Ditemukan: {max_val_in_range:.2f}")
        print(f"  -> Ambang Batas Aturan: {rule['min_log_epsilon']:.2f}")
        print(f"  -> Status: {status}")

    # 4. Tampilkan plot untuk verifikasi visual
    plt.figure(figsize=(12, 6))
    plt.plot(spectrum_df['wavelength'], spectrum_df['log_epsilon'])
    plt.title(f'Spektrum UV-Vis (Un-normalized) - {os.path.basename(file_path)}')
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Log(ε)")
    plt.grid(True, linestyle='--')
    
    for rule in UV_SPECTRAL_RULES:
        plt.axvspan(rule['range'][0], rule['range'][1], color='red', alpha=0.1)
        plt.axhline(y=rule['min_log_epsilon'], color='green', linestyle='--', 
                    xmin=(rule['range'][0]-200)/600, xmax=(rule['range'][1]-200)/600)

    print("\nMenampilkan plot... Tutup jendela plot untuk keluar.")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Debug Auto-Labeler UV-Vis")
    parser.add_argument("--file", required=True, help="Path ke satu file .jdx UV-Vis untuk dianalisis.")
    args = parser.parse_args()

    with open("configs/main_config.yaml", 'r') as f:
        config = yaml.safe_load(f)

    debug_single_file(args.file, config)