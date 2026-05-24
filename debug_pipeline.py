# src/debug_fuction/debug_pipeline.py (Versi Diperbaiki)

import yaml
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import sys

# --- Pengaturan Path untuk memastikan impor berfungsi ---
# Ini akan menambahkan direktori utama proyek ke path


# --- Impor fungsi-fungsi inti dari proyek ---
from src.data.data_processing import parse_jdx, preprocess_spectrum
from src.labeling.auto_labeler import SPECTRAL_RULES, UV_SPECTRAL_RULES
from scipy.signal import find_peaks

def debug_with_preprocess_function(file_path, config, spectrum_type):
    """
    Memvisualisasikan data mentah vs. data yang telah diproses
    menggunakan fungsi preprocess_spectrum.
    """
    print(f"--- Memulai Debug untuk: {os.path.basename(file_path)} ---")
    
    # --- TAHAP 1: PARSING DATA MENTAH ---
    raw_data = parse_jdx(file_path)
    if not raw_data:
        print("❌ GAGAL: Parsing file tidak menghasilkan data.")
        return
    print("✅ Tahap 1: Parsing data mentah berhasil.")
    
    # Simpan data mentah untuk perbandingan
    raw_df = pd.DataFrame({'x': raw_data['x'], 'y': raw_data['y']}).dropna()
    
    # --- TAHAP 2: JALANKAN FUNGSI PREPROCESS_SPECTRUM ---
    # Panggil dengan normalize=True agar data konsisten dengan aturan yang dibuat.
    processed_df = preprocess_spectrum(raw_data, config, spectrum_type, normalize=True)
    if processed_df is None:
        print(f"❌ GAGAL: Fungsi preprocess_spectrum tidak menghasilkan data.")
        return
    print(f"✅ Tahap 2: Pra-pemrosesan dengan 'preprocess_spectrum' berhasil.")
    
    # --- TAHAP 3: DETEKSI & PELABELAN PUNCAK (PADA DATA YANG SUDAH DINORMALISASI) ---
    print("✅ Tahap 3: Mendeteksi dan melabeli puncak...")

    labeled_peaks = []
    if spectrum_type == 'ir':
        x_vals = processed_df['wavenumber'].values
        y_vals = processed_df['absorbance'].values
        rules = SPECTRAL_RULES
    else: # uv
        x_vals = processed_df['wavelength'].values
        y_vals = processed_df['log_epsilon'].values
        rules = UV_SPECTRAL_RULES

    # --- PERBAIKAN: Gunakan parameter yang lebih ketat untuk mengurangi noise ---
    # Ini akan membuat visualisasi lebih bersih dan relevan.
    peaks, properties = find_peaks(y_vals, prominence=0.03, height=0.03)

    for i, peak_idx in enumerate(peaks):
        peak_location = x_vals[peak_idx]
        peak_height = y_vals[peak_idx]
        peak_prominence = properties['prominences'][i]

        for rule in rules:
            is_in_range = min(rule['range']) <= peak_location <= max(rule['range'])
            passes_prominence = peak_prominence >= rule.get('min_prominence', 0)
            passes_height = peak_height >= rule.get('min_height', 0)

            if is_in_range and passes_prominence and passes_height:
                labeled_peaks.append({"location": peak_location, "height": peak_height, "label": rule['group']})
                break # Lanjut ke puncak berikutnya setelah ditemukan kecocokan
    print(f"    -> Ditemukan {len(labeled_peaks)} puncak yang terlabel.")

    # --- TAHAP 4: VISUALISASI PERBANDINGAN ---
    plt.figure(figsize=(15, 7))
    
    # Tentukan label sumbu berdasarkan tipe spektrum
    if spectrum_type == 'ir':
        x_col_raw, y_col_raw = 'x', 'y'
        x_col_processed, y_col_processed = 'wavenumber', 'absorbance'
        plt.title("Perbandingan Spektrum IR Mentah vs. Hasil Pra-pemrosesan")
        plt.xlabel("Wavenumber (cm⁻¹)")
        plt.ylabel("Intensitas")
    else: # uv
        x_col_raw, y_col_raw = 'x', 'y'
        x_col_processed, y_col_processed = 'wavelength', 'log_epsilon'
        plt.title("Perbandingan Spektrum UV-Vis Mentah vs. Hasil Pra-pemrosesan")
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Intensitas")

    # Plot data mentah (mungkin dalam Transmittance)
    plt.plot(raw_df[x_col_raw], raw_df[y_col_raw], label=f'Data Mentah ({raw_data["metadata"].get("yunits", "N/A")})', alpha=0.6)
    
    # Plot data yang sudah diproses (dalam Absorbance/log_epsilon ternormalisasi)
    plt.plot(processed_df[x_col_processed], processed_df[y_col_processed], label='Data Setelah Pra-pemrosesan', linewidth=2)
    
    # Tambahkan anotasi untuk puncak yang terlabel
    for peak in labeled_peaks:
        plt.plot(peak['location'], peak['height'], 'x', color='red')
        plt.text(peak['location'], peak['height'] + 0.02, peak['label'], rotation=45, ha='left', fontsize=9, color='darkred')

    # Balik sumbu-x untuk IR agar sesuai konvensi
    if spectrum_type == 'ir':
        plt.gca().invert_xaxis()
        
    plt.grid(True, linestyle='--')
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Debug hasil akhir dari fungsi preprocess_spectrum.")
    parser.add_argument("--file", required=True, help="Path ke file JDX yang akan di-debug.")
    parser.add_argument("--type", required=True, choices=["ir", "uv"], help="Tipe spektrum.")
    args = parser.parse_args()
    
    # Muat konfigurasi
    config_path = os.path.join( 'main_config.yaml')
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"❌ GAGAL: File konfigurasi '{config_path}' tidak ditemukan.")
    else:
        debug_with_preprocess_function(args.file, config, args.type)


        #python debug_pipeline.py --file "data/raw/nist_jdx/563-54-2/IR_0_563-54-2.jdx" --type "ir"