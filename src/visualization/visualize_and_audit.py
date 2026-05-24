# visualize_and_audit.py (Versi Diperbaiki)

import os
import yaml
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import glob

# Impor fungsi-fungsi dari proyek Anda
# Pastikan path ini benar jika Anda memindahkan file
from src.data_processing import parse_jdx, preprocess_spectrum

def plot_audited_spectrum(spectrum_df, peak_data, output_path):
    """
    Membuat plot spektrum IR asli dan menambahkan anotasi dari data audit.
    """
    plt.figure(figsize=(18, 8))
    
    plt.plot(spectrum_df['wavenumber'], spectrum_df['absorbance'], label="Spektrum Asli", color='blue', alpha=0.8)
    
    for _, peak in peak_data.iterrows():
        wavenumber = peak['peak_location']
        height = peak['peak_height']
        label = peak['descriptive_label']
        
        plt.plot(wavenumber, height, 'x', color='red', markersize=8, markeredgewidth=2)
        plt.text(wavenumber, height + 0.03, label, 
                rotation=90, ha='center', va='bottom', fontsize=9, color='darkred',
                bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.5))

    # Membersihkan nama file untuk judul plot
    file_name_raw = os.path.basename(output_path).replace('_audit_plot.png', '')
    cleaned_file_name = file_name_raw.replace('$', r'\$').replace('_', ' ')
    
    plt.title(f"Hasil Audit Visual untuk Spektrum: {cleaned_file_name}", fontsize=16)
    plt.xlabel("Wavenumber (cm⁻¹)", fontsize=12)
    plt.ylabel("Absorbance (Normalized)", fontsize=12)
    plt.gca().invert_xaxis()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.ylim(bottom=-0.05)
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(output_path)
    plt.close()

def main(config_path, audit_file_path, output_dir):
    """
    Fungsi utama untuk memuat hasil audit dan membuat visualisasi untuk setiap spektrum.
    """
    print("--- 📊 Memulai Visualisasi Hasil Audit ---")
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"❌ Error: File konfigurasi '{config_path}' tidak ditemukan.")
        return

    try:
        audit_df = pd.read_csv(audit_file_path)
    except FileNotFoundError:
        print(f"❌ Error: File audit '{audit_file_path}' tidak ditemukan.")
        return

    raw_data_dir = config['paths']['raw_data_dir']
    unique_files = audit_df['file_path'].unique()
    
    print(f"Ditemukan {len(unique_files)} spektrum unik untuk divisualisasikan.")

    # Loop utama untuk memproses setiap file
    for file_name in tqdm(unique_files, desc="Membuat Plot Audit"):
        
        # --- Semua logika yang berhubungan dengan satu file harus ada di dalam loop ini ---
        
        original_file_path = None
        # Cari file di direktori utama dan subdirektori data mentah
        for root, dirs, files in os.walk(raw_data_dir):
            if file_name in files:
                original_file_path = os.path.join(root, file_name)
                break
        
        if not original_file_path:
            print(f"⚠️ Peringatan: File spektrum asli '{file_name}' tidak ditemukan. Melewati.")
            continue

        peak_data_for_file = audit_df[audit_df['file_path'] == file_name]

        raw_data = parse_jdx(original_file_path)
        if not raw_data: continue
        
        processed_df = preprocess_spectrum(raw_data, config, 'ir', normalize=True)
        if processed_df is None: continue

        # Definisikan output_path di sini, di dalam loop
        output_plot_path = os.path.join(output_dir, file_name.replace('.jdx', '_audit_plot.png'))
        
        # Panggil fungsi plot
        plot_audited_spectrum(processed_df, peak_data_for_file, output_plot_path)
        
    print(f"\n✅ Visualisasi selesai. Semua gambar disimpan di: {output_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Visualisasikan hasil dari skrip audit data.")
    parser.add_argument('--config', default='main_config.yaml', help='Path ke file konfigurasi.')
    parser.add_argument('--audit_file', required=True, help='Path ke file .csv hasil audit.')
    parser.add_argument('--output', default='reports/audit_visualizations', help='Direktori untuk menyimpan gambar plot.')
    
    args = parser.parse_args()
    
    # Pastikan 'src' ada di path jika menjalankan skrip ini secara langsung
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

    os.makedirs(args.output, exist_ok=True)
    main(config_path=args.config, audit_file_path=args.audit_file, output_dir=args.output)