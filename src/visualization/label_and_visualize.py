# src/label_and_visualize.py

import os
import matplotlib.pyplot as plt

# Mengimpor fungsi-fungsi yang sudah ada
from src.data_processing import parse_jdx, preprocess_spectrum
from src.auto_labeler import SPECTRAL_RULES
from scipy.signal import find_peaks

def label_spectrum_peaks(spectrum_df):
    """
    Fungsi untuk mendeteksi dan melabeli puncak dalam satu spektrum.
    Mengembalikan daftar puncak yang ditemukan beserta labelnya.
    """
    wavenumbers = spectrum_df['wavenumber'].values
    absorbances = spectrum_df['absorbance'].values
    
    # Deteksi puncak
    # Puncak absorbans menunjuk ke atas, jadi kita cari puncak biasa
    # --- PERBAIKAN: Naikkan prominence dan height untuk mengurangi deteksi noise ---
    peaks, properties = find_peaks(absorbances, prominence=0.03, height=0.03)
    
    labeled_peaks = []
    if len(peaks) == 0:
        return labeled_peaks

    # Cocokkan setiap puncak dengan aturan
    for i, peak_idx in enumerate(peaks):
        wavenumber = wavenumbers[peak_idx]
        prominence = properties['prominences'][i]
        
        for rule in SPECTRAL_RULES:
            if rule['range'][0] <= wavenumber <= rule['range'][1]:
                if prominence >= rule['min_prominence']:
                    labeled_peaks.append({
                        "wavenumber": wavenumber,
                        "absorbance": absorbances[peak_idx],
                        "label": rule['group']
                    })
                    # Hentikan pencarian setelah menemukan kecocokan pertama untuk puncak ini
                    break 
    return labeled_peaks

def visualize_labeled_spectrum(spectrum_df, labeled_peaks, output_path):
    """
    Membuat plot spektrum dan menambahkan anotasi untuk puncak yang terlabel.
    """
    plt.figure(figsize=(15, 7))
    plt.plot(spectrum_df['wavenumber'], spectrum_df['absorbance'], label="Spektrum")
    
    # Tambahkan tanda dan label untuk setiap puncak yang terdeteksi
    for peak in labeled_peaks:
        plt.plot(peak['wavenumber'], peak['absorbance'], 'x', color='red')
        plt.text(peak['wavenumber'], peak['absorbance'] + 0.02, peak['label'], 
                rotation=45, ha='left', fontsize=9, color='darkred')

    plt.title("Spektrum IR dengan Anotasi Gugus Fungsi")
    plt.xlabel("Wavenumber (cm⁻¹)")
    plt.ylabel("Absorbance (Normalized)")
    plt.gca().invert_xaxis()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    
    # Simpan plot
    plt.savefig(output_path)
    plt.close()
    print(f"✅ Plot dengan label disimpan di: {output_path}")

def main(output_dir, config, target_file):
    """
    Fungsi utama untuk memuat, melabeli, dan memvisualisasikan satu spektrum.
    Dirancang untuk dipanggil oleh main.py pusat komando.
    """
    if not os.path.exists(target_file):
        print(f"❌ Error: File target '{target_file}' tidak ditemukan.")
        return

    print(f"--- 🔬 Menganalisis dan melabeli file: {target_file} ---")

    # 1. Pra-pemrosesan
    raw_data = parse_jdx(target_file)
    if not raw_data:
        print("❌ Gagal mem-parsing file JDX.")
        return
        
    processed_df = preprocess_spectrum(raw_data, config, 'ir', normalize=True)
    if processed_df is None or processed_df.empty:
        print("❌ Gagal memproses spektrum.")
        return

    # 2. Pelabelan Puncak
    labeled_peaks = label_spectrum_peaks(processed_df)
    print(f"🔍 Ditemukan {len(labeled_peaks)} puncak yang terlabel:")
    for peak in labeled_peaks:
        print(f"  - Gugus: {peak['label']:<25} | Posisi: {peak['wavenumber']:.2f} cm⁻¹")

    # 3. Visualisasi
    output_filename = os.path.basename(target_file).replace('.jdx', '_labeled.png')
    output_path = os.path.join(output_dir, output_filename)
    visualize_labeled_spectrum(processed_df, labeled_peaks, output_path)



    #python main.py visualize --file "results/data_farmer/smart_downloader/554-84-7/554-84-7_UV-Vis_0.jdx"