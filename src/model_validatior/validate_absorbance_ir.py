# validate_absorbance_ir.py

import yaml
import glob
import numpy as np
import os
import re
from tqdm import tqdm
from scipy.signal import find_peaks

# Impor fungsi dan ATURAN IR yang kita perlukan
from src.data_processing import parse_jdx, preprocess_spectrum
from src.auto_labeler import SPECTRAL_RULES

def validate_height_across_dataset_ir(config):
    """
    Menganalisis seluruh dataset IR untuk mengumpulkan statistik tinggi puncak (Absorbance)
    bagi setiap aturan gugus fungsi.
    """
    paths = config['paths']
    
    # Menggunakan logika penemuan file IR yang sama dengan data_processing.py
    all_jdx = glob.glob(os.path.join(paths['raw_data_dir'], '**', '*.jdx'), recursive=True)
    all_files = [f for f in all_jdx if re.search(r'(ftir|ft-ir|ir)', os.path.basename(f), re.IGNORECASE)]

    height_collections = {rule['group']: [] for rule in SPECTRAL_RULES}

    print(f"Memindai {len(all_files)} file IR untuk validasi tinggi puncak (Absorbance)...")
    
    for file_path in tqdm(all_files):
        raw_data = parse_jdx(file_path)
        if not raw_data:
            continue

        # Proses spektrum DENGAN normalisasi, karena tinggi puncak di sini bersifat relatif
        spectrum_df = preprocess_spectrum(raw_data, config, 'ir', normalize=True)
        wavenumbers = spectrum_df['wavenumber'].values
        absorbances = spectrum_df['absorbance'].values

        # Temukan semua puncak di spektrum ini
        # Gunakan height dasar yang sangat rendah untuk menangkap hampir semuanya
        all_peaks, properties = find_peaks(absorbances, height=0.01, width=1)
        
        # Kelompokkan tinggi dari setiap puncak ke dalam aturan yang sesuai
        for i, peak_idx in enumerate(all_peaks):
            wavenumber_at_peak = wavenumbers[peak_idx]
            height_at_peak = properties['peak_heights'][i]

            for rule in SPECTRAL_RULES:
                if rule['range'][0] <= wavenumber_at_peak <= rule['range'][1]:
                    height_collections[rule['group']].append(height_at_peak)
    
    print("\n--- Hasil Validasi Tinggi Puncak (Absorbance Ternormalisasi) ---")
    
    # Hitung dan tampilkan statistik untuk setiap grup
    for group, heights in height_collections.items():
        print(f"\n## Statistik untuk Grup: '{group}'")
        if not heights:
            print("  -> Tidak ada puncak yang ditemukan di rentang ini.")
            continue
            
        height_array = np.array(heights)
        
        count = len(height_array)
        p25, p50, p75 = np.percentile(height_array, [25, 50, 75])
        
        print(f"  -> Jumlah Puncak Ditemukan: {count}")
        print(f"  -> Rata-rata Tinggi Puncak: {np.mean(height_array):.2f}")
        print(f"  -> Median Tinggi Puncak   : {p50:.2f}")
        print(f"  -> Min / Maks             : {np.min(height_array):.2f} / {np.max(height_array):.2f}")
        print(f"  -> Kuartil Bawah (25%)    : {p25:.2f}")
        print(f"  --------------------------------------------------")
        print(f"  💡 REKOMENDASI min_height: {p25:.2f} (untuk menangkap ~75% puncak)")

if __name__ == "__main__":
    with open("main_config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    validate_height_across_dataset_ir(config)