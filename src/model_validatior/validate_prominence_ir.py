# validate_prominence_ir.py

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

def validate_prominence_across_dataset_ir(config):
    """
    Menganalisis seluruh dataset IR untuk mengumpulkan statistik prominence
    bagi setiap aturan gugus fungsi.
    """
    paths = config['paths']
    
    # Menggunakan logika penemuan file IR yang sama dengan data_processing.py
    all_jdx = glob.glob(os.path.join(paths['raw_data_dir'], '**', '*.jdx'), recursive=True)
    all_files = [f for f in all_jdx if re.search(r'(ftir|ft-ir|ir)', os.path.basename(f), re.IGNORECASE)]

    prominence_collections = {rule['group']: [] for rule in SPECTRAL_RULES}

    print(f"Memindai {len(all_files)} file IR untuk validasi prominence...")
    
    for file_path in tqdm(all_files):
        raw_data = parse_jdx(file_path)
        if not raw_data:
            continue

        # Proses spektrum DENGAN normalisasi, karena aturan IR kita dirancang untuk itu
        spectrum_df = preprocess_spectrum(raw_data, config, 'ir', normalize=True)
        wavenumbers = spectrum_df['wavenumber'].values
        absorbances = spectrum_df['absorbance'].values

        # Temukan semua puncak di spektrum ini
        all_peaks, properties = find_peaks(absorbances, prominence=0.02, width=1)
        
        # Kelompokkan prominence dari setiap puncak ke dalam aturan yang sesuai
        for i, peak_idx in enumerate(all_peaks):
            wavenumber_at_peak = wavenumbers[peak_idx]
            prominence_at_peak = properties['prominences'][i]

            for rule in SPECTRAL_RULES:
                if rule['range'][0] <= wavenumber_at_peak <= rule['range'][1]:
                    prominence_collections[rule['group']].append(prominence_at_peak)
    
    print("\n--- Hasil Validasi Prominence (IR) ---")
    
    # Hitung dan tampilkan statistik untuk setiap grup
    for group, prominences in prominence_collections.items():
        print(f"\n## Statistik untuk Grup: '{group}'")
        if not prominences:
            print("  -> Tidak ada puncak yang ditemukan di rentang ini.")
            continue
            
        prom_array = np.array(prominences)
        
        count = len(prom_array)
        p25, p50, p75 = np.percentile(prom_array, [25, 50, 75])
        
        print(f"  -> Jumlah Puncak Ditemukan: {count}")
        print(f"  -> Rata-rata Prominence   : {np.mean(prom_array):.2f}")
        print(f"  -> Median Prominence      : {p50:.2f}")
        print(f"  -> Min / Maks             : {np.min(prom_array):.2f} / {np.max(prom_array):.2f}")
        print(f"  -> Kuartil Bawah (25%)    : {p25:.2f}")
        print(f"  --------------------------------------------------")
        print(f"  💡 REKOMENDASI min_prominence: {p25:.2f} (untuk menangkap ~75% puncak)")

if __name__ == "__main__":
    with open("main_config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    validate_prominence_across_dataset_ir(config)