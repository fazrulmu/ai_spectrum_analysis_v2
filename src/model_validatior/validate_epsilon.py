# validate_epsilon.py

import yaml
import glob
import numpy as np
import os
import re
from tqdm import tqdm
from scipy.signal import find_peaks

# Impor fungsi dan aturan yang kita perlukan
from src.data_processing import parse_jdx, preprocess_spectrum
from src.auto_labeler_uv import UV_SPECTRAL_RULES

def validate_height_across_dataset(config):
    """
    Menganalisis seluruh dataset UV-Vis untuk mengumpulkan statistik tinggi puncak (log ε)
    bagi setiap aturan kromofor.
    """
    paths = config['paths']

    # Menggunakan logika penemuan file yang sama dengan data_processing.py
    all_jdx = glob.glob(os.path.join(paths['raw_data_dir'], '**', '*.jdx'), recursive=True)
    all_files = [f for f in all_jdx if re.search(r'uv[\-\s]?vis', os.path.basename(f), re.IGNORECASE)]

    height_collections = {rule['group']: [] for rule in UV_SPECTRAL_RULES}

    print(f"Memindai {len(all_files)} file UV-Vis untuk validasi tinggi puncak (log ε)...")
    
    for file_path in tqdm(all_files):
        raw_data = parse_jdx(file_path)
        if not raw_data:
            continue

        spectrum_df = preprocess_spectrum(raw_data, config, 'uv', normalize=False)
        wavelengths = spectrum_df['wavelength'].values
        log_epsilons = spectrum_df['log_epsilon'].values

        all_peaks, properties = find_peaks(log_epsilons, height=0.01, width=1)
        
        for i, peak_idx in enumerate(all_peaks):
            wavelength_at_peak = wavelengths[peak_idx]
            height_at_peak = properties['peak_heights'][i]

            for rule in UV_SPECTRAL_RULES:
                if rule['range'][0] <= wavelength_at_peak <= rule['range'][1]:
                    height_collections[rule['group']].append(height_at_peak)
    
    print("\n--- Hasil Validasi Tinggi Puncak (log ε) ---")
    
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
        print(f"  💡 REKOMENDASI min_log_epsilon: {p25:.2f} (untuk menangkap ~75% puncak)")

if __name__ == "__main__":
    with open("main_config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    validate_height_across_dataset(config)