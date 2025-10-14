# src/data_analyzer/run_data_audit.py (Versi Baru dengan Opsi --source dan --limit)

import os
import yaml
import glob
import re
import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.signal import find_peaks, peak_widths
import argparse

# Impor relatif dari dalam paket 'src'
from src.data_processing import parse_jdx, preprocess_spectrum
from src.auto_labeler import SPECTRAL_RULES, get_elements_from_molform

def analyze_spectrum_dynamically(spectrum_df, metadata):
    """
    Menganalisis satu spektrum IR, memberikan label deskriptif dinamis,
    dan memvalidasi temuan menggunakan rumus molekul (MOLFORM).
    """
    # ... (Isi fungsi ini tidak perlu diubah, tetap sama seperti sebelumnya) ...
    wavenumbers = spectrum_df['wavenumber'].values
    absorbances = spectrum_df['absorbance'].values
    molform = metadata.get('molform', '')
    present_elements = get_elements_from_molform(molform)
    peaks, properties = find_peaks(absorbances, prominence=0.01, height=0.01)
    if len(peaks) < 1: return []
    try:
        widths_data = peak_widths(absorbances, peaks, rel_height=0.5)
        peak_width_map = {peak: width for peak, width in zip(peaks, widths_data[0])}
    except Exception:
        peak_width_map = {peak: 0 for peak in peaks}
    all_heights = properties.get('peak_heights', np.array([0]))
    all_widths = np.array(list(peak_width_map.values()))
    max_height = np.max(all_heights) if all_heights.size > 0 else 0
    median_width = np.median(all_widths) if all_widths.size > 0 else 0
    analyzed_peaks = []
    for i, peak_idx in enumerate(peaks):
        wavenumber = wavenumbers[peak_idx]
        height = properties['peak_heights'][i]
        prominence = properties['prominences'][i]
        width = peak_width_map.get(peak_idx, 0)
        for rule in SPECTRAL_RULES:
            is_math_match = (rule['range'][0] <= wavenumber <= rule['range'][1] and 
                             prominence >= rule.get('min_prominence', 0.01))
            if is_math_match:
                passes_molform_check = True
                if present_elements and 'required_atoms' in rule:
                    if not rule['required_atoms'].issubset(present_elements):
                        passes_molform_check = False
                if passes_molform_check:
                    if height >= 0.65 * max_height: intensity_label = "strong"
                    elif height >= 0.25 * max_height: intensity_label = "medium"
                    else: intensity_label = "weak"
                    shape_label = "broad" if width > 2.5 * median_width and median_width > 0 else "sharp"
                    descriptive_label = f"{rule['group']} ({intensity_label}, {shape_label})"
                    analyzed_peaks.append({
                        "assigned_label": rule['group'], "descriptive_label": descriptive_label,
                        "peak_location": wavenumber, "peak_prominence": prominence,
                        "peak_height": height, "peak_width": width
                    })
                    break 
    return analyzed_peaks

def audit_dataset(config, output_dir, source_dir=None, limit=None):
    """
    Menjalankan audit komprehensif pada dataset JDX, dengan opsi untuk membatasi sumber.
    """
    all_analyzed_peaks = []

    # --- PERUBAHAN UTAMA DIMULAI DI SINI ---
    if source_dir:
        # Jika direktori sumber spesifik diberikan, cari file di sana
        print(f"Mencari file JDX di direktori yang dipilih: {source_dir}")
        search_path = os.path.join(source_dir, '**', '*.jdx')
    else:
        # Jika tidak, gunakan path dari file konfigurasi
        print(f"Mencari file JDX di direktori default dari config: {config['paths']['raw_data_dir']}")
        search_path = os.path.join(config['paths']['raw_data_dir'], '**', '*.jdx')

    all_jdx_files = glob.glob(search_path, recursive=True)
    ir_files = [f for f in all_jdx_files if re.search(r'(ftir|ft-ir|ir)', os.path.basename(f).lower())]

    # Terapkan batasan jumlah file jika diberikan
    if limit:
        print(f"Ditemukan {len(ir_files)} file, akan diproses {limit} file pertama.")
        ir_files = ir_files[:limit]
    else:
        print(f"Ditemukan {len(ir_files)} file untuk diaudit.")
    # --- AKHIR PERUBAHAN ---

    for file_path in tqdm(ir_files, desc="Auditing IR Spectra"):
        try:
            raw_data = parse_jdx(file_path)
            if not raw_data: continue

            processed_df = preprocess_spectrum(raw_data, config, 'ir', normalize=True)
            if processed_df is None: continue

            analyzed_peaks = analyze_spectrum_dynamically(processed_df, raw_data.get('metadata', {}))
            
            for peak_info in analyzed_peaks:
                peak_info['file_path'] = os.path.basename(file_path)
                peak_info['molform'] = raw_data.get('metadata', {}).get('molform', 'N/A')
                all_analyzed_peaks.append(peak_info)
                
        except Exception as e:
            continue
            
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if all_analyzed_peaks:
        df_audit = pd.DataFrame(all_analyzed_peaks)
        output_path = os.path.join(output_dir, f"dynamic_ir_audit_results_{timestamp}.csv")
        df_audit.to_csv(output_path, index=False)
        print(f"\n✅ Audit selesai. Data disimpan di: {output_path}")
    else:
        print("\nTidak ada puncak valid yang ditemukan selama audit.")

def main(config_path, output_dir, source=None, limit=None):
    """Fungsi utama untuk menjalankan modul audit."""
    print("--- 🔬 Memulai Modul Audit Kualitas Data (Versi Canggih) ---")
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"❌ Error: File konfigurasi '{config_path}' tidak ditemukan.")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    # Teruskan argumen baru ke fungsi audit_dataset
    audit_dataset(config, output_dir, source_dir=source, limit=limit)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Jalankan audit kualitas dataset spektral dengan analisis dinamis.")
    parser.add_argument('--config', default='main_config.yaml', help='Path ke file konfigurasi.')
    parser.add_argument('--output', default='reports/data_audit', help='Direktori untuk menyimpan hasil audit.')
    # --- Tambahkan argumen baru di sini ---
    parser.add_argument('--source', default=None, help='(Opsional) Path ke direktori spesifik yang berisi file JDX.')
    parser.add_argument('--limit', type=int, default=None, help='(Opsional) Batasi jumlah file yang akan diproses.')
    
    args = parser.parse_args()
    main(config_path=args.config, output_dir=args.output, source=args.source, limit=args.limit)