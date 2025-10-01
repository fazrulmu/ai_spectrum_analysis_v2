# explore_spectral_rules.py

import os
import yaml
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from scipy.signal import find_peaks

# Impor fungsi dari proyek Anda
from src.data_processing import parse_jdx, preprocess_spectrum
from src.auto_labeler import (
    SPECTRAL_RULES, autogenerate_functional_groups,
    UV_SPECTRAL_RULES, autogenerate_chromophores
)

def build_exploration_database(config):
    """Membangun database dari semua spektrum untuk analisis aturan."""
    paths = config['paths']
    compound_data = defaultdict(lambda: {
        'ir_labels': set(), 'uv_labels': set(),
        'ir_peaks': [], 'uv_peaks': []
    })

    # 1. Proses semua data IR
    print("Menganalisis data IR...")
    ir_files = glob.glob(os.path.join(paths['raw_data_dir'], '**', '*ir*.jdx'), recursive=True)
    for file_path in tqdm(ir_files):
        cas_no = os.path.basename(os.path.dirname(file_path))
        raw_data = parse_jdx(file_path)
        if not raw_data: continue
        processed_df = preprocess_spectrum(raw_data, config, 'ir', True)
        if processed_df is None or processed_df.empty: continue
        
        labels = autogenerate_functional_groups(processed_df)
        compound_data[cas_no]['ir_labels'].update(labels)
        
        peaks, props = find_peaks(processed_df['absorbance'].values, prominence=0.02)
        for p_idx in peaks:
            compound_data[cas_no]['ir_peaks'].append(processed_df['wavenumber'].iloc[p_idx])

    # 2. Proses semua data UV
    print("\nMenganalisis data UV...")
    uv_files = glob.glob(os.path.join(paths['raw_data_dir'], '**', '*uv*.jdx'), recursive=True)
    for file_path in tqdm(uv_files):
        cas_no = os.path.basename(os.path.dirname(file_path))
        raw_data = parse_jdx(file_path)
        if not raw_data: continue
        processed_df = preprocess_spectrum(raw_data, config, 'uv', True)
        if processed_df is None or processed_df.empty: continue
            
        labels = autogenerate_chromophores(processed_df)
        compound_data[cas_no]['uv_labels'].update(labels)
        
        peaks, props = find_peaks(processed_df['log_epsilon'].values, prominence=0.01)
        for p_idx in peaks:
            compound_data[cas_no]['uv_peaks'].append(processed_df['wavelength'].iloc[p_idx])

    # 3. Konversi ke DataFrame
    df = pd.DataFrame.from_dict(compound_data, orient='index')
    df.index.name = 'cas'
    df = df.reset_index()
    # Ubah set menjadi list agar lebih mudah dibaca
    df['ir_labels'] = df['ir_labels'].apply(list)
    df['uv_labels'] = df['uv_labels'].apply(list)
    return df

def run_analysis(df):
    """Menjalankan beberapa contoh analisis pada database yang sudah dibuat."""
    
    print("\n\n" + "="*50)
    print("ANALISIS EKSPLORATIF UNTUK ATURAN SPEKTRAL")
    print("="*50)

    # --- Analisis 1: Mencari Aturan Lintas Spektrum (IR -> UV) ---
    print("\n\n[Analisis 1] Hubungan antara 'ketone_co' (IR) dengan label UV")
    print("------------------------------------------------------------------")
    ketone_df = df[df['ir_labels'].apply(lambda x: 'ketone_co' in x)]
    if not ketone_df.empty:
        # Hitung kemunculan setiap label UV pada senyawa yang punya 'ketone_co'
        uv_labels_with_ketone = ketone_df['uv_labels'].explode().dropna()
        print("Label UV yang paling sering muncul bersama 'ketone_co':")
        print(uv_labels_with_ketone.value_counts().head(5))
        print("\n>>> Insight: Jika Anda melihat 'carbonyl_n_pi_star' atau 'enone_pi_pi_star' di UV,")
        print("    kemungkinan besar ada gugus C=O di IR. Ini adalah aturan konfirmasi.")
    else:
        print("Tidak ditemukan senyawa dengan label 'ketone_co' untuk dianalisis.")

    # --- Analisis 2: Mencari Aturan Pola/Kombinasi (dalam IR) ---
    print("\n\n[Analisis 2] Konfirmasi label 'ester' dari dua puncak IR")
    print("---------------------------------------------------------")
    # Cari senyawa yang punya puncak C-O ester
    ester_c_o_c_df = df[df['ir_labels'].apply(lambda x: 'ester_co_c' in x)]
    if not ester_c_o_c_df.empty:
        count_with_carbonyl = 0
        # Untuk setiap senyawa tsb, periksa apakah juga punya puncak C=O ester
        for _, row in ester_c_o_c_df.iterrows():
            if 'ester_co' in row['ir_labels']:
                count_with_carbonyl += 1
        
        percentage = (count_with_carbonyl / len(ester_c_o_c_df)) * 100
        print(f"Dari {len(ester_c_o_c_df)} senyawa yang memiliki 'ester_co_c' (puncak C-O),")
        print(f"{count_with_carbonyl} ({percentage:.1f}%) di antaranya juga memiliki 'ester_co' (puncak C=O).")
        print("\n>>> Insight: Ini adalah bukti kuat untuk aturan kombinasi.")
        print("    Label 'ester' sebaiknya hanya diberikan jika kedua puncak ini ada.")
    else:
        print("Tidak ditemukan senyawa dengan label 'ester_co_c' untuk dianalisis.")

    # --- Analisis 3: Mencari Aturan Ketiadaan Puncak ---
    print("\n\n[Analisis 3] Validasi 'carboxylic_acid_co' dengan puncah O-H")
    print("-------------------------------------------------------------------")
    acid_co_df = df[df['ir_labels'].apply(lambda x: 'carboxylic_acid_co' in x)]
    if not acid_co_df.empty:
        count_missing_oh = 0
        for _, row in acid_co_df.iterrows():
            if 'carboxylic_acid_oh_broad' not in row['ir_labels']:
                count_missing_oh += 1
        
        percentage = (count_missing_oh / len(acid_co_df)) * 100
        print(f"Dari {len(acid_co_df)} senyawa yang dilabeli 'carboxylic_acid_co',")
        print(f"{count_missing_oh} ({percentage:.1f}%) TIDAK memiliki label 'carboxylic_acid_oh_broad'.")
        print("\n>>> Insight: Sampel-sampel ini kemungkinan adalah false positive.")
        print("    Ini adalah bukti untuk aturan ketiadaan: 'carboxylic_acid_co' tidak valid TANPA 'carboxylic_acid_oh_broad'.")
    else:
        print("Tidak ditemukan senyawa dengan label 'carboxylic_acid_co' untuk dianalisis.")


if __name__ == "__main__":
    with open('main_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
        
    master_df = build_exploration_database(config)
    
    print(f"\nDatabase eksplorasi berhasil dibuat dengan {len(master_df)} senyawa unik.")
    # Simpan ke CSV untuk analisis manual jika diinginkan
    master_df.to_csv("exploration_database.csv", index=False)
    print("Database disimpan ke exploration_database.csv")
    
    run_analysis(master_df)