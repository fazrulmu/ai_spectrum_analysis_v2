
# src/data_analyzer/generate_rule_statistics.py

import os
import glob
import pandas as pd
import numpy as np
import yaml
from tqdm import tqdm
from scipy.signal import find_peaks, peak_widths
from scipy.stats import skew, kurtosis
from scipy.integrate import simps
from itertools import combinations
import collections
import re

# Mengimpor fungsi dan aturan dari file lain
from data_processing import parse_jdx, preprocess_spectrum
from auto_labeler import SPECTRAL_RULES, UV_SPECTRAL_RULES

def analyze_ir_spectrum_features(spectrum_df):
    """
    Fungsi untuk mengekstrak fitur puncak dari spektrum IR (pendekatan lama).
    """
    wavenumbers = spectrum_df['wavenumber'].values
    absorbances = spectrum_df['absorbance'].values
    
    peaks, properties = find_peaks(absorbances, prominence=0.02, height=0.02)
    
    if len(peaks) == 0:
        return []

    widths_data = peak_widths(absorbances, peaks, rel_height=0.5)
    peak_width_map = {peak: width for peak, width in zip(peaks, widths_data[0])}

    found_peaks = []
    for i, peak_idx in enumerate(peaks):
        wavenumber = wavenumbers[peak_idx]
        prominence = properties['prominences'][i]
        height = properties['peak_heights'][i]
        width = peak_width_map.get(peak_idx, 0)
        
        matched_group = "unknown"
        for rule in SPECTRAL_RULES:
            if rule['range'][0] <= wavenumber <= rule['range'][1] and prominence >= rule['min_prominence']:
                matched_group = rule['group']
                break
        
        if matched_group != "unknown":
            found_peaks.append({
                "group": matched_group,
                "position": wavenumber,
                "prominence": prominence,
                "height": height,
                "width": width
            })
    return found_peaks

def analyze_uv_holistic_features(spectrum_df):
    """
    Fungsi BARU untuk menganalisis bentuk spektrum UV-Vis secara holistik.
    """
    wavelengths = spectrum_df['wavelength'].values
    log_epsilons = spectrum_df['log_epsilon'].values

    if len(log_epsilons) < 5:
        return None

    # 1. Fitur Puncak Utama (Main Peak)
    if np.any(log_epsilons):
        main_peak_idx = np.argmax(log_epsilons)
        lambda_max = wavelengths[main_peak_idx]
        intensity_at_lambda_max = log_epsilons[main_peak_idx]
        
        # Hitung FWHM untuk puncak utama
        try:
            widths_data = peak_widths(log_epsilons, [main_peak_idx], rel_height=0.5)
            fwhm = widths_data[0][0] if len(widths_data[0]) > 0 else 0
        except:
            fwhm = 0
    else:
        lambda_max, intensity_at_lambda_max, fwhm = 0, 0, 0

    # 2. Fitur Bentuk Holistik
    auc = simps(log_epsilons, dx=(wavelengths[1]-wavelengths[0])) if len(wavelengths) > 1 else 0
    skewness = skew(log_epsilons)
    kurtosis_val = kurtosis(log_epsilons)

    # 3. Fitur dari Spektrum Turunan Kedua
    # Turunan pertama untuk gradien, turunan kedua untuk kelengkungan (menemukan puncak tersembunyi)
    first_derivative = np.gradient(log_epsilons)
    second_derivative = np.gradient(first_derivative)
    
    # Puncak pada spektrum asli adalah lembah (nilai negatif besar) pada turunan kedua
    deriv_peaks, _ = find_peaks(-second_derivative, prominence=0.001)
    num_hidden_peaks = len(deriv_peaks)
    
    return {
        'lambda_max': lambda_max,
        'intensity_at_lambda_max': intensity_at_lambda_max,
        'fwhm': fwhm,
        'auc': auc,
        'skewness': skewness,
        'kurtosis': kurtosis_val,
        'num_hidden_peaks_deriv2': num_hidden_peaks
    }

def calculate_ir_feature_statistics(all_spectra_features, ratio_pairs):
    """
    Menghitung statistik agregat untuk fitur IR.
    """
    # 1. Statistik Dasar
    group_stats = collections.defaultdict(lambda: collections.defaultdict(list))
    for spectrum_features in all_spectra_features:
        for peak in spectrum_features:
            group = peak['group']
            group_stats[group]['position'].append(peak['position'])
            group_stats[group]['prominence'].append(peak['prominence'])
            group_stats[group]['width'].append(peak['width'])

    summary = {}
    for group, data in group_stats.items():
        summary[group] = {
            "count": len(data['position']),
            "mean_position": np.mean(data['position']),
            "std_position": np.std(data['position']),
            "mean_prominence": np.mean(data['prominence']),
            "std_prominence": np.std(data['prominence']),
            "mean_width": np.mean(data['width']),
            "std_width": np.std(data['width'])
        }
    basic_stats_df = pd.DataFrame.from_dict(summary, orient='index')

    # 2. Statistik Ko-okurensi
    co_occurrence = collections.Counter()
    for spectrum_features in all_spectra_features:
        present_groups = sorted(list(set([peak['group'] for peak in spectrum_features])))
        for pair in combinations(present_groups, 2):
            co_occurrence[pair] += 1
    
    co_occurrence_df = pd.DataFrame(co_occurrence.items(), columns=['group_pair', 'count']).sort_values(by='count', ascending=False)

    # 3. Statistik Rasio Puncak
    ratio_stats = collections.defaultdict(list)
    for spectrum_features in all_spectra_features:
        peaks_by_group = {p['group']: p for p in spectrum_features}
        for group1, group2 in ratio_pairs:
            if group1 in peaks_by_group and group2 in peaks_by_group:
                prom1 = peaks_by_group[group1]['prominence']
                prom2 = peaks_by_group[group2]['prominence']
                if prom2 > 1e-6:
                    ratio = prom1 / prom2
                    ratio_name = f"{group1}_vs_{group2}"
                    ratio_stats[ratio_name].append(ratio)

    ratio_summary = {}
    for name, values in ratio_stats.items():
        ratio_summary[name] = {
            "count": len(values),
            "mean_ratio": np.mean(values),
            "std_ratio": np.std(values),
            "min_ratio": np.min(values),
            "max_ratio": np.max(values)
        }
    ratio_stats_df = pd.DataFrame.from_dict(ratio_summary, orient='index')

    return basic_stats_df, co_occurrence_df, ratio_stats_df

def main(output_dir=".", config_path="main_config.yaml"):
    """
    Fungsi utama untuk memindai semua data, mengekstrak fitur IR & UV, 
    dan menghasilkan statistik dalam file CSV.
    """
    print("--- 📊 Memulai Analisis Statistik Aturan Spektral (IR & UV-Vis) ---")

    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"❌ Error: File konfigurasi '{config_path}' tidak ditemukan.")
        return

    raw_data_dir = config['paths']['raw_data_dir']
    all_jdx_files = glob.glob(os.path.join(raw_data_dir, '**', '*.jdx'), recursive=True)

    ir_files = [f for f in all_jdx_files if re.search(r'(ftir|ft-ir|ir)', os.path.basename(f), re.IGNORECASE)]
    uv_files = [f for f in all_jdx_files if re.search(r'uv[\-\s]?vis', os.path.basename(f), re.IGNORECASE)]

    # --- Analisis IR (Pendekatan lama yang sudah baik) ---
    if ir_files:
        print(f"\n🔬 Menganalisis {len(ir_files)} file spektrum IR...")
        all_ir_features = []
        for file_path in tqdm(ir_files, desc="Memproses IR"):
            raw_data = parse_jdx(file_path)
            if not raw_data: continue
            processed_df = preprocess_spectrum(raw_data, config, 'ir', normalize=True)
            if processed_df is None or processed_df.empty: continue
            features = analyze_ir_spectrum_features(processed_df)
            if features: all_ir_features.append(features)
            ir_ratio_pairs = [
            # Gugus Karbonil vs. Kerangka
            ('ketone_co', 'alkane_ch_sp3'),
            ('aldehyde_co', 'alkane_ch_sp3'),
            ('ester_co', 'alkane_ch_sp3'),
            ('amide_co', 'alkane_ch_sp3'),
            # Gugus dengan O-H atau N-H
            ('carboxylic_acid_co', 'carboxylic_acid_oh_broad'),
            ('alcohol_phenol_oh', 'ether_co'),
            ('amide_co', 'amide_nh'),
            # Gugus Ester
            ('ester_co', 'ether_co'),
            # Gugus Aromatik
            ('aromatic_ch', 'aromatic_cc'),
            ('aromatic_cc', 'aromatic_out_of_plane_ch'),
            # Kejenuhan (Saturated vs Unsaturated)
            ('alkene_ch_sp2', 'alkane_ch_sp3'),
            ('alkyne_ch_sp', 'alkane_ch_sp3'),
            # Gugus Nitro
            ('nitro_no2_asym', 'nitro_no2_sym'),
            # Gugus Amine
            ('amine_nh_primary', 'amine_cn'),
            # Halida
            ('alkyl_halide_ccl', 'alkane_ch_sp3')
        ]
        
        ir_basic, ir_cooc, ir_ratio = calculate_ir_feature_statistics(all_ir_features, ir_ratio_pairs)
        ir_basic.to_csv(os.path.join(output_dir, "ir_basic_statistics.csv"))
        ir_cooc.to_csv(os.path.join(output_dir, "ir_co_occurrence_statistics.csv"), index=False)
        ir_ratio.to_csv(os.path.join(output_dir, "ir_peak_ratio_statistics.csv"))
        print(f"✅ Statistik IR disimpan di: {output_dir}")

    # --- Analisis UV-Vis (Pendekatan HOLISTIK BARU) ---
    if uv_files:
        print(f"\n🔬 Menganalisis {len(uv_files)} file spektrum UV-Vis dengan pendekatan holistik...")
        all_uv_holistic_features = []
        for file_path in tqdm(uv_files, desc="Memproses UV-Vis"):
            raw_data = parse_jdx(file_path)
            if not raw_data: continue
            processed_df = preprocess_spectrum(raw_data, config, 'uv', normalize=False)
            if processed_df is None or processed_df.empty: continue
            holistic_features = analyze_uv_holistic_features(processed_df)
            if holistic_features:
                holistic_features['file_path'] = os.path.basename(file_path) # Tambahkan nama file untuk referensi
                all_uv_holistic_features.append(holistic_features)
        
        # Buat DataFrame dari semua fitur holistik yang diekstrak
        uv_holistic_df = pd.DataFrame(all_uv_holistic_features)
        
        # Simpan ke satu file CSV yang kaya informasi
        uv_holistic_df.to_csv(os.path.join(output_dir, "uv_holistic_shape_statistics.csv"), index=False)
        print(f"✅ Statistik Holistik UV-Vis disimpan di: {output_dir}")

    if not ir_files and not uv_files:
        print("❌ Tidak ada file IR atau UV-Vis yang ditemukan untuk dianalisis.")
        return
        
    print("\n--- Analisis Statistik Selesai ---")

if __name__ == '__main__':
    default_folder = "default_output/generate_rule_statistics"
    os.makedirs(default_folder, exist_ok=True)
    main(output_dir=default_folder)




     