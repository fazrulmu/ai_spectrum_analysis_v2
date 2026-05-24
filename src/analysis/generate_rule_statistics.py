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
import json

# Mengimpor fungsi dan aturan dari file lain
from data.data_processing import parse_jdx, preprocess_spectrum

# =================================================================
# == TEMPLATE ATURAN DASAR (Akan diisi oleh skrip ini)          ==
# =================================================================

BASE_IR_RULES = [
    {"group": "alkene_ch_sp2", "range": (3010, 3100), "vibration_type": "sp² C-H stretch", "required_atoms": {'C', 'H'}},
    {"group": "alkane_ch_sp3", "range": (2850, 2960), "vibration_type": "sp³ C-H stretch", "required_atoms": {'C', 'H'}},
    {"group": "carboxylic_acid_co", "range": (1700, 1725), "vibration_type": "C=O stretch", "required_atoms": {'C', 'O'}},
    {"group": "aromatic_cc", "range": (1450, 1600), "vibration_type": "Aromatic C=C ring stretch", "required_atoms": {'C'}},
    {"group": "amide_cn", "range": (1180, 1360), "vibration_type": "C-N stretch", "required_atoms": {'C', 'N'}},
    {"group": "ester_co_c", "range": (1150, 1250), "vibration_type": "C-O stretch (ester)", "required_atoms": {'C', 'O'}},
    {"group": "ether_co", "range": (1050, 1150), "vibration_type": "C-O stretch", "required_atoms": {'C', 'O'}},
    {"group": "benzene_ring_breathing", "range": (1000, 1075), "vibration_type": "Benzene ring mode", "required_atoms": {'C', 'H'}},
    {"group": "alkene_out_of_plane_ch", "range": (900, 990), "vibration_type": "Alkene C-H bend (out-of-plane)", "required_atoms": {'C', 'H'}},
    {"group": "alkyl_halide_ccl", "range": (600, 800), "vibration_type": "C-Cl stretch", "required_atoms": {'C', 'Cl'}},
    {"group": "alkyl_halide_cbr", "range": (500, 650), "vibration_type": "C-Br stretch", "required_atoms": {'C', 'Br'}},
    {"group": "alkyl_halide_cI", "range": (450, 600), "vibration_type": "C-I stretch", "required_atoms": {'C', 'I'}},
    {"group": "amide_co", "range": (1640, 1690), "vibration_type": "C=O stretch (amide I band)", "required_atoms": {'C', 'O', 'N'}},
    {"group": "carboxylic_acid_oh_broad", "range": (2500, 3300), "vibration_type": "O-H stretch (very broad)", "required_atoms": {'C', 'O', 'H'}},
    {"group": "anhydride_co", "range": (1740, 1850), "vibration_type": "C=O stretch (anhydride, 2 bands)", "required_atoms": {'C', 'O'}},
    {"group": "aldehyde_co", "range": (1720, 1740), "vibration_type": "C=O stretch", "required_atoms": {'C', 'O', 'H'}},
    {"group": "aromatic_out_of_plane_ch", "range": (690, 900), "vibration_type": "Aromatic C-H bend (out-of-plane)", "required_atoms": {'C', 'H'}},
    {"group": "alkyne_cc", "range": (2100, 2260), "vibration_type": "C≡C stretch", "required_atoms": {'C'}},
    {"group": "alkene_cc", "range": (1620, 1680), "vibration_type": "C=C stretch", "required_atoms": {'C'}},
    {"group": "alcohol_phenol_oh", "range": (3200, 3600), "vibration_type": "O-H stretch (broad)", "required_atoms": {'O', 'H'}},
    {"group": "amine_nh_secondary", "range": (3300, 3500), "vibration_type": "N-H stretch (1 peak)", "required_atoms": {'N', 'H'}},
    {"group": "nitro_no2_sym", "range": (1300, 1370), "vibration_type": "Symmetric N-O stretch", "required_atoms": {'N', 'O'}},
    {"group": "thiol_sh", "range": (2550, 2700), "vibration_type": "S-H stretch, weak", "required_atoms": {'S', 'H'}},
    {"group": "nitrile_cn", "range": (2210, 2260), "vibration_type": "C≡N stretch", "required_atoms": {'C', 'N'}},
    {"group": "ketone_co", "range": (1705, 1725), "vibration_type": "C=O stretch", "required_atoms": {'C', 'O'}},
    {"group": "ester_co", "range": (1735, 1750), "vibration_type": "C=O stretch", "required_atoms": {'C', 'O'}},
    {"group": "amine_cn", "range": (1020, 1230), "vibration_type": "C-N stretch", "required_atoms": {'C', 'N'}},
    {"group": "alkyne_ch_sp", "range": (3300, 3320), "vibration_type": "sp C-H stretch, sharp", "required_atoms": {'C', 'H'}},
    {"group": "isocyanate_nco", "range": (2250, 2270), "vibration_type": "N=C=O asymmetric stretch", "required_atoms": {'N', 'C', 'O'}},
    {"group": "aromatic_ch", "range": (3000, 3100), "vibration_type": "Aromatic C-H stretch", "required_atoms": {'C', 'H'}},
    {"group": "amine_nh_primary", "range": (3300, 3500), "vibration_type": "N-H stretch (primary, 2 bands)", "required_atoms": {'N', 'H'}},
    {"group": "amide_nh", "range": (3100, 3500), "vibration_type": "N-H stretch", "required_atoms": {'C', 'O', 'N', 'H'}},
    {"group": "sulfone_so2", "range": (1300, 1350), "vibration_type": "S=O stretch", "required_atoms": {'S', 'O'}},
    {"group": "sulfate_so4", "range": (1050, 1150), "vibration_type": "S=O stretch", "required_atoms": {'S', 'O'}},
    {"group": "nitro_no2_asym", "range": (1500, 1570), "vibration_type": "Asymmetric N-O stretch", "required_atoms": {'N', 'O'}},
]

BASE_UV_RULES = [
    {"group": "alkene_isolated", "range": (170, 195), "comment": "Transisi π -> π* pada C=C terisolasi.", "required_atoms": {'C'}},
    {"group": "carbonyl_pi_pi_star", "range": (180, 195), "comment": "Transisi π -> π* yang kuat pada C=O.", "required_atoms": {'C', 'O'}},
    {"group": "carbonyl_n_pi_star", "range": (270, 300), "comment": "Transisi n -> π* yang lemah pada C=O.", "required_atoms": {'C', 'O'}},
    {"group": "diene_conjugated_acyclic", "range": (215, 230), "comment": "Transisi π -> π* untuk diena terkonjugasi.", "required_atoms": {'C'}},
    {"group": "diene_conjugated_cyclic", "range": (250, 270), "comment": "Diena terkonjugasi dalam cincin.", "required_atoms": {'C'}},
    {"group": "enone_pi_pi_star", "range": (210, 250), "comment": "Transisi π -> π* (pita K) yang kuat pada enon (C=C-C=O).", "required_atoms": {'C', 'O'}},
    {"group": "enone_n_pi_star", "range": (310, 330), "comment": "Transisi n -> π* (pita R) yang lemah, bergeser ke λ lebih panjang.", "required_atoms": {'C', 'O'}},
    {"group": "benzene_primary_E2_band", "range": (200, 210), "comment": "Pita E2 primer benzena.", "required_atoms": {'C'}},
    {"group": "benzene_secondary_B_band", "range": (250, 270), "comment": "Pita B sekunder benzena (struktur halus).", "required_atoms": {'C'}},
    {"group": "phenol_or_aniline", "range": (270, 285), "comment": "Gugus -OH atau -NH2 menggeser pita B.", "required_atoms": {'C'}}, # C is common to both
    {"group": "styrene_conjugated_aromatic", "range": (245, 260), "comment": "Konjugasi cincin benzena dengan C=C.", "required_atoms": {'C'}},
    {"group": "benzaldehyde_conjugated_aromatic", "range": (240, 255), "comment": "Konjugasi cincin benzena dengan C=O.", "required_atoms": {'C', 'O'}},
    {"group": "naphthalene_polycyclic_aromatic", "range": (300, 320), "comment": "Sistem polisiklik aromatik seperti naftalena.", "required_atoms": {'C'}},
]
def analyze_uv_spectrum_features(spectrum_df):
    """
    Menganalisis spektrum UV, menemukan puncak yang cocok dengan aturan,
    dan mengekstrak fitur-fiturnya.
    """
    wavelengths = spectrum_df['wavelength'].values
    log_epsilons = spectrum_df['log_epsilon'].values

    # Temukan semua puncak yang signifikan
    peaks, properties = find_peaks(log_epsilons, prominence=0.01, height=0.01)
    if len(peaks) == 0:
        return []

    found_peaks = []
    for i, peak_idx in enumerate(peaks):
        wavelength = wavelengths[peak_idx]
        prominence = properties['prominences'][i]
        log_epsilon_val = log_epsilons[peak_idx]

        # Cocokkan puncak dengan aturan UV
        for rule in BASE_UV_RULES:
            if rule['range'][0] <= wavelength <= rule['range'][1]:
                # Periksa kondisi prominence dan log_epsilon
                if prominence >= rule.get('min_prominence', 0) and log_epsilon_val >= rule.get('min_log_epsilon', 0):
                    found_peaks.append({
                        "group": rule['group'],
                        "lambda_max": wavelength,
                        "intensity_at_lambda_max": log_epsilon_val,
                        "prominence": prominence
                    })
                    break  # Lanjut ke puncak berikutnya setelah ditemukan kecocokan
    return found_peaks

def analyze_ir_spectrum_features(spectrum_df):
    # ... (Fungsi ini tidak berubah) ...
    wavenumbers = spectrum_df['wavenumber'].values
    absorbances = spectrum_df['absorbance'].values
    peaks, properties = find_peaks(absorbances, prominence=0.02, height=0.02)
    if len(peaks) == 0: return []
    widths_data = peak_widths(absorbances, peaks, rel_height=0.5)
    peak_width_map = {peak: width for peak, width in zip(peaks, widths_data[0])}
    found_peaks = []
    for i, peak_idx in enumerate(peaks):
        wavenumber = wavenumbers[peak_idx]
        prominence = properties['prominences'][i]
        height = properties['peak_heights'][i]
        width = peak_width_map.get(peak_idx, 0)
        matched_group = "unknown"
        for rule in BASE_IR_RULES:
            # --- PERBAIKAN: Gunakan .get() untuk menghindari KeyError jika 'min_prominence' tidak ada ---
            if rule['range'][0] <= wavenumber <= rule['range'][1] and prominence >= rule.get('min_prominence', 0):
                passes_width_check = True
                if 'min_width' in rule and width < rule['min_width']: passes_width_check = False
                if 'max_width' in rule and width > rule['max_width']: passes_width_check = False
                if passes_width_check:
                    matched_group = rule['group']
                    break
        if matched_group != "unknown":
            found_peaks.append({"group": matched_group, "position": wavenumber, "prominence": prominence, "height": height, "width": width})
    return found_peaks

def analyze_uv_holistic_features(spectrum_df):
    # ... (Fungsi ini tidak berubah) ...
    wavelengths = spectrum_df['wavelength'].values
    absorbances = spectrum_df['absorbance'].values
    if len(absorbances) < 5 or np.all(absorbances == 0):
        return {'lambda_max': 0, 'intensity_at_lambda_max': 0, 'fwhm': 0, 'auc': 0, 'skewness': 0, 'kurtosis': 0, 'num_hidden_peaks_deriv2': 0}
    main_peak_idx = np.argmax(absorbances)
    lambda_max = wavelengths[main_peak_idx]
    intensity_at_lambda_max = absorbances[main_peak_idx]
    try:
        if intensity_at_lambda_max > 0.01:
            widths_data = peak_widths(absorbances, [main_peak_idx], rel_height=0.5)
            fwhm = widths_data[0][0] if len(widths_data[0]) > 0 else 0
        else: fwhm = 0
    except Exception: fwhm = 0
    auc = simps(absorbances, dx=(wavelengths[1]-wavelengths[0])) if len(wavelengths) > 1 else 0
    skewness = skew(absorbances)
    kurtosis_val = kurtosis(absorbances)
    second_derivative = np.gradient(np.gradient(absorbances))
    deriv_peaks, _ = find_peaks(-second_derivative, prominence=np.std(second_derivative))
    num_hidden_peaks = len(deriv_peaks)
    return {'lambda_max': lambda_max, 'intensity_at_lambda_max': intensity_at_lambda_max, 'fwhm': fwhm, 'auc': auc, 'skewness': skewness, 'kurtosis': kurtosis_val, 'num_hidden_peaks_deriv2': num_hidden_peaks}


def calculate_ir_feature_statistics(all_spectra_features, ratio_pairs):
    """
    Menghitung statistik agregat untuk fitur IR.
    """
    # --- PERUBAHAN DI SINI: Tambahkan 'height' ke dalam group_stats ---
    group_stats = collections.defaultdict(lambda: collections.defaultdict(list))
    for spectrum_features in all_spectra_features:
        for peak in spectrum_features:
            group = peak['group']
            group_stats[group]['position'].append(peak['position'])
            group_stats[group]['prominence'].append(peak['prominence'])
            group_stats[group]['width'].append(peak['width'])
            group_stats[group]['height'].append(peak['height']) # <-- BARIS BARU

    # ... (Sisa fungsi ini tidak berubah) ...
    summary = {}
    for group, data in group_stats.items():
        summary[group] = {"count": len(data['position']), "mean_position": np.mean(data['position']), "std_position": np.std(data['position']), "mean_prominence": np.mean(data['prominence']), "std_prominence": np.std(data['prominence']), "mean_width": np.mean(data['width']), "std_width": np.std(data['width'])}
    basic_stats_df = pd.DataFrame.from_dict(summary, orient='index')

    co_occurrence = collections.Counter()
    for spectrum_features in all_spectra_features:
        present_groups = sorted(list(set([peak['group'] for peak in spectrum_features])))
        for pair in combinations(present_groups, 2):
            co_occurrence[pair] += 1
    co_occurrence_df = pd.DataFrame(co_occurrence.items(), columns=['group_pair', 'count']).sort_values(by='count', ascending=False)

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
        ratio_summary[name] = {"count": len(values), "mean_ratio": np.mean(values), "std_ratio": np.std(values), "min_ratio": np.min(values), "max_ratio": np.max(values)}
    ratio_stats_df = pd.DataFrame.from_dict(ratio_summary, orient='index')
    
    return basic_stats_df, co_occurrence_df, ratio_stats_df, group_stats


def suggest_new_rules_from_stats(group_stats):
    """
    Menganalisis statistik mentah dan menyarankan nilai-nilai aturan baru.
    """
    suggestions = {}
    for group, data in group_stats.items():
        if len(data['prominence']) < 10: 
            continue
            
        prominences = np.array(data['prominence'])
        widths = np.array(data['width'])
        heights = np.array(data['height'])
        
        suggested_min_prominence = float(np.percentile(prominences, 10))
        suggested_min_width = float(np.percentile(widths, 10))
        suggested_max_width = float(np.percentile(widths, 95))
        suggested_min_height = float(np.percentile(heights, 10))
        
        suggestions[group] = {
            "min_prominence": round(suggested_min_prominence, 4),
            "min_height": round(suggested_min_height, 4),
            "min_width": round(suggested_min_width, 2),
            "max_width": round(suggested_max_width, 2)
        }
        
    return suggestions

def suggest_new_uv_rules_from_stats(group_stats):
    """
    Menganalisis statistik mentah UV dan menyarankan nilai-nilai aturan baru.
    """
    suggestions = {}
    for group, data in group_stats.items():
        if len(data['prominence']) < 5:  # Butuh setidaknya beberapa sampel
            continue

        prominences = np.array(data['prominence'])
        intensities = np.array(data['intensity_at_lambda_max'])

        # Sarankan nilai pada persentil ke-10 untuk menjadi lebih inklusif
        suggested_min_prominence = float(np.percentile(prominences, 10))
        suggested_min_log_epsilon = float(np.percentile(intensities, 10))

        suggestions[group] = {
            "min_prominence": round(suggested_min_prominence, 4),
            "min_log_epsilon": round(suggested_min_log_epsilon, 4)
        }

    return suggestions


def main(output_dir, config_path):
    """
    Fungsi utama untuk memindai semua data, mengekstrak fitur, 
    dan menghasilkan statistik dalam file CSV.
    """
    print("--- 📊 Memulai Analisis Statistik Aturan Spektral (IR & UV-Vis) ---")

    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"❌ Error: File konfigurasi '{config_path}' tidak ditemukan.")
        return

    config_dir = os.path.dirname(config_path)
    raw_data_dir = config['paths']['raw_data_dir']
    all_jdx_files = glob.glob(os.path.join(raw_data_dir, '**', '*.jdx'), recursive=True)
    ir_files = [f for f in all_jdx_files if re.search(r'(ftir|ft-ir|ir)', os.path.basename(f), re.IGNORECASE)]
    uv_files = [f for f in all_jdx_files if re.search(r'uv[\-\s]?vis', os.path.basename(f), re.IGNORECASE)]

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
            ('carboxylic_acid_co', 'carboxylic_acid_oh_broad'), ('ester_co', 'ether_co'),
            ('ketone_co', 'alkane_ch_sp3'), ('aromatic_ch', 'aromatic_cc')
        ]
        ir_basic, ir_cooc, ir_ratio, ir_group_stats = calculate_ir_feature_statistics(all_ir_features, ir_ratio_pairs)
        
        # Generate and save new IR rules as JSON
        ir_suggestions = suggest_new_rules_from_stats(ir_group_stats)
        updated_ir_rules = []
        for rule in BASE_IR_RULES:
            if rule['group'] in ir_suggestions:
                # Create a copy and update it
                new_rule = rule.copy()
                new_rule.update(ir_suggestions[rule['group']])
                # Convert set to list for JSON serialization
                if 'required_atoms' in new_rule:
                    new_rule['required_atoms'] = sorted(list(new_rule['required_atoms']))
                updated_ir_rules.append(new_rule)
            else:
                # Also convert set to list for rules that are not updated
                new_rule = rule.copy()
                if 'required_atoms' in new_rule:
                    new_rule['required_atoms'] = sorted(list(new_rule['required_atoms']))
                updated_ir_rules.append(new_rule)
        ir_rules_path = os.path.join(config_dir, 'ir_rules.json')
        with open(ir_rules_path, 'w') as f:
            json.dump(updated_ir_rules, f, indent=4)
        print(f"✅ Aturan IR yang diperbarui disimpan sebagai JSON di: {ir_rules_path}")
        
    if uv_files:
        print(f"\n🔬 Menganalisis {len(uv_files)} file spektrum UV-Vis...")
        all_uv_features = []
        for file_path in tqdm(uv_files, desc="Memproses UV-Vis"):
            raw_data = parse_jdx(file_path)
            if not raw_data: continue
            # Gunakan normalize=False untuk mendapatkan nilai log_epsilon asli
            processed_df = preprocess_spectrum(raw_data, config, 'uv', normalize=False) 
            if processed_df is None or processed_df.empty: continue
            features = analyze_uv_spectrum_features(processed_df)
            if features: all_uv_features.append(features)

        # Hitung statistik untuk setiap grup UV
        uv_group_stats = collections.defaultdict(lambda: collections.defaultdict(list))
        for spectrum_features in all_uv_features:
            for peak in spectrum_features:
                for key, value in peak.items():
                    uv_group_stats[peak['group']][key].append(value)

        # Generate and save new UV rules as JSON
        uv_suggestions = suggest_new_uv_rules_from_stats(uv_group_stats)
        updated_uv_rules = []
        for rule in BASE_UV_RULES:
            if rule['group'] in uv_suggestions:
                new_rule = rule.copy()
                new_rule.update(uv_suggestions[rule['group']])
                # Convert set to list for JSON serialization
                if 'required_atoms' in new_rule:
                    new_rule['required_atoms'] = sorted(list(new_rule['required_atoms']))
                updated_uv_rules.append(new_rule)
            else:
                # Also convert set to list for rules that are not updated
                new_rule = rule.copy()
                if 'required_atoms' in new_rule:
                    new_rule['required_atoms'] = sorted(list(new_rule['required_atoms']))
                updated_uv_rules.append(new_rule)
        uv_rules_path = os.path.join(config_dir, 'uv_rules.json')
        with open(uv_rules_path, 'w') as f:
            json.dump(updated_uv_rules, f, indent=4)
        print(f"✅ Aturan UV-Vis yang diperbarui disimpan sebagai JSON di: {uv_rules_path}")

    if not ir_files and not uv_files:
        print("❌ Tidak ada file IR atau UV-Vis yang ditemukan untuk dianalisis.")
        return
        
    print("\n--- Analisis Statistik Selesai ---")