# screen_and_label.py (Versi Final yang Sangat Tangguh)

import os
import yaml
import glob
import re
import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.signal import find_peaks
from rdkit import Chem
from rdkit.Chem import Descriptors
import matplotlib.pyplot as plt
# Impor fungsi dari proyek Anda

# Impor fungsi dari proyek Anda
from src.data_processing import parse_jdx, preprocess_spectrum
from src.auto_labeler import SPECTRAL_RULES, UV_SPECTRAL_RULES

# --- BAGIAN 1 & 2: OTAK KIMIA & FUNGSI ANALISIS (Tidak berubah) ---
NAME_TO_SMILES = {
    "benzene": "c1ccccc1", "toluene": "Cc1ccccc1", "ethanol": "CCO", "acetic acid": "CC(=O)O",
    "aspirin": "CC(=O)OC1=CC=CC=C1C(=O)O", "acetyl chloride": "CC(=O)Cl", "cyclohexane": "C1CCCCC1",
    "carbon disulfide": "C(=S)=S"
}
FUNCTIONAL_GROUP_SMARTS = {
    "alcohol_phenol_oh": "[#6][OX2H]", "carboxylic_acid_oh_broad": "[CX3](=O)[OX2H]",
    "alkane_ch_sp3": "[CX4]", "aromatic_cc": "a", "ester_co": "[#6][CX3](=O)[OX2][#6]",
    "ester_co_c": "[#6][CX3](=O)[OX2][#6]", "ketone_co": "[#6][CX3](=O)[#6]",
    "aldehyde_co": "[CX3H1](=O)[#6]", "alkene_cc": "[#6]=[#6]", "alkyne_cc": "[#6]#[#6]",
    "alkyne_ch_sp": "[C]#[CH1]", "amine_cn": "[NX3;H2,H1;!$(NC=O)][#6]",
    "amide_cn": "[NX3][CX3](=[OX1])[#6]", "ether_co": "[OD2]([#6])[#6]",
}
def get_expected_labels_from_name(name):
    name_clean = name.lower().split(',')[0].strip(); smiles = NAME_TO_SMILES.get(name_clean)
    if not smiles: return set()
    mol = Chem.MolFromSmiles(smiles);
    if not mol: return set()
    expected = set()
    for group, smarts in FUNCTIONAL_GROUP_SMARTS.items():
        patt = Chem.MolFromSmarts(smarts)
        if mol.HasSubstructMatch(patt): expected.add(group)
    if "carboxylic_acid_oh_broad" in expected: expected.add("carboxylic_acid_co")
    return expected

def calculate_peak_area(y_data, x_data, peak_index, peak_properties):
    start_index = int(peak_properties['left_ips'][peak_index]); end_index = int(peak_properties['right_ips'][peak_index])
    x_peak = x_data[start_index:end_index+1]; y_peak = y_data[start_index:end_index+1]
    if len(x_peak) < 2: return 0.0
    baseline = np.linspace(y_peak[0], y_peak[-1], len(y_peak)); y_corrected = y_peak - baseline
    y_corrected[y_corrected < 0] = 0; area = np.trapezoid(y_corrected, x=x_peak)
    return abs(area)

# --- BAGIAN 3: PIPELINE SCREENING UTAMA (Dengan Logika Fallback) ---
def screen_all_spectra(config):
    paths = config['paths']; all_peak_data = []
    all_jdx_files = glob.glob(os.path.join(paths['raw_data_dir'], '**', '*.jdx'), recursive=True)
    ir_rule_map = {rule['group']: rule for rule in SPECTRAL_RULES}
    uv_rule_map = {rule['group']: rule for rule in UV_SPECTRAL_RULES}

    for file_path in tqdm(all_jdx_files, desc="Screening Spectra"):
        try:
            raw_data = parse_jdx(file_path)
            if not raw_data or len(raw_data['x']) != len(raw_data['y']): continue
            metadata = raw_data.get('metadata', {}); title = metadata.get('title', 'N/A')
            spectrum_type = 'ir' if re.search(r'(ftir|ft-ir|ir)', os.path.basename(file_path).lower()) else ('uv' if re.search(r'uv[\-\s]?vis', os.path.basename(file_path).lower()) else None)
            if not spectrum_type: continue
            
            processed_df = preprocess_spectrum(raw_data, config, spectrum_type, normalize=False)
            if processed_df is None: continue

            y_col = 'absorbance' if spectrum_type == 'ir' else 'log_epsilon'
            x_col = 'wavenumber' if spectrum_type == 'ir' else 'wavelength'
            y_vals = processed_df[y_col].values; x_vals = processed_df[x_col].values
            y_vals_norm = (y_vals - y_vals.min()) / (y_vals.max() - y_vals.min() + 1e-9)

            all_peaks_indices, props = find_peaks(y_vals_norm, prominence=0.015, width=1) # Sedikit menurunkan prominence
            if all_peaks_indices.size == 0: continue

            detected_peaks = []
            for i, peak_idx in enumerate(all_peaks_indices):
                detected_peaks.append({
                    "index": peak_idx, "location": x_vals[peak_idx],
                    "height": props["peak_heights"][i], "prominence": props["prominences"][i],
                    "width": props["widths"][i], "area": calculate_peak_area(y_vals_norm, x_vals, i, props)
                })

            # =========================================================
            # == UPGRADE: LOGIKA IF/ELSE UNTUK PENUGASAN LABEL       ==
            # =========================================================
            expected_labels = get_expected_labels_from_name(title)
            
            # --- JALUR 1: Jika nama senyawa dikenali (Penugasan Cerdas) ---
            if expected_labels:
                peak_assignments = {peak['index']: 'unassigned' for peak in detected_peaks}
                available_peaks = detected_peaks.copy()
                rule_map = ir_rule_map if spectrum_type == 'ir' else uv_rule_map

                for label in expected_labels:
                    if label not in rule_map: continue
                    rule = rule_map[label]
                    best_peak = None; highest_prominence = -1
                    for peak in available_peaks:
                        if rule['range'][0] <= peak['location'] <= rule['range'][1] and peak['prominence'] >= rule.get('min_prominence', 0):
                            if peak['prominence'] > highest_prominence: best_peak = peak; highest_prominence = peak['prominence']
                    if best_peak:
                        peak_assignments[best_peak['index']] = label
                        available_peaks.remove(best_peak)
                
                # Masukkan hasil penugasan cerdas
                for peak in detected_peaks: peak['assigned_label'] = peak_assignments[peak['index']]

            # --- JALUR 2: Jika nama tidak dikenali (Fallback ke Deteksi Sederhana) ---
            else:
                rule_map = SPECTRAL_RULES if spectrum_type == 'ir' else UV_SPECTRAL_RULES
                for peak in detected_peaks:
                    assigned_label = 'unassigned'
                    for rule in rule_map:
                        if rule['range'][0] <= peak['location'] <= rule['range'][1]:
                            assigned_label = rule['group']; break
                    peak['assigned_label'] = assigned_label
            # =========================================================

            # Kumpulkan hasil
            for peak in detected_peaks:
                all_peak_data.append({
                    "cas": metadata.get('cas_registry_no', 'N/A'), "molform": metadata.get('molform', 'N/A'),
                    "title": title, "expected_labels": sorted(list(expected_labels)) if expected_labels else [],
                    "assigned_label": peak['assigned_label'], "peak_location": peak['location'],
                    "peak_prominence": peak['prominence'], "peak_width": peak['width'], "peak_area": peak['area'],
                })

        except Exception: continue
            
    return pd.DataFrame(all_peak_data)

if __name__ == "__main__":
    # ... (sisa kode __main__ tidak perlu diubah) ...
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    with open('main_config.yaml', 'r') as f: config = yaml.safe_load(f)
    output_dir = os.path.join(config['paths']['reports_dir'], 'screening')
    os.makedirs(output_dir, exist_ok=True)
    master_peak_df = screen_all_spectra(config)
    if not master_peak_df.empty:
        csv_path = os.path.join(output_dir, f'peak_proportion_database_{timestamp}.csv')
        master_peak_df.to_csv(csv_path, index=False)
        print(f"\n✅ Screening selesai. Database proporsi puncak disimpan di: {csv_path}")
    else:
        print("\nTidak ada puncak yang terdeteksi di seluruh dataset.")