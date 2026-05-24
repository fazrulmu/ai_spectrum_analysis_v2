# src/auto_labeler.py (Versi Lengkap dengan Rules IR & UV)

import numpy as np
from scipy.signal import find_peaks
import os 
import json
import re


def get_elements_from_molform(molform_string):
    """
    Mengekstrak elemen-elemen unik dari string rumus molekul (MOLFORM).
    Contoh: 'C6H6O' -> {'C', 'H', 'O'}
    """
    if not isinstance(molform_string, str):
        return set() # Kembalikan set kosong jika input tidak valid
        
    # Menemukan semua simbol elemen (huruf kapital diikuti huruf kecil opsional)
    elements = re.findall('[A-Z][a-z]*', molform_string)
    return set(elements)

def load_rules_from_json(file_path):
    """Memuat aturan spektral dari file JSON."""
    if not os.path.exists(file_path):
        print(f"⚠️  Peringatan: File aturan '{file_path}' tidak ditemukan. Menggunakan daftar kosong.")
        return []
    try:
        with open(file_path, 'r') as f:
            rules = json.load(f)
        # Konversi 'range' menjadi tuple dan 'required_atoms' menjadi set
        for rule in rules:
            if 'range' in rule:
                rule['range'] = tuple(rule['range'])
            if 'required_atoms' in rule:
                rule['required_atoms'] = set(rule['required_atoms'])
        return rules
    except (json.JSONDecodeError, TypeError) as e:
        print(f"❌ Error memuat atau mem-parsing file aturan '{file_path}': {e}")
        return []

# --- Tentukan path ke file JSON (relatif terhadap root proyek) ---
# Asumsi file ini ada di `src/`, dan file JSON ada di `configs/`
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
IR_RULES_PATH = os.path.join(PROJECT_ROOT, 'ir_rules.json')
UV_RULES_PATH = os.path.join(PROJECT_ROOT, 'uv_rules.json')

# --- Muat aturan secara dinamis ---
SPECTRAL_RULES = load_rules_from_json(IR_RULES_PATH)
UV_SPECTRAL_RULES = load_rules_from_json(UV_RULES_PATH)

# =================================================================
# == FUNGSI AUTO-LABELER UNTUK IR                                  ==
# =================================================================
def autogenerate_functional_groups(spectrum_df, metadata):
    """
    Melabeli spektrum IR secara otomatis berdasarkan aturan dan memvalidasi
    dengan rumus molekul dari metadata.
    """
    wavenumbers = spectrum_df['wavenumber'].values
    absorbances = spectrum_df['absorbance'].values
    detected_groups = set()

    # --- PERBAIKAN: Gunakan prominence rendah, filter dengan molform ---
    all_peaks, properties = find_peaks(absorbances, prominence=0.01, height=0.01, width=1)

    # Dapatkan elemen dari rumus molekul sekali saja
    molform = metadata.get('molform', '')
    present_elements = get_elements_from_molform(molform)

    for rule in SPECTRAL_RULES:
        # --- VALIDASI KIMIA: Periksa apakah atom yang dibutuhkan ada ---
        required_atoms = rule.get('required_atoms', set())
        if required_atoms and not required_atoms.issubset(present_elements):
            continue # Lewati aturan ini jika atom tidak ada

        for i, peak_idx in enumerate(all_peaks):
            wavenumber_at_peak = wavenumbers[peak_idx]
            prominence_at_peak = properties['prominences'][i]
            # --- PERBAIKAN: Gunakan min() dan max() untuk pengecekan rentang yang robust ---
            # Ini akan berfungsi baik untuk rentang (high, low) maupun (low, high).
            if min(rule['range']) <= wavenumber_at_peak <= max(rule['range']) and prominence_at_peak >= rule.get('min_prominence', 0.0):
                detected_groups.add(rule['group'])
                break # Lanjut ke aturan berikutnya setelah ditemukan kecocokan
    return list(detected_groups)

# =================================================================
# == FUNGSI AUTO-LABELER UNTUK UV-VIS                              ==
# =================================================================
def autogenerate_chromophores(spectrum_df, metadata):
    """
    Melabeli spektrum UV-Vis secara otomatis berdasarkan aturan dan memvalidasi
    dengan rumus molekul dari metadata.
    """
    wavelengths = spectrum_df['wavelength'].values
    log_epsilons = spectrum_df['log_epsilon'].values
    detected_groups = set()

    # --- PERBAIKAN: Gunakan prominence rendah, filter dengan molform ---
    all_peaks, properties = find_peaks(log_epsilons, prominence=0.01, height=0.01, width=3)

    # Dapatkan elemen dari rumus molekul sekali saja
    molform = metadata.get('molform', '')
    present_elements = get_elements_from_molform(molform)

    for rule in UV_SPECTRAL_RULES:
        # --- VALIDASI KIMIA: Periksa apakah atom yang dibutuhkan ada ---
        required_atoms = rule.get('required_atoms', set())
        if required_atoms and not required_atoms.issubset(present_elements):
            continue # Lewati aturan ini jika atom tidak ada

        for i, peak_idx in enumerate(all_peaks):
            wavelength_at_peak = wavelengths[peak_idx]
            prominence_at_peak = properties['prominences'][i]
            log_epsilon_at_peak = log_epsilons[peak_idx]
            # --- PERBAIKAN: Gunakan min() dan max() untuk konsistensi ---
            if min(rule['range']) <= wavelength_at_peak <= max(rule['range']) and \
               prominence_at_peak >= rule.get('min_prominence', 0.0) and log_epsilon_at_peak >= rule.get('min_log_epsilon', 0.0):
                detected_groups.add(rule['group'])
                break # Lanjut ke aturan berikutnya setelah ditemukan kecocokan
    return list(detected_groups)