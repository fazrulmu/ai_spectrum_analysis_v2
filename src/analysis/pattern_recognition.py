# src/pattern_recognition.py

import numpy as np
import os
import json
from scipy.signal import find_peaks

def load_substitution_rules(file_path):
    """Memuat aturan pola substitusi dari file JSON."""
    if not os.path.exists(file_path):
        print(f"⚠️  Peringatan: File aturan '{file_path}' tidak ditemukan. Menggunakan daftar kosong.")
        return []
    try:
        with open(file_path, 'r') as f:
            rules = json.load(f)
        return rules
    except (json.JSONDecodeError, TypeError) as e:
        print(f"❌ Error memuat atau mem-parsing file aturan '{file_path}': {e}")
        return []

# --- 1. Muat Aturan Pola Substitusi dari JSON ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
SUBSTITUTION_RULES_PATH = os.path.join(PROJECT_ROOT, 'substitution_rules.json')
OVERTONE_RULES_PATH = os.path.join(PROJECT_ROOT, 'overtone_rules.json')
SUBSTITUTION_RULES = load_substitution_rules(SUBSTITUTION_RULES_PATH)
OVERTONE_RULES = load_substitution_rules(OVERTONE_RULES_PATH)


def check_peak_in_range(peak_locations, required_range):
    """Fungsi helper untuk memeriksa apakah ada puncak dalam rentang yang diberikan."""
    for peak_loc in peak_locations:
        if required_range[0] <= peak_loc <= required_range[1]:
            return True
    return False

def recognize_substitution_pattern(spectrum_df):
    """
    Menganalisis DataFrame spektrum IR yang sudah diproses dan mencoba
    mengenali pola substitusi aromatik.
    """
    # --- 2. Fokus pada Daerah Sidik Jari yang Relevan ---
    fingerprint_df = spectrum_df[
        (spectrum_df['wavenumber'] >= 650) & (spectrum_df['wavenumber'] <= 900)
    ].copy()

    if fingerprint_df.empty:
        return "No data in fingerprint region"

    # --- 3. Cari Puncak yang Signifikan di Daerah Tersebut ---
    # --- PERBAIKAN: Set prominence ke 0 untuk sensitivitas maksimum ---
    # Ini akan mendeteksi semua puncak lokal, tidak peduli seberapa kecilnya.
    peaks_indices, _ = find_peaks(fingerprint_df['absorbance'], prominence=0)
    
    # Jika tidak ada puncak yang ditemukan sama sekali
    if len(peaks_indices) == 0:
        return "No significant peaks found in 900-650 cm-1 region"
        
    found_peak_locations = fingerprint_df['wavenumber'].iloc[peaks_indices].values
    
    # --- 4. Cocokkan Puncak dengan Aturan ---
    for rule in SUBSTITUTION_RULES:
        pattern = rule['pattern']
        required_ranges = rule['ranges']
        all_conditions_met = True
        for req_range in required_ranges:
            if not check_peak_in_range(found_peak_locations, req_range):
                all_conditions_met = False
                break  # Jika satu syarat tidak terpenuhi, lanjut ke aturan berikutnya
        
        if all_conditions_met:
            # Jika semua syarat untuk satu pola terpenuhi, kembalikan hasilnya
            return pattern
        

def recognize_overtone_pattern(spectrum_df):
    """
    Mencoba mengenali pola 'jari-jari' overtone di daerah 2000-1665 cm-1.
    """
    # 1. Isolasi Daerah Overtone
    # Kita menggunakan 'wavenumber' untuk IR dan 'absorbance_norm' jika ada
    overtone_region = spectrum_df[
        (spectrum_df['wavenumber'] >= 1665) & (spectrum_df['wavenumber'] <= 2000)
    ].copy()

    if overtone_region.empty or len(overtone_region) < 10:
        return "No data in overtone region"

    # Gunakan absorbansi yang sudah dinormalisasi secara lokal jika ada
    y_col = 'absorbance_norm' if 'absorbance_norm' in overtone_region.columns else 'absorbance'
    y_values = overtone_region[y_col].values
    
    # Normalisasi lokal jika belum ada
    if y_col == 'absorbance':
        min_val = np.min(y_values)
        max_val = np.max(y_values)
        if max_val - min_val < 0.01: # Jika daerah ini hampir datar
            return "Overtone region is flat"
        y_values = (y_values - min_val) / (max_val - min_val)

    # 3. Hitung Jumlah Puncak (puncak overtone sangat lemah)
    overtone_peaks, _ = find_peaks(y_values, prominence=0.3, width=2)
    num_peaks = len(overtone_peaks)

    # 4. Cocokkan Pola menggunakan aturan dari JSON
    for rule in OVERTONE_RULES:
        if rule['num_peaks_min'] <= num_peaks <= rule['num_peaks_max']:
            return rule['pattern']

    return "No clear overtone pattern detected"


            
    