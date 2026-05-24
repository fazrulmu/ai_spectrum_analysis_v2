# src/train_pattern_model.py

import os
import glob
import numpy as np
import pandas as pd
import yaml
import joblib
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

from data.data_processing import parse_jdx, preprocess_spectrum
from analysis.pattern_recognition import SUBSTITUTION_RULES # Impor aturan

def generate_synthetic_fingerprint(pattern_rule, grid, all_rules):
    """
    Membuat satu sampel spektrum sidik jari sintetis berdasarkan aturan pola.
    """
    # 1. Buat baseline dengan noise
    baseline = np.random.normal(0, 0.03, len(grid)) + \
               np.sin(np.linspace(0, np.random.uniform(1, 5) * np.pi, len(grid))) * 0.05
    
    y_values = baseline
    
    # 2. Tambahkan puncak sesuai aturan
    for r in pattern_rule['ranges']:
        # Posisi puncak diacak di dalam rentang aturan
        peak_pos = np.random.uniform(min(r), max(r))
        # Tinggi puncak diacak
        peak_height = np.random.uniform(0.4, 1.0)
        # Lebar puncak diacak
        peak_width = np.random.uniform(5, 15)
        
        peak = peak_height * np.exp(-((grid - peak_pos)**2) / (2 * peak_width**2))
        y_values += peak

    # --- PERBAIKAN: Tambahkan "puncak pengganggu" dari aturan lain yang mirip ---
    # Ini akan melatih model untuk lebih tangguh terhadap noise dan overlap.
    if np.random.rand() < 0.4: # 40% kemungkinan untuk menambahkan pengganggu
        # Pilih aturan lain secara acak, yang bukan aturan saat ini
        distractor_rule = pattern_rule
        while distractor_rule['pattern'] == pattern_rule['pattern']:
            distractor_rule = np.random.choice(all_rules)
        
        # Pilih satu rentang acak dari aturan pengganggu
        distractor_range = np.random.choice(distractor_rule['ranges'])
        
        # Tambahkan puncak pengganggu yang lemah
        peak_pos = np.random.uniform(min(distractor_range), max(distractor_range))
        peak_height = np.random.uniform(0.1, 0.3) # Jauh lebih lemah dari puncak utama
        peak_width = np.random.uniform(3, 10)
        peak = peak_height * np.exp(-((grid - peak_pos)**2) / (2 * peak_width**2))
        y_values += peak

    # 3. Tambahkan beberapa puncak noise acak
    for _ in range(np.random.randint(0, 3)):
        peak_pos = np.random.uniform(grid.min(), grid.max())
        peak_height = np.random.uniform(0.05, 0.2)
        peak_width = np.random.uniform(3, 10)
        peak = peak_height * np.exp(-((grid - peak_pos)**2) / (2 * peak_width**2))
        y_values += peak

    # 4. Normalisasi final ke 0-1
    min_val, max_val = y_values.min(), y_values.max()
    if max_val > min_val:
        y_norm = (y_values - min_val) / (max_val - min_val)
    else:
        y_norm = y_values
        
    return y_norm

def prepare_synthetic_pattern_dataset(config, samples_per_pattern=500):
    """
    Membangun dataset latih untuk model pengenalan pola dengan
    membuat data spektrum sintetis.
    """
    # Gunakan grid yang lebih padat di daerah fingerprint
    FINGERPRINT_START = 650
    FINGERPRINT_STOP = 900 # Fokus pada daerah aturan substitusi
    FINGERPRINT_POINTS = 250 
    fingerprint_grid = np.linspace(FINGERPRINT_START, FINGERPRINT_STOP, FINGERPRINT_POINTS)

    all_spectra = []
    all_labels = []

    print(f"🧬 Membuat data sintetis untuk {len(SUBSTITUTION_RULES)} pola...")
    for rule in tqdm(SUBSTITUTION_RULES, desc="Generating Synthetic Data"):
        pattern_name = rule['pattern']
        for _ in range(samples_per_pattern):
            synthetic_spectrum = generate_synthetic_fingerprint(rule, fingerprint_grid, SUBSTITUTION_RULES)
            all_spectra.append(synthetic_spectrum)
            all_labels.append(pattern_name)

    return np.array(all_spectra), np.array(all_labels)

def train_pattern_model(config):
    """Fungsi utama untuk melatih dan menyimpan model pengenalan pola."""
    paths = config['paths']
    X, y = prepare_synthetic_pattern_dataset(config)

    if len(X) == 0:
        print("❌ Tidak ada data yang cukup untuk melatih model pengenalan pola.")
        return

    print(f"\nTotal {len(X)} sampel valid ditemukan untuk pelatihan.")
    
    # Encode label string menjadi integer
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.25, random_state=42, stratify=y_encoded)

    # Latih model (Random Forest adalah pilihan yang baik dan cepat)
    print("🌳 Melatih model RandomForestClassifier...")
    model = RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    # Evaluasi
    y_pred = model.predict(X_test)
    print("\n--- Laporan Klasifikasi Model Pola Substitusi ---")
    print(classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0))

    # Simpan model dan encoder
    # --- PERBAIKAN: Pastikan direktori untuk menyimpan model sudah ada ---
    os.makedirs(paths['saved_models_dir'], exist_ok=True)

    model_save_path = os.path.join(paths['saved_models_dir'], 'pattern_recognition_model.joblib')
    encoder_save_path = os.path.join(paths['saved_models_dir'], 'pattern_label_encoder.joblib')
    joblib.dump(model, model_save_path)
    joblib.dump(le, encoder_save_path)

    print(f"\n✅ Model pengenalan pola disimpan di: {model_save_path}")
    print(f"✅ Encoder label disimpan di: {encoder_save_path}")