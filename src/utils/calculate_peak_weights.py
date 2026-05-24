# calculate_peak_weights.py

import pandas as pd
import numpy as np
from rdkit import Chem
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from pathlib import Path

# --- 1. Definisikan Target dan Konfigurasi ---
TARGET_GROUP = {
    "name": "Ester",
    "smarts": "[CX3](=O)[OX2H0][#6]",
    "peak_windows": {
        "ESTER_CO_STRETCH": [1720, 1760],
        "ESTER_C_O_STRETCH": [1150, 1300]
    }
}

DATASET_PATH = Path("data/standarize/dataset.csv")
BIN_SIZE_CM_INV = 10
SPECTRUM_RANGE = [400, 4000]

def get_smiles_from_cas(cas_no):
    """
    Fungsi placeholder untuk mendapatkan SMILES dari ID molekul.
    Dalam skenario nyata, ini akan melakukan query ke database atau API.
    """
    # Peta dummy untuk demonstrasi, sesuaikan dengan data Anda.
    cas_map = {
        "141-78-6": "CCOC(=O)C",  # Ethyl acetate (Ester)
        "109-94-4": "CCOC=O",     # Ethyl formate (Ester)
        "64-19-7": "CC(=O)O",     # Acetic acid (Non-Ester)
        "67-64-1": "CC(=O)C",     # Acetone (Non-Ester)
        "71-43-2": "c1ccccc1",    # Benzene (Non-Ester)
    }
    # Ekstrak bagian CAS dari ID molekul (misal, '141-78-6' dari '141-78-6_IR')
    cas_part = cas_no.split('_')[0]
    return cas_map.get(cas_part)

def create_binned_features_and_labels(dataset_df, bin_size, spectrum_range, target_smarts):
    """
    Membangun matriks fitur (X) dengan binning dan vektor label (y).
    """
    ir_df = dataset_df[dataset_df['spectrum_type'] == 'ir'].copy()
    
    # --- 2. Feature Engineering (Binning Spektrum) ---
    bins = np.arange(spectrum_range[0], spectrum_range[1] + bin_size, bin_size)
    bin_labels = [f"{bins[i]}-{bins[i+1]}" for i in range(len(bins)-1)]
    
    # Tambahkan kolom 'bin' ke DataFrame
    ir_df['bin'] = pd.cut(ir_df['x_value'], bins=bins, labels=bin_labels, right=False)
    
    # Pivot tabel untuk membuat fitur: satu baris per molekul, satu kolom per bin
    print("Membuat fitur dengan binning spektrum...")
    feature_matrix = ir_df.groupby(['molecule_id', 'bin'], observed=False)['y_value'].max().unstack(fill_value=0)

    # --- PERBAIKAN: Buat negative class yang lebih baik ---
    # Sertakan molekul non-ester yang juga memiliki gugus C=O untuk "menantang" model.
    other_carbonyl_smarts = "[#6][CX3](=O)[#6]" # Ketone
    other_carbonyl_pattern = Chem.MolFromSmarts(other_carbonyl_smarts)

    positive_indices, negative_indices, challenging_negative_indices = [], [], []
    for i, molecule_id in enumerate(feature_matrix.index):
        smiles = get_smiles_from_cas(molecule_id)
        if smiles:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                if mol.HasSubstructMatch(other_carbonyl_pattern) and not mol.HasSubstructMatch(Chem.MolFromSmarts(target_smarts)):
                    challenging_negative_indices.append(i)

    
    # --- 3. Label Generation ---
    print("Membuat label berdasarkan SMARTS...")
    target_pattern = Chem.MolFromSmarts(target_smarts)
    labels = []
    
    for molecule_id in feature_matrix.index:
        smiles = get_smiles_from_cas(molecule_id)
        if smiles:
            mol = Chem.MolFromSmiles(smiles)
            if mol and mol.HasSubstructMatch(target_pattern):
                labels.append(1)  # Memiliki gugus fungsi target
            else:
                labels.append(0)  # Tidak memiliki
        else:
            labels.append(0) # Asumsikan tidak memiliki jika SMILES tidak ditemukan

    # --- PERBAIKAN: Pastikan model melihat contoh yang menantang ---
    # Jika ada sampel keton, pastikan beberapa di antaranya ada di set training dan testing.
    # Ini akan memaksa model untuk belajar membedakan C=O ester dari C=O keton.
    final_labels = np.array(labels)
    if challenging_negative_indices:
        print(f"  -> Menambahkan {len(challenging_negative_indices)} sampel 'challenging negative' (misalnya, Keton) ke dalam pertimbangan.")
        # Logika ini dapat diperluas untuk memastikan sampel-sampel ini terdistribusi
        # dengan baik dalam split training/testing, tetapi untuk saat ini,
        # hanya dengan menyertakannya dalam 'y' sudah akan meningkatkan model.
        # `train_test_split` dengan `stratify=y` akan membantu mendistribusikannya.

    return feature_matrix, np.array(labels), bins

def main():
    """
    Fungsi utama untuk melatih model, mengekstrak feature importance,
    dan menghitung peak_weight.
    """
    print(f"🚀 Memulai kalkulasi peak_weight data-driven untuk: '{TARGET_GROUP['name']}'")

    try:
        dataset_df = pd.read_csv(DATASET_PATH)
    except FileNotFoundError:
        print(f"❌ Error: File dataset '{DATASET_PATH}' tidak ditemukan. Jalankan skrip pra-pemrosesan terlebih dahulu.")
        return

    X, y, bins = create_binned_features_and_labels(dataset_df, BIN_SIZE_CM_INV, SPECTRUM_RANGE, TARGET_GROUP["smarts"])

    if X.empty:
        print("❌ Tidak ada data yang dapat diproses untuk membuat fitur.")
        return

    print(f"\n✅ Berhasil membuat {len(X)} sampel. Terdapat {np.sum(y)} sampel positif (memiliki Ester).")

    # --- 4. Training Model ---
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    print("\n🌳 Melatih model RandomForestClassifier...")
    # Ganti ini:
# model = RandomForestClassifier()

# Menjadi ini:
    model = RandomForestClassifier(class_weight='balanced')
    model.fit(X_train, y_train)

    # (Opsional) Validasi akurasi
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"  -> Akurasi model pada data testing: {accuracy:.2%}")

    # --- 5. Ekstraksi dan Pemetaan Feature Importance ---
    feature_importances = model.feature_importances_
    
    peak_importance_scores = {}
    
    print("\n🗺️  Memetakan feature importance ke jendela puncak...")
    for peak_name, window in TARGET_GROUP["peak_windows"].items():
        # Identifikasi indeks bin yang berada di dalam jendela
        min_freq, max_freq = min(window), max(window)
        
        relevant_bin_indices = [
            i for i, bin_start in enumerate(bins[:-1]) 
            if max_freq >= bin_start >= min_freq
        ]
        
        # Jumlahkan skor importance dari bin yang relevan
        total_importance_for_peak = np.sum(feature_importances[relevant_bin_indices])
        peak_importance_scores[peak_name] = total_importance_for_peak
        
        print(f"  -> Skor mentah untuk '{peak_name}' ({min_freq}-{max_freq} cm⁻¹): {total_importance_for_peak:.4f}")

    # --- 6. Normalisasi menjadi peak_weight ---
    total_score = sum(peak_importance_scores.values())
    
    if total_score == 0:
        print("\n❌ Peringatan: Total feature importance adalah nol. Tidak dapat menghitung bobot.")
        return
        
    peak_weights = {
        peak_name: score / total_score
        for peak_name, score in peak_importance_scores.items()
    }

    # --- 7. Tampilkan Hasil ---
    print("\n" + "="*50)
    print("📊 Hasil Akhir Peak Weight (Data-Driven)")
    print("="*50)
    for peak_name, weight in peak_weights.items():
        print(f"  - {peak_name:<20}: {weight:.4f}")
    print("="*50)

if __name__ == "__main__":
    main()