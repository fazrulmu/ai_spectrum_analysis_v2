# build_training_dataset.py

import pandas as pd
import numpy as np
from rdkit import Chem
from tqdm import tqdm
from pathlib import Path
import pubchempy as pcp # Tambahkan impor ini
import re                # Tambahkan impor ini
import time              # Tambahkan impor ini
import json              # Tambahkan impor ini

# --- KONFIGURASI ---
DATASET_PATH = Path("data/standarize/dataset.csv") # PERUBAHAN: Path ke dataset "long"
LABELS_PATH = Path("data/standarize/labels.csv")
OUTPUT_PATH = Path("training_dataset_binned.csv")

BIN_SIZE_CM_INV = 20
SPECTRUM_RANGE = [600, 4000]

# Definisikan SMARTS untuk auto-labeling
LABEL_DEFINITIONS = {
    "label_ester": "[CX3](=O)[OX2H0][#6]",
    "label_acid": "C(=O)[O;H1]",
    "label_alcohol": "[#6][OX2H1]",
    "label_ketone": "[#6][CX3](=O)[#6]",
    "label_alkene": "[#6]=[#6]"
}

# --- PERBAIKAN: Fungsi yang di-upgrade untuk mengambil SMILES dari PubChem ---
SMILES_CACHE_PATH = Path("data/standarize/smiles_cache.json")

def load_smiles_cache():
    """Memuat cache SMILES dari file JSON."""
    if SMILES_CACHE_PATH.exists():
        with open(SMILES_CACHE_PATH, 'r') as f:
            return json.load(f)
    return {}

def save_smiles_cache(cache):
    """Menyimpan cache SMILES ke file JSON."""
    with open(SMILES_CACHE_PATH, 'w') as f:
        json.dump(cache, f, indent=2)

def get_smiles_from_cas(cas_no, cache):
    """
    Mendapatkan SMILES dari nomor CAS menggunakan PubChem, dengan sistem caching.
    """
    # --- PERBAIKAN: Gunakan regex untuk menemukan Nomor CAS yang valid ---
    # Pola ini mencari format seperti XXX-XX-X, XX-XX-X, dll.
    # PERBAIKAN: Regex diubah untuk menangani kasus di mana nomor CAS bersebelahan dengan '_'
    match = re.search(r'(\d{2,7}-\d{2}-\d)', cas_no)
    
    if not match:
        # Jika tidak ada pola CAS yang ditemukan, simpan kegagalan dan kembalikan None
        cache[cas_no] = None 
        return None

    cas_number = match.group(1)
    if cas_number in cache:
        return cache[cas_number]

    try:
        # PERBAIKAN: Hapus 'name' agar pubchempy bisa mencari berdasarkan CAS number
        compounds = pcp.get_compounds(cas_number)
        if compounds:
            smiles = compounds[0].canonical_smiles
            cache[cas_number] = smiles
            time.sleep(0.2) # Jeda untuk menghormati API PubChem
            return smiles
    except Exception:
        cache[cas_number] = None # Simpan kegagalan agar tidak diulang
        return None
    
    cache[cas_number] = None # Simpan jika tidak ditemukan
    return None

def generate_labels_file(dataset_df, output_path):
    """Membuat file labels.csv berdasarkan SMARTS."""
    print(f"🔍 File label tidak ditemukan. Membuat file baru di '{output_path}'...")
    label_data = []
    
    patterns = {name: Chem.MolFromSmarts(smarts) for name, smarts in LABEL_DEFINITIONS.items()}

    # Muat cache SMILES yang sudah ada
    smiles_cache = load_smiles_cache()

    # PERUBAHAN: Hanya iterasi pada ID molekul yang unik untuk efisiensi
    unique_molecule_ids = dataset_df['molecule_id'].unique()

    for molecule_id in tqdm(unique_molecule_ids, desc="Generating Labels"):
        smiles = get_smiles_from_cas(molecule_id, smiles_cache)
        
        labels = {"molecule_id": molecule_id}
        if smiles:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                for name, pattern in patterns.items():
                    labels[name] = 1 if mol.HasSubstructMatch(pattern) else 0
            else: # SMILES tidak valid
                for name in patterns: labels[name] = 0
        else: # SMILES tidak ditemukan
            for name in patterns: labels[name] = 0
        
        label_data.append(labels)
        
    labels_df = pd.DataFrame(label_data)
    labels_df.to_csv(output_path, index=False)
    # Simpan cache yang sudah diperbarui
    save_smiles_cache(smiles_cache)
    print(f"💾 Cache SMILES diperbarui dan disimpan di '{SMILES_CACHE_PATH}'.")
    print("✅ File label berhasil dibuat.")
    return labels_df

def bin_all_spectra(ir_df, bins, bin_labels):
    """
    Menerapkan binning pada seluruh dataset IR menggunakan pivot_table yang efisien.
    """
    # Terapkan binning
    ir_df['bin'] = pd.cut(ir_df['x_value'], bins=bins, labels=bin_labels, right=False)
    
    # Pivot tabel untuk membuat fitur: satu baris per molekul, satu kolom per bin
    print("Membuat fitur dengan binning spektrum (menggunakan pivot)...")
    feature_matrix = ir_df.groupby(['molecule_id', 'bin'], observed=False)['y_value'].max().unstack(fill_value=0)
    
    # Reset index untuk menjadikan 'molecule_id' sebagai kolom biasa
    feature_matrix.reset_index(inplace=True)
    return feature_matrix

def main():
    """
    Fungsi utama untuk membangun dataset training "lebar" yang sudah di-binning.
    """
    print("🚀 Memulai pembuatan dataset training 'lebar'...")

    # --- 1. Definisi Binning ---
    # Buat bins dalam urutan menaik, ini wajib untuk pd.cut
    bins = np.arange(SPECTRUM_RANGE[0], SPECTRUM_RANGE[1] + BIN_SIZE_CM_INV, BIN_SIZE_CM_INV)
    # PERBAIKAN: Buat label dalam urutan MENURUN agar nama kolomnya intuitif
    bin_labels = [f"bin_{bins[i]}_{bins[i+1]}" for i in range(len(bins)-1)]

    # --- 2. Muat atau Buat File Label ---
    try:
        dataset_df = pd.read_csv(DATASET_PATH)
    except FileNotFoundError:
        print(f"❌ Error: File dataset '{DATASET_PATH}' tidak ditemukan. Jalankan 'preprocess_universal.py' terlebih dahulu.")
        return

    try:
        labels_df = pd.read_csv(LABELS_PATH)
        if labels_df.empty:
            raise FileNotFoundError # Force regeneration if file is empty
    except (FileNotFoundError, pd.errors.EmptyDataError):
        labels_df = generate_labels_file(dataset_df, LABELS_PATH)

    # --- 3. Alur Kerja Utama (Looping) ---
    if labels_df.empty:
        print("❌ Gagal membuat atau memuat label. Periksa fungsi `get_smiles_from_cas` Anda.")
        return

    # a. Filter hanya data IR
    ir_df = dataset_df[dataset_df['spectrum_type'] == 'ir'].copy()
    if ir_df.empty:
        print("❌ Tidak ada data spektrum IR yang ditemukan di dalam dataset.")
        return
    
    # b. Lakukan binning pada semua data IR sekaligus
    binned_df = bin_all_spectra(ir_df, bins, bin_labels)

    # c. Gabungkan data binned dengan label
    final_df = pd.merge(labels_df, binned_df, on='molecule_id', how='inner')

    # --- PERBAIKAN: Urutkan kolom agar sesuai format yang diinginkan ---
    # 1. Pisahkan kolom ID dan label dari kolom bin
    id_and_label_cols = [col for col in final_df.columns if not col.startswith('bin_')]
    bin_cols = [col for col in final_df.columns if col.startswith('bin_')]
    # 2. Gabungkan kembali dengan kolom bin yang sudah diurutkan secara numerik dari tertinggi ke terendah
    final_df = final_df[id_and_label_cols + sorted(bin_cols, reverse=True)]

    # --- 4. Output Final ---
    if final_df.empty:
        print("\n❌ Tidak ada data yang berhasil diproses. File training tidak dibuat.")
        return

    final_df.to_csv(OUTPUT_PATH, index=False)

    print(f"\n🎉 Dataset training 'lebar' berhasil dibuat!")
    print(f"   -> Disimpan di: '{OUTPUT_PATH}'")
    print(f"   -> Dimensi dataset: {final_df.shape[0]} baris x {final_df.shape[1]} kolom")

if __name__ == "__main__":
    main()
