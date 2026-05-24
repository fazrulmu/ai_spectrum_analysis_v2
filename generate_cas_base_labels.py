"""
generate_cas_base_labels.py
Versi sederhana — labeling spektrum IR berbasis aturan biner (0/1) berdasarkan deteksi puncak.
"""

import pandas as pd
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
import re
from typing import Optional
from scipy.signal import find_peaks
from rdkit import Chem
from rdkit import RDLogger

# Menonaktifkan logging error RDKit yang terlalu verbose di konsol
RDLogger.DisableLog('rdApp.*')

# === Path konfigurasi ===
INDEX_FILE_PATH = Path("data/standarize/dataset_index.csv")
RULES_FILE_PATH = Path("data/standarize/data_labels.csv")
UNIQUE_CAS_FILE_PATH = Path("data/reports/unique_cas_numbers.csv")
STRUCTURAL_CONFIDENCE_PATH = Path("data/reports/structural_confidence.json") # BARU: Input utama untuk aturan
OUTPUT_DIR = Path("data/reports/generated_labels") # Direktori output bisa disesuaikan
OUTPUT_FILE_PATH = OUTPUT_DIR / "cas_base_labels_no_confidence.csv" # Nama file bisa disesuaikan

# === Fungsi utama: deteksi puncak ===
def detect_groups_with_structural_guidance(spectrum_df, structural_groups, all_possible_groups, sample_state):
    """
    Mendeteksi puncak spektral (spectral peaks) berdasarkan panduan dari gugus fungsi yang secara struktural mungkin ada.

    Args:
        spectrum_df (pd.DataFrame): DataFrame spektrum IR (kolom 'x_value', 'y_value').
        structural_groups (dict): Kamus gugus fungsi yang mungkin ada di molekul ini, diambil dari structural_confidence.json.
        all_possible_groups (set): Set dari semua nama gugus fungsi yang mungkin ada untuk memastikan kolom output konsisten.
        sample_state (str): Fasa sampel (misal: 'gas', 'liquid') untuk penyesuaian rentang.

    Returns:
        dict: Kamus label biner (0 atau 1) untuk setiap gugus fungsi.
    """

    # Normalisasi sinyal
    y_norm = (spectrum_df['y_value'] - spectrum_df['y_value'].min()) / (
        spectrum_df['y_value'].max() - spectrum_df['y_value'].min()
    )

    # Deteksi semua puncak lokal yang mungkin. Prominence rendah membuat deteksi lebih sensitif.
    # Puncak yang tidak relevan akan difilter berdasarkan rentang dari aturan.
    peaks, _ = find_peaks(y_norm)
    peaks_df = spectrum_df.iloc[peaks][['x_value']].copy()

    labels = {}

    # Inisialisasi semua label dengan 0
    for group_name in all_possible_groups:
        labels[group_name] = 0

    # Iterasi hanya pada gugus yang secara struktural mungkin ada
    for group_name, group_info in structural_groups.items():
        range_min = float(group_info["range_min"])
        range_max = float(group_info["range_max"])

        # Penyesuaian rentang berdasarkan fasa (state) untuk akurasi yang lebih baik
        if sample_state == "gas":
            # Puncak pada fasa gas cenderung bergeser ke frekuensi lebih tinggi (blue-shift).
            # Pergeseran +25 cm⁻¹ adalah heuristik yang masuk akal untuk memulai.
            range_min += 5
            range_max += 25

        # Periksa apakah ada puncak spektral di rentang yang ditentukan
        peak_found = not peaks_df[
            (peaks_df["x_value"] >= range_min) & (peaks_df["x_value"] <= range_max)
        ].empty

        # Jika ada puncak yang ditemukan dalam rentang yang ditentukan, set labelnya menjadi 1.
        if peak_found:
            labels[group_name] = 1

    return labels


def extract_cas_from_filename(filename_stem: str) -> Optional[str]:
    """
    Mengekstrak Nomor CAS dari nama file menggunakan regular expression.
    Mencari pola angka yang dipisahkan oleh tanda hubung (contoh: 123-45-6).
    """
    # Pola regex untuk menemukan Nomor CAS: beberapa digit, hubung, beberapa digit, hubung, satu digit.
    match = re.search(r'\d{2,7}-\d{2,2}-\d{1,1}', filename_stem)
    if match:
        return match.group(0)
    return None


# === Fungsi utama ===
def main():
    print("🚀 Membuat file label IR berbasis analisis spektral dan struktural...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Muat file dasar
    try:
        index_df = pd.read_csv(INDEX_FILE_PATH)
        rules_df = pd.read_csv(RULES_FILE_PATH)
        unique_cas_df = pd.read_csv(UNIQUE_CAS_FILE_PATH)
        structural_data = json.load(open(STRUCTURAL_CONFIDENCE_PATH))
    except FileNotFoundError as e:
        if "index.csv" in str(e):
            print(f"❌ Error: File indeks tidak ditemukan di '{INDEX_FILE_PATH}'.")
            print("   Pastikan Anda telah menjalankan 'preprocess_universal.py' terlebih dahulu.")
        elif "unique_cas_numbers.csv" in str(e):
            print(f"❌ Error: File '{UNIQUE_CAS_FILE_PATH}' tidak ditemukan.")
            print("   Pastikan Anda telah menjalankan 'list_unique_cas.py'.")
        elif "structural_confidence.json" in str(e):
            print(f"❌ Error: File '{STRUCTURAL_CONFIDENCE_PATH}' tidak ditemukan.")
            print("   Pastikan Anda telah menjalankan 'generate_structural_confidence.py' terlebih dahulu.")
        else:
            print(f"❌ Error: File tidak ditemukan: {e}")
        return

    # 2. Filter index_df berdasarkan CAS number yang diizinkan
    allowed_cas_numbers = set(unique_cas_df['cas_registry_no'].astype(str))
    initial_count = len(index_df)
    index_df = index_df[index_df['molecule_id'].astype(str).isin(allowed_cas_numbers)]
    filtered_count = len(index_df)
    print(f"🔍 Ditemukan {initial_count} spektrum, difilter menjadi {filtered_count} spektrum berdasarkan {len(allowed_cas_numbers)} CAS number unik.")

    # Hapus duplikat berdasarkan nomor CAS, hanya ambil entri pertama untuk setiap CAS.
    # Ini memastikan setiap molekul hanya dianalisis satu kali.
    index_df.drop_duplicates(subset='molecule_id', keep='first', inplace=True)
    final_count = len(index_df)
    print(f"✨ Setelah menghapus duplikat CAS, {final_count} spektrum unik akan diproses.")

    # Dapatkan daftar lengkap semua kemungkinan nama gugus fungsi untuk nama kolom
    all_possible_groups = set(rules_df['name'].unique())

    # 3. Persiapan dataframe output
    results = []
    if "processed_id" in index_df.columns:
        index_df.rename(columns={"processed_id": "file_id"}, inplace=True)

    print(f"📊 Akan memproses {len(index_df)} file spektrum...")

    # 4. Iterasi tiap spektrum
    for _, row in tqdm(index_df.iterrows(), total=len(index_df), desc="Menganalisis spektrum"):        
        file_id = Path(row["processed_filepath"]).stem # ID unik berdasarkan nama file
        sample_state = str(row.get('state', 'unknown')).lower() # Ambil state dari index

        # Ekstrak Nomor CAS dari nama file untuk dicocokkan dengan data struktural
        cas_number = extract_cas_from_filename(file_id)
        if not cas_number:
            tqdm.write(f"⚠️ Peringatan: Tidak dapat mengekstrak Nomor CAS dari '{file_id}'. Melewati file ini.")
            continue

        try:
            spectrum_df = pd.read_csv(row["processed_filepath"])
        except (FileNotFoundError, pd.errors.EmptyDataError) as e:
            tqdm.write(f"⚠️ Peringatan: Gagal membaca file spektrum '{row['processed_filepath']}'. Error: {e}")
            continue
        
        # Ambil data gugus fungsi yang mungkin ada dari structural_confidence.json
        molecule_structural_data = structural_data.get(cas_number)
        if not molecule_structural_data or "detected_functional_groups" not in molecule_structural_data:
            tqdm.write(f"ℹ️ Info: Tidak ada data struktural untuk CAS '{cas_number}'. Melewati.")
            continue # Lewati jika tidak ada data struktural untuk CAS ini

        structural_groups = molecule_structural_data["detected_functional_groups"]

        # Pastikan IR
        if 'spectrum_type' in spectrum_df.columns and spectrum_df['spectrum_type'].iloc[0] != 'ir':
            continue

        final_labels = {"file_id": file_id}

        # Lakukan deteksi puncak berdasarkan panduan struktural
        detected = detect_groups_with_structural_guidance(spectrum_df, structural_groups, all_possible_groups, sample_state)
        final_labels.update(detected)

        results.append(final_labels)

    # 5. Simpan hasil
    if not results:
        print("❌ Tidak ada data valid untuk labeling.")
        return

    final_df = pd.DataFrame(results)

    # Urutkan kolom
    ordered_cols = ['file_id']
    group_cols = [col for col in final_df.columns if col != 'file_id']
    ordered_cols.extend(sorted(group_cols))
    final_df = final_df[ordered_cols]

    # Konversi kolom label menjadi integer (0/1) untuk kejelasan
    # Iterasi semua kolom kecuali 'file_id' dan ubah tipenya menjadi integer.
    for col in group_cols:
        if col in final_df.columns:
            final_df[col] = final_df[col].astype(int)

    final_df.to_csv(OUTPUT_FILE_PATH, index=False)
    print(f"\n✅ Labeling selesai! File tersimpan di: {OUTPUT_FILE_PATH}")
    print(f"📈 Total spektrum: {len(final_df)}")

if __name__ == "__main__":
    main()
