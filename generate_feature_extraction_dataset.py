"""
generate_feature_extraction_dataset.py
Versi 2.0: Menggunakan data spektrum mentah (.jdx) secara langsung.

Skrip ini membuat dataset untuk analisis fitur spektral.
Ini menggabungkan informasi struktural dari 'structural_confidence.json'
dengan data spektrum mentah untuk mengekstrak segmen spektrum yang relevan
untuk setiap gugus fungsi yang terdeteksi.

Outputnya adalah file CSV dengan format:
| file_id | gugus_fungsi | wavenumber_min | wavenumber_max | spectrum_values (array) |
"""

import pandas as pd
import json
import yaml
from pathlib import Path
from tqdm import tqdm
import sys

# --- Pengaturan Path untuk Impor Modul Proyek ---
sys.path.insert(0, str(Path(__file__).resolve().parent))
# PERUBAHAN: Impor fungsi yang lebih spesifik dan modular
from src.data.data_processing import parse_jdx, preprocess_spectrum

def get_spectrum_type_from_metadata_local(metadata: dict) -> str:
    """
    Fungsi lokal untuk mendeteksi tipe spektrum dari metadata JCAMP.
    Tidak bergantung pada modul eksternal.
    """
    data_type = metadata.get('data type', '').lower()
    if 'infrared' in data_type:
        return 'ir'
    elif 'uv/vis' in data_type or 'uv-vis' in data_type:
        return 'uv'
    return 'unknown'
def create_feature_extraction_dataset():
    """
    Menghasilkan dataset untuk ekstraksi fitur berdasarkan kepercayaan struktural dan data spektral.
    """
    # --- 1. Konfigurasi Path ---
    STRUCTURAL_CONFIDENCE_PATH = Path("data/reports/structural_confidence.json")
    RAW_SPECTRUM_DIR = Path("data/raw/nist_jdx") # BARU: Menggunakan data mentah
    CONFIG_PATH = Path("main_config.yaml")
    OUTPUT_DIR = Path("data/for_train")
    OUTPUT_FILE_PATH = OUTPUT_DIR / "feature_extraction_dataset.csv"

    print("🚀 Memulai pembuatan dataset ekstraksi fitur...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- 2. Muat File Input & Konfigurasi ---
    try:
        with open(STRUCTURAL_CONFIDENCE_PATH, 'r') as f:
            structural_data = json.load(f)
        with open(CONFIG_PATH, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError as e:
        print(f"❌ Error: File yang dibutuhkan tidak ditemukan. Pastikan semua file input ada. Detail: {e}")
        return
    except yaml.YAMLError as e:
        print(f"❌ Error saat memuat file konfigurasi YAML: {e}")
        return

    # --- 3. Proses Data ---
    output_data = []

    # Iterasi melalui setiap molekul dalam data kepercayaan struktural
    for cas_number, data in tqdm(structural_data.items(), desc="Processing molecules"):
        # Periksa apakah ada data spektrum untuk CAS number ini
        cas_dir = RAW_SPECTRUM_DIR / cas_number
        if not cas_dir.is_dir():
            continue

        # PERBAIKAN: Iterasi melalui SEMUA file .jdx di dalam direktori CAS
        # Ini memastikan semua spektrum (misal: fasa berbeda) untuk satu senyawa diproses.
        for jdx_file_path in cas_dir.glob('*.jdx'):
            # --- Alur kerja pemrosesan baru untuk setiap file ---
            # Langkah A: Parse file JDX mentah menggunakan fungsi dari data_processing
            try:
                raw_data = parse_jdx(str(jdx_file_path))
                if not raw_data or 'x' not in raw_data or len(raw_data['x']) == 0:
                    continue
                
                # Langkah B: Deteksi tipe spektrum (IR/UV) dari metadata
                spectrum_type = get_spectrum_type_from_metadata_local(raw_data['metadata'])
                if spectrum_type != 'ir':
                    continue # Hanya proses spektrum IR

                # Langkah C: Lakukan pra-pemrosesan untuk mendapatkan DataFrame standar
                processed_df = preprocess_spectrum(raw_data, config, 'ir', normalize=True)
                if processed_df is None or processed_df.empty:
                    continue

            except Exception as e:
                tqdm.write(f"⚠️ Peringatan: Gagal memproses file spektrum '{jdx_file_path.name}'. Error: {e}")
                continue

            # Iterasi melalui setiap gugus fungsi yang terdeteksi untuk molekul ini
            for group_name, group_info in data.get("detected_functional_groups", {}).items():
                wavenumber_min = group_info.get("range_min")
                wavenumber_max = group_info.get("range_max")

                if wavenumber_min is None or wavenumber_max is None:
                    continue

                # Pastikan rentang benar (min < max)
                min_wn, max_wn = min(wavenumber_min, wavenumber_max), max(wavenumber_min, wavenumber_max)

                # Ekstrak nilai absorbansi dalam rentang yang ditentukan dari DataFrame yang sudah diproses
                segment_df = processed_df[
                    (processed_df['wavenumber'] >= min_wn) & (processed_df['wavenumber'] <= max_wn)
                ]
                # PERUBAHAN: Ekstrak nilai absorbansi (y) dan bilangan gelombang (x)
                absorbance_values = segment_df['absorbance'].tolist()
                wavenumber_values = segment_df['wavenumber'].tolist()
                
                # PERBAIKAN: Hanya tambahkan data jika segmen spektrum tidak kosong
                # DAN tidak semua nilainya adalah nol. Ini untuk menghindari data yang tidak informatif.
                if absorbance_values and not all(v == 0 for v in absorbance_values):
                    output_data.append({
                        "file_id": jdx_file_path.stem,
                        "gugus_fungsi": group_name,
                        "wavenumber_min": min_wn,
                        "wavenumber_max": max_wn,
                        "spectrum_values": absorbance_values,
                        "wavenumber_values": wavenumber_values # BARU: Simpan nilai x
                    })

    # --- 4. Simpan Hasil ---
    final_df = pd.DataFrame(output_data)
    final_df.to_csv(OUTPUT_FILE_PATH, index=False)
    print(f"\n✅ Dataset ekstraksi fitur berhasil dibuat dan disimpan di: '{OUTPUT_FILE_PATH}'")
    print(f"📊 Total baris yang dihasilkan: {len(final_df)}")

if __name__ == "__main__":
    create_feature_extraction_dataset()