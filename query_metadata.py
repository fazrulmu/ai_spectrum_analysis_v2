# /home/acer/ai_spectrum_analysis_v2/query_metadata.py

import pandas as pd
from pathlib import Path
import argparse
import re

def query_metadata_by_cas(cas_number: str):
    """
    Mencari dan menampilkan metadata spesifik dari file CSV
    berdasarkan Nomor Registri CAS.

    Args:
        cas_number (str): Nomor CAS yang ingin dicari.
    """
    # --- 1. Konfigurasi Path ---
    METADATA_CSV_PATH = Path("data/reports/metadata/extracted_metadata.csv")

    # --- 2. Validasi File ---
    if not METADATA_CSV_PATH.exists():
        print(f"❌ Error: File '{METADATA_CSV_PATH}' tidak ditemukan.")
        print("   Pastikan Anda sudah menjalankan skrip 'extract_metadata.py' terlebih dahulu.")
        return

    # --- 3. Muat Data CSV ---
    try:
        df = pd.read_csv(METADATA_CSV_PATH)
        # Pastikan kolom CAS number dibaca sebagai string untuk pencocokan yang akurat
        df['cas_registry_no'] = df['cas_registry_no'].astype(str)
    except Exception as e:
        print(f"❌ Error saat membaca file CSV: {e}")
        return

    # --- 4. Cari Data Berdasarkan CAS Number ---
    # Menggunakan .str.contains() untuk fleksibilitas jika ada spasi atau karakter lain
    results_df = df[df['cas_registry_no'].str.contains(cas_number, na=False)]

    # --- 5. Tampilkan Hasil ---
    if results_df.empty:
        print(f"\n🤷‍♂️ Tidak ada data yang ditemukan untuk CAS Registry No: '{cas_number}'")
    else:
        # Sesuai permintaan Anda "salah satu dari list cas yang identik",
        # kita akan ambil data dari baris pertama yang ditemukan.
        first_record = results_df.iloc[0]

        # Membersihkan rumus molekul dari spasi berlebih
        molform_raw = first_record.get('molform', 'N/A')
        molform_cleaned = ' '.join(str(molform_raw).split()) if pd.notna(molform_raw) else "N/A"

        print("\n" + "="*50)
        print(f"🔍 Hasil Pencarian untuk CAS No: {cas_number}")
        print("="*50)
        print(f"  - Nama Senyawa (Title)  : {first_record.get('title', 'N/A')}")
        print(f"  - Rumus Molekul (MolForm) : {molform_cleaned}")
        print(f"  - Nomor Registri CAS      : {first_record.get('cas_registry_no', 'N/A')}")
        print(f"  - Fasa (State)            : {first_record.get('state', 'N/A')}")
        print("-" * 50)
        print(f"  ℹ️  Ditemukan {len(results_df)} file spektrum yang cocok untuk CAS ini.")
        print(f"      Menampilkan detail dari file: '{first_record.get('id_file', 'N/A')}'")
        print("="*50)


def main():
    """
    Fungsi utama untuk menjalankan skrip dari command line.
    """
    # Membuat parser untuk argumen command-line
    parser = argparse.ArgumentParser(
        description="Ambil metadata spesifik untuk senyawa berdasarkan Nomor CAS dari file 'extracted_metadata.csv'.",
        epilog="Contoh penggunaan: python query_metadata.py 50-00-0"
    )
    # Menambahkan argumen 'cas' yang wajib diisi
    parser.add_argument(
        "cas_number",
        type=str,
        help="Nomor Registri CAS yang ingin dicari (contoh: 50-00-0)."
    )

    args = parser.parse_args()
    
    # Panggil fungsi query dengan CAS number yang diberikan
    query_metadata_by_cas(args.cas_number)


if __name__ == "__main__":
    main()