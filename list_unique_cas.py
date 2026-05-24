# /home/acer/ai_spectrum_analysis_v2/list_unique_cas.py

import pandas as pd
from pathlib import Path

def list_all_cas_numbers():
    """
    Membaca file metadata CSV, mengekstrak semua Nomor CAS unik, dan
    menyimpannya ke dalam file CSV baru.
    """
    # --- 1. Konfigurasi Path ---
    METADATA_CSV_PATH = Path("data/reports/metadata/extracted_metadata.csv")
    OUTPUT_DIR = Path("data/reports")
    OUTPUT_CSV_PATH = OUTPUT_DIR / "unique_cas_numbers.csv"
    
    # --- 2. Validasi dan Muat File ---
    if not METADATA_CSV_PATH.exists():
        print(f"❌ Error: File '{METADATA_CSV_PATH}' tidak ditemukan.")
        print("   Pastikan Anda sudah menjalankan skrip 'extract_metadata.py' terlebih dahulu.")
        return

    try:
        df = pd.read_csv(METADATA_CSV_PATH, dtype={'cas_registry_no': str})
    except Exception as e:
        print(f"❌ Error saat membaca file CSV: {e}")
        return

    # --- 3. Ekstrak dan Bersihkan Nomor CAS ---
    if 'cas_registry_no' not in df.columns:
        print(f"❌ Error: Kolom 'cas_registry_no' tidak ditemukan di dalam file CSV.")
        return

    # Ambil nilai unik, hapus nilai NaN/kosong, dan urutkan
    unique_cas_numbers = sorted(df['cas_registry_no'].dropna().unique())

    # --- 4. Simpan ke CSV dan Tampilkan Hasil ---
    if not unique_cas_numbers:
        print("\n🤷‍♂️ Tidak ada Nomor CAS yang valid ditemukan dalam file.")
    else:
        # Buat DataFrame dari daftar unik
        unique_cas_df = pd.DataFrame(unique_cas_numbers, columns=['cas_registry_no'])
        
        # Pastikan direktori output ada
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        
        # Simpan ke file CSV
        unique_cas_df.to_csv(OUTPUT_CSV_PATH, index=False)
        
        print("\n" + "="*50)
        print("🎉 Ekstraksi Nomor CAS unik selesai!")
        print(f"💾 File CSV disimpan di: '{OUTPUT_CSV_PATH}'")
        print(f"📊 Total {len(unique_cas_df)} Nomor CAS unik ditemukan.")
        print("="*50)

if __name__ == "__main__":
    list_all_cas_numbers()