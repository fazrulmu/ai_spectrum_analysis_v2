# create_training_dataset.py

import pandas as pd
from pathlib import Path
from tqdm import tqdm

def create_training_dataset():
    """
    Skrip utama untuk mengubah data spektrum dari format "panjang" menjadi "lebar",
    menggabungkannya dengan label, dan menyiapkannya untuk training machine learning.
    """
    # --- 1. KONFIGURASI PATH ---
    # Menggunakan pathlib untuk manajemen path yang lebih baik di berbagai OS
    PROCESSED_SPECTRUM_DIR = Path("data/standarize/processed_spectrum/")
    LABELS_FILE_PATH = Path("data/for_train/labels.csv")
    OUTPUT_DIR = Path("data/for_train/")
    OUTPUT_FILE_PATH = OUTPUT_DIR / "training_dataset.csv"

    print("🚀 Memulai pembuatan dataset training...")

    # Pastikan direktori output ada
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- 2. MEMUAT SEMUA DATA SPEKTRUM (FORMAT PANJANG) ---
    spectrum_files = list(PROCESSED_SPECTRUM_DIR.glob("*.csv"))
    if not spectrum_files:
        print(f"❌ Error: Tidak ada file spektrum yang ditemukan di '{PROCESSED_SPECTRUM_DIR}'.")
        print("Pastikan Anda telah menjalankan skrip 'preprocess_universal.py' terlebih dahulu.")
        return

    print(f"📂 Menemukan {len(spectrum_files)} file spektrum yang telah diproses.")

    all_spectra_df_list = []
    for file in tqdm(spectrum_files, desc="Membaca file spektrum"):
        try:
            df = pd.read_csv(file)
            all_spectra_df_list.append(df)
        except pd.errors.EmptyDataError:
            print(f"⚠️ Peringatan: Melewati file kosong: {file.name}")
            continue

    if not all_spectra_df_list:
        print("❌ Error: Tidak ada data spektrum yang valid untuk diproses.")
        return

    long_df = pd.concat(all_spectra_df_list, ignore_index=True)

    # Filter hanya untuk spektrum inframerah (IR)
    ir_df = long_df[long_df['spectrum_type'] == 'ir'].copy()
    if ir_df.empty:
        print("❌ Error: Tidak ada data spektrum tipe 'ir' yang ditemukan dalam file.")
        return
        
    print(f"🔬 Menyaring data... Menggunakan {ir_df['molecule_id'].nunique()} spektrum IR unik.")

    # --- 3. BINNING & TRANSFORMASI (PIVOTING) ---
    print("\n📊 Melakukan binning pada 'x_value' untuk mengurangi dimensi...")
    # Buat kolom baru dengan membulatkan 'x_value'. Ini adalah kunci untuk efisiensi memori.
    # Ini secara efektif mengelompokkan nilai-nilai seperti 3902.94 dan 3902.01 ke dalam bin '3902'.
    ir_df['x_bin'] = ir_df['x_value'].round().astype(int)

    print("\n🔄 Melakukan pivot pada data spektrum untuk mengubah format menjadi lebar...")
    
    # Pivot menggunakan kolom 'x_bin' yang baru.
    # Gunakan aggfunc='max' untuk mengambil nilai absorbansi tertinggi jika beberapa titik jatuh ke bin yang sama.
    wide_df = ir_df.pivot_table(
        index='molecule_id',
        columns='x_bin',
        values='y_value',
        aggfunc='max', # Mengambil nilai maksimum jika ada duplikat dalam satu bin
        fill_value=0
    )
    
    print("✅ Pivoting selesai.")

    # --- 4. MEMBERSIHKAN NAMA KOLOM ---
    # Mengubah nama kolom dari angka (misal: 3902) menjadi string yang valid (misal: bin_3902)
    new_columns = {col: f"bin_{round(col)}" for col in wide_df.columns}
    wide_df.rename(columns=new_columns, inplace=True)

    # Reset index agar 'molecule_id' menjadi kolom biasa untuk proses merge
    wide_df.reset_index(inplace=True)
    print("🎨 Nama kolom fitur telah diubah (misal: 'bin_4000').")

    # --- 5. MEMUAT DAN MENGGABUNGKAN DENGAN LABEL ---
    print(f"\n🏷️  Memuat file label dari '{LABELS_FILE_PATH}'...")
    try:
        labels_df = pd.read_csv(LABELS_FILE_PATH)
        # Ganti nama kolom 'file_id' menjadi 'molecule_id' agar konsisten untuk merge
        labels_df.rename(columns={'file_id': 'molecule_id'}, inplace=True)
    except FileNotFoundError:
        print(f"❌ Error: File label '{LABELS_FILE_PATH}' tidak ditemukan.")
        print("Pastikan Anda telah menjalankan skrip 'generate_labels.py' terlebih dahulu.")
        return

    print("🤝 Menggabungkan data fitur (spektrum) dengan data target (label)...")
    # Menggunakan 'inner' merge untuk memastikan hanya sampel yang ada di kedua file yang digunakan
    training_df = pd.merge(labels_df, wide_df, on='molecule_id', how='inner')

    # --- 6. FINALISASI DAN PENYIMPANAN ---
    # Mengurutkan kolom: ID, lalu semua label, lalu semua fitur bin
    label_cols = [col for col in labels_df.columns if col != 'molecule_id']
    feature_cols = [col for col in wide_df.columns if col != 'molecule_id']
    # Urutkan fitur spektrum dari bilangan gelombang tertinggi ke terendah
    feature_cols_sorted = sorted(feature_cols, key=lambda x: int(x.split('_')[1]), reverse=True)
    
    final_columns_order = ['molecule_id'] + label_cols + feature_cols_sorted
    training_df = training_df[final_columns_order]

    training_df.to_csv(OUTPUT_FILE_PATH, index=False)

    print("\n" + "="*50)
    print("🎉 Dataset training berhasil dibuat!")
    print(f"💾 File disimpan di: '{OUTPUT_FILE_PATH}'")
    print(f"📊 Dimensi dataset final: {training_df.shape[0]} baris (sampel) x {training_df.shape[1]} kolom (fitur + label)")
    print("="*50)

if __name__ == "__main__":
    create_training_dataset()