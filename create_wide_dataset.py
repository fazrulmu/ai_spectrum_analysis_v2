import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from src.data.data_processing import parse_jdx, preprocess_spectrum

def create_wide_format_dataset(raw_data_dir, label_file_path, output_path):
    """
    Membuat dataset format lebar dari file JDX mentah dan file label.

    Proses:
    1. Mendapatkan daftar file JDX yang akan diproses.
    2. Mendapatkan label gugus fungsi dari `feature_extraction_dataset.csv`.
    3. Untuk setiap file JDX:
        a. Parse file untuk mendapatkan data spektrum mentah.
        b. Pra-pemrosesan spektrum (interpolasi ke grid standar 4000-400 cm⁻¹).
        c. Mengubah (pivot) spektrum menjadi format lebar.
        d. Menggabungkan dengan label gugus fungsi.
    4. Menyimpan hasil gabungan ke dalam satu file CSV.

    Args:
        raw_data_dir (str): Path ke direktori berisi file-file JDX.
        label_file_path (str): Path ke file CSV yang berisi file_id dan gugus_fungsi.
        output_path (str): Path untuk menyimpan file CSV hasil.
    """
    print("🚀 Memulai proses pembuatan dataset format lebar...")

    # --- 1. Dapatkan Label Gugus Fungsi ---
    print(f"📖 Membaca label gugus fungsi dari: {label_file_path}")
    try:
        label_df = pd.read_csv(label_file_path)
    except FileNotFoundError:
        print(f"❌ Error: File label tidak ditemukan di '{label_file_path}'.")
        print("   Pastikan Anda telah menjalankan 'generate_feature_extraction_dataset.py' terlebih dahulu.")
        return

    # Buat tabel pivot untuk multi-hot encoding label
    label_pivot = label_df.pivot_table(
        index='file_id', 
        columns='gugus_fungsi', 
        aggfunc='size', 
        fill_value=0
    ).astype(int).reset_index()
    
    all_functional_groups = [col for col in label_pivot.columns if col != 'file_id']
    print(f"✅ Ditemukan {len(all_functional_groups)} gugus fungsi unik.")

    # --- 2. Proses Setiap File JDX ---
    all_rows = []
    jdx_files = [f for f in os.listdir(raw_data_dir) if f.endswith(('.jdx', '.dx'))]
    
    # Definisikan grid bilangan gelombang standar
    wavenumber_grid = np.arange(4000, 399, -1)

    print(f"🔄 Memproses {len(jdx_files)} file spektrum mentah...")
    for filename in tqdm(jdx_files, desc="Memproses file JDX"):
        file_path = os.path.join(raw_data_dir, filename)
        file_id = os.path.splitext(filename)[0]

        # Hanya proses file yang ada di data label kita
        if file_id not in label_pivot['file_id'].values:
            continue

        try:
            # a. Parse dan pra-pemrosesan spektrum
            raw_spectrum = parse_jdx(file_path)
            if raw_spectrum is None:
                continue
            
            # b. Interpolasi ke grid standar
            processed_spectrum_df = preprocess_spectrum(raw_spectrum, wavenumber_grid)

            # c. Buat baris data untuk file ini
            row_data = {'file_id': file_id}
            
            # Tambahkan label gugus fungsi (one-hot encoded)
            labels = label_pivot[label_pivot['file_id'] == file_id]
            for group in all_functional_groups:
                row_data[group] = labels[group].iloc[0] if group in labels.columns else 0

            # d. Tambahkan nilai absorbansi dengan header bilangan gelombang
            # Ini adalah proses "pivot" atau "unstack"
            for _, spec_row in processed_spectrum_df.iterrows():
                wavenumber_col = int(spec_row['wavenumber'])
                absorbance_val = spec_row['absorbance']
                row_data[wavenumber_col] = absorbance_val
            
            all_rows.append(row_data)

        except Exception as e:
            print(f"⚠️ Gagal memproses file {filename}: {e}")

    # --- 3. Buat dan Simpan DataFrame Final ---
    print("📊 Membuat DataFrame final...")
    final_df = pd.DataFrame(all_rows)
    
    print(f"💾 Menyimpan dataset ke: {output_path}")
    final_df.to_csv(output_path, index=False)
    print(f"✅ Proses selesai. Dataset dengan {len(final_df)} sampel dan {len(final_df.columns)} kolom telah dibuat.")

if __name__ == '__main__':
    RAW_JDX_DIR = "data/raw/nist_jdx"
    LABEL_FILE = "data/for_train/feature_extraction_dataset.csv"
    OUTPUT_FILE = "data/for_train/wide_training_dataset.csv"
    create_wide_format_dataset(RAW_JDX_DIR, LABEL_FILE, OUTPUT_FILE)