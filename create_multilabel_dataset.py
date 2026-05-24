import pandas as pd
import ast
from tqdm import tqdm

def create_multilabel_dataset(input_path, output_path):
    """
    Mengubah dataset ekstraksi fitur menjadi format multi-label.

    Fungsi ini membaca dataset yang setiap barisnya adalah segmen gugus fungsi,
    dan mengubahnya menjadi format di mana setiap baris adalah satu spektrum unik
    dengan kolom multi-hot encoded untuk setiap gugus fungsi yang ada.

    Args:
        input_path (str): Path ke file feature_extraction_dataset.csv.
        output_path (str): Path untuk menyimpan file multilabel_training_dataset.csv.
    """
    print(f"📖 Membaca dataset dari: {input_path}")
    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        print(f"❌ Error: File tidak ditemukan di '{input_path}'.")
        print("   Pastikan Anda telah menjalankan 'generate_feature_extraction_dataset.py' terlebih dahulu.")
        return

    # Konversi kolom string-list menjadi list aktual
    print("🔄 Mengonversi kolom spektrum...")
    df['spectrum_values'] = df['spectrum_values'].apply(ast.literal_eval)
    df['wavenumber_values'] = df['wavenumber_values'].apply(ast.literal_eval)

    # --- Proses Penggabungan dan Transformasi ---
    print("🔄 Menggabungkan data berdasarkan file_id...")
    
    # 1. Gabungkan data spektrum dan bilangan gelombang untuk setiap file_id
    agg_funcs = {
        'spectrum_values': lambda x: sum(x, []),
        'wavenumber_values': lambda x: sum(x, []),
        'gugus_fungsi': lambda x: list(set(x)) # Ambil daftar gugus fungsi unik per file
    }
    df_grouped = df.groupby('file_id').agg(agg_funcs).reset_index()

    # 2. Buat kolom multi-hot encoding (dummy variables) dari gugus_fungsi
    print("🔄 Membuat kolom label (multi-hot encoding)...")
    dummies = pd.get_dummies(df_grouped['gugus_fungsi'].apply(pd.Series).stack()).groupby(level=0).sum()
    
    # 3. Gabungkan DataFrame asli dengan kolom dummy
    df_final = pd.concat([df_grouped.drop('gugus_fungsi', axis=1), dummies], axis=1)

    # Pastikan semua kolom gugus fungsi ada, isi dengan 0 jika tidak ada di beberapa sampel
    all_functional_groups = df['gugus_fungsi'].unique()
    for group in all_functional_groups:
        if group not in df_final.columns:
            df_final[group] = 0

    # --- Penyimpanan Hasil ---
    print(f"💾 Menyimpan dataset multi-label ke: {output_path}")
    df_final.to_csv(output_path, index=False)
    print("✅ Proses selesai.")
    print(f"Total {len(df_final)} spektrum unik dengan {len(all_functional_groups)} kolom gugus fungsi.")

if __name__ == '__main__':
    INPUT_DATASET = "data/for_train/feature_extraction_dataset.csv"
    OUTPUT_DATASET = "data/for_train/multilabel_training_dataset.csv"
    create_multilabel_dataset(INPUT_DATASET, OUTPUT_DATASET)