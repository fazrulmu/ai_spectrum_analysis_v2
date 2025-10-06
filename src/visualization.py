# src/visualization.py
# src/visualization.py

import matplotlib.pyplot as plt
import os
import pandas as pd
# Anggap saja ada fungsi untuk memuat data dari skrip lain
from data_processing import load_processed_spectrum 

def plot_spectrum(spectrum_df, title, output_path):
    """
    Membuat dan menyimpan plot dari data spektrum. (Fungsi ini tidak berubah)
    """
    plt.figure(figsize=(12, 6))
    plt.plot(spectrum_df['wavenumber'], spectrum_df['absorbance'], color='teal')
    plt.title(title, fontsize=16)
    plt.xlabel("Wavenumber (cm⁻¹)", fontsize=12)
    plt.ylabel("Absorbance (Normalized)", fontsize=12)
    plt.gca().invert_xaxis()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    plt.savefig(output_path)
    plt.close()
    print(f"📈 Plot disimpan di: {output_path}")

def create_sample_data():
    """Fungsi khusus untuk membuat data dummy untuk pengujian."""
    print("Membuat data contoh untuk pengujian...")
    sample_data = {
        'wavenumber': range(4000, 499, -1),
        'absorbance': [0.1, 0.2, 0.5, 0.8, 0.6, 0.3] * (3501 // 6) + [0.1]
    }
    return pd.DataFrame(sample_data)


def main(output_dir=".", use_sample_data=False):
    """
    Fungsi utama yang akan dipanggil oleh orchestrator (main.py).
    """
    print("--- Menjalankan Modul Visualisasi ---")
    
    if use_sample_data:
        spectrum_df = create_sample_data()
        title = "Contoh Spektrum IR (Data Sampel)"
    else:
        # --- PERUBAHAN DI SINI ---
        # Path data sekarang dinamis, merujuk ke output dari data_processing
        # Ganti 'X_spec_ir.npy' jika nama file Anda berbeda
        processed_data_path = "results/data_processing/ir_dataset/y_ir.npy" 
        
        print(f"Memuat data asli dari: {processed_data_path}")
        
        # Cek apakah file ada sebelum memuat
        if not os.path.exists(processed_data_path):
            print(f"❌ Error: File data yang diproses tidak ditemukan.")
            print("➡️ Pastikan Anda sudah menjalankan 'python main.py data_processing.py' terlebih dahulu.")
            return # Keluar dari fungsi jika file tidak ada

        # Logika untuk memuat file .npy (bukan .csv)
        data_array = np.load(processed_data_path)
        # Anda perlu menyesuaikan ini untuk mengubah array numpy menjadi DataFrame
        # Contoh sederhana (mungkin perlu disesuaikan):
        spectrum_df = pd.DataFrame({
            'wavenumber': range(len(data_array[0])), # Asumsi
            'absorbance': data_array[0] # Ambil spektrum pertama sebagai contoh
        })
        title = "Hasil Analisis Spektrum IR (Data Asli)"

    file_name = "hasil_visualisasi_spektrum.png"
    full_output_path = os.path.join(output_dir, file_name)

    plot_spectrum(spectrum_df, title, full_output_path)
    print("--- Modul Visualisasi Selesai ---")
    
if __name__ == '__main__':
    """
    Saat file dijalankan langsung, gunakan sample_data.
    """
    default_folder = "default_output/visualization"
    print(f"Menjalankan {__file__} secara mandiri untuk pengujian...")
    # Memanggil main dengan flag use_sample_data=True
    main(output_dir=default_folder, use_sample_data=True)