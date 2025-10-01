import pandas as pd
import matplotlib.pyplot as plt
import ast # Untuk mengubah string list menjadi list sungguhan

# Muat database yang dihasilkan oleh script analisis
df = pd.read_csv("exploration_database.csv")

# Fungsi untuk mengubah string menjadi list dengan aman
def parse_list_string(s):
    try:
        return ast.literal_eval(s)
    except (ValueError, SyntaxError):
        return []

# Konversi kolom list dari string ke list
df['ir_labels'] = df['ir_labels'].apply(parse_list_string)
df['ir_peaks'] = df['ir_peaks'].apply(parse_list_string)

# --- Analisis untuk 'ketone_co' ---
# 1. Filter baris yang mengandung label 'ketone_co'
ketone_df = df[df['ir_labels'].apply(lambda labels: 'aromatic_out_of_plane_ch' in labels)]

# 2. Ambil semua puncak dari baris yang terfilter dan ratakan menjadi satu list
all_ketone_peaks = ketone_df['ir_peaks'].explode().dropna()

# 3. Plot histogramnya
plt.figure(figsize=(12, 6))
plt.hist(all_ketone_peaks, bins=100, edgecolor='k', alpha=0.7) # Gunakan lebih banyak bin untuk data besar
plt.title("Distribusi Puncak 'ketone_co' dari Ribuan Sampel", fontsize=16)
plt.xlabel("Wavenumber (cm⁻¹)", fontsize=12)
plt.ylabel("Jumlah Puncak Terdeteksi", fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()