# src/visualization.py

import matplotlib.pyplot as plt
import os

def plot_spectrum(spectrum_df, title, output_path):
    """
    Membuat dan menyimpan plot dari data spektrum.

    Args:
        spectrum_df (pd.DataFrame): DataFrame dengan kolom 'wavenumber' dan 'absorbance'.
        title (str): Judul plot.
        output_path (str): Path untuk menyimpan file gambar.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(spectrum_df['wavenumber'], spectrum_df['absorbance'])
    plt.title(title)
    plt.xlabel("Wavenumber (cm⁻¹)")
    plt.ylabel("Absorbance (Normalized)")
    plt.gca().invert_xaxis() # Khas untuk spektrum IR
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    # Pastikan direktori ada
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print(f"Plot disimpan di: {output_path}")