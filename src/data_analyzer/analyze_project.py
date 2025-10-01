# analyze_project.py

import os
import yaml
import glob
import re
import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import seaborn as sns

# Impor fungsi dari proyek Anda
from src.data_processing import parse_jdx, preprocess_spectrum
from src.auto_labeler import SPECTRAL_RULES, autogenerate_functional_groups
from src.auto_labeler import UV_SPECTRAL_RULES, autogenerate_chromophores

def process_all_spectra_to_dataframe(config):
    """
    Membangun database dari semua spektrum di mana setiap baris adalah satu puncak.
    """
    paths = config['paths']
    all_peak_data = []

    # Buat mapping aturan ke rentang untuk pencarian cepat
    ir_rule_map = {rule['group']: rule['range'] for rule in SPECTRAL_RULES}
    uv_rule_map = {rule['group']: rule['range'] for rule in UV_SPECTRAL_RULES}
    
    all_jdx_files = glob.glob(os.path.join(paths['raw_data_dir'], '**', '*.jdx'), recursive=True)

    for file_path in tqdm(all_jdx_files):
        raw_data = parse_jdx(file_path)
        if not raw_data or not raw_data['x'].any(): continue

        # Ekstrak metadata kunci
        metadata = raw_data.get('metadata', {})
        molform = metadata.get('molform', 'N/A')
        title = metadata.get('title', 'N/A')
        state = metadata.get('state', 'unknown')
        cas_no = os.path.basename(os.path.dirname(file_path))

        # Tentukan tipe spektrum (IR atau UV)
        filename = os.path.basename(file_path).lower()
        spectrum_type = None
        if re.search(r'(ftir|ft-ir|ir)', filename):
            spectrum_type = 'ir'
        elif re.search(r'uv[\-\s]?vis', filename):
            spectrum_type = 'uv'
        else:
            continue

        # Proses spektrum
        processed_df = preprocess_spectrum(raw_data, config, spectrum_type, normalize=True)
        if processed_df is None or processed_df.empty: continue
        
        # Dapatkan label menggunakan auto_labeler
        if spectrum_type == 'ir':
            labels = autogenerate_functional_groups(processed_df)
            x_vals = processed_df['wavenumber'].values
            y_vals = processed_df['absorbance'].values
            peaks, props = find_peaks(y_vals, prominence=0.02, width=1)
        else: # uv
            unnormalized_df = preprocess_spectrum(raw_data, config, 'uv', normalize=False)
            labels = autogenerate_chromophores(unnormalized_df)
            x_vals = processed_df['wavelength'].values
            y_vals = processed_df['log_epsilon'].values
            peaks, props = find_peaks(y_vals, prominence=0.01, width=3)
        
        # Simpan setiap puncak sebagai baris data
        for i, peak_idx in enumerate(peaks):
            peak_location = x_vals[peak_idx]
            
            # Cari label yang cocok dengan lokasi puncak ini
            associated_label = 'unassigned'
            if spectrum_type == 'ir':
                for label in labels:
                    if label in ir_rule_map and ir_rule_map[label][0] <= peak_location <= ir_rule_map[label][1]:
                        associated_label = label
                        break
            else:
                 for label in labels:
                    if label in uv_rule_map and uv_rule_map[label][0] <= peak_location <= uv_rule_map[label][1]:
                        associated_label = label
                        break
            
            peak_info = {
                "cas": cas_no,
                "molform": molform,
                "title": title,
                "state": state,
                "spectrum_type": spectrum_type,
                "label": associated_label,
                "peak_location": peak_location,
                "peak_prominence": props['prominences'][i],
                "peak_width": props['widths'][i],
                "peak_intensity": y_vals[peak_idx]
            }
            all_peak_data.append(peak_info)
            
    return pd.DataFrame(all_peak_data)

def generate_analysis_plots(df, output_dir, timestamp):
    """Menghasilkan beberapa plot analisis dari database puncak."""
    print(f"\n🎨 Menghasilkan plot analisis di: {output_dir}")
    plt.style.use('seaborn-v0_8-whitegrid')

    # --- Plot 1: Analisis Lebar Puncak IR Berdasarkan Fasa (State) ---
    ir_df = df[df['spectrum_type'] == 'ir'].copy()
    ir_df['state_simple'] = ir_df['state'].apply(lambda s: 'gas' if 'gas' in s.lower() else ('liquid' if 'liquid' in s.lower() else 'solid/other'))
    
    plt.figure(figsize=(10, 7))
    sns.boxplot(data=ir_df, x='state_simple', y='peak_width', order=['gas', 'liquid', 'solid/other'])
    plt.title('Distribusi Lebar Puncak IR Berdasarkan Fasa Sampel', fontsize=16)
    plt.xlabel('Fasa Sampel', fontsize=12)
    plt.ylabel('Lebar Puncak (Wavenumber cm⁻¹)', fontsize=12)
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'ir_width_by_state_{timestamp}.png'))
    plt.close()
    print("  -> Plot 1/4 (IR Width by State) disimpan.")

    # --- Plot 2: Analisis Lebar Puncak IR Berdasarkan Gugus Fungsi (untuk fasa padat) ---
    solid_ir_df = ir_df[ir_df['state_simple'] == 'solid/other']
    top_labels = solid_ir_df['label'].value_counts().nlargest(15).index
    plot_df = solid_ir_df[solid_ir_df['label'].isin(top_labels) & (solid_ir_df['label'] != 'unassigned')]
    
    plt.figure(figsize=(12, 8))
    sns.boxplot(data=plot_df, x='peak_width', y='label', order=sorted(top_labels.drop('unassigned')))
    plt.title('Distribusi Lebar Puncak untuk Gugus Fungsi IR Umum (Fasa Padat)', fontsize=16)
    plt.xlabel('Lebar Puncak (Wavenumber cm⁻¹)', fontsize=12)
    plt.ylabel('Gugus Fungsi', fontsize=12)
    plt.xscale('log')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'ir_width_by_group_{timestamp}.png'))
    plt.close()
    print("  -> Plot 2/4 (IR Width by Group) disimpan.")
    
    # --- Plot 3: Distribusi Jumlah Puncak per Spektrum UV ---
    uv_df = df[df['spectrum_type'] == 'uv']
    peak_counts = uv_df.groupby('cas').size()
    
    plt.figure(figsize=(10, 7))
    sns.histplot(peak_counts, bins=np.arange(1, peak_counts.max() + 2) - 0.5, discrete=True)
    plt.title('Distribusi Jumlah Puncak per Spektrum UV-Vis', fontsize=16)
    plt.xlabel('Jumlah Puncak Terdeteksi per Spektrum', fontsize=12)
    plt.ylabel('Jumlah Senyawa', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'uv_peak_counts_{timestamp}.png'))
    plt.close()
    print("  -> Plot 3/4 (UV Peak Counts) disimpan.")

    # --- Plot 4: Hubungan Intensitas (log ε) vs. Posisi Puncak (λ) di UV ---
    uv_peaks_df = df[df['spectrum_type'] == 'uv'].copy()
    uv_peaks_df['label_category'] = uv_peaks_df['label'].apply(lambda l: 'π -> π*' if 'pi_pi' in l else ('n -> π*' if 'n_pi' in l else 'other'))

    plt.figure(figsize=(12, 8))
    sns.scatterplot(data=uv_peaks_df, x='peak_location', y='peak_intensity', hue='label_category', alpha=0.7, s=50)
    plt.title('Hubungan Intensitas vs. Posisi Puncak UV-Vis', fontsize=16)
    plt.xlabel('Posisi Puncak (λ, nm)', fontsize=12)
    plt.ylabel('Intensitas Puncak (log ε)', fontsize=12)
    plt.legend(title='Tipe Transisi (Dugaan)')
    plt.grid(True, which='both', linestyle='--')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'uv_intensity_vs_lambda_{timestamp}.png'))
    plt.close()
    print("  -> Plot 4/4 (UV Intensity vs Lambda) disimpan.")


if __name__ == "__main__":
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    with open('main_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
        
    output_dir = os.path.join(config['paths']['reports_dir'], 'spectrum_analysis')
    os.makedirs(output_dir, exist_ok=True)
    
    print("Membangun database puncak dari semua spektrum. Ini mungkin memakan waktu...")
    master_peak_df = process_all_spectra_to_dataframe(config)
    
    if not master_peak_df.empty:
        csv_path = os.path.join(output_dir, f'full_peak_database_{timestamp}.csv')
        master_peak_df.to_csv(csv_path, index=False)
        print(f"\nDatabase puncak lengkap disimpan di: {csv_path}")
        
        generate_analysis_plots(master_peak_df, output_dir, timestamp)
        print("\nAnalisis selesai.")
    else:
        print("\nTidak ada puncak yang terdeteksi di seluruh dataset. Analisis dihentikan.")