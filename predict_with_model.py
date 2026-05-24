import pandas as pd
import numpy as np
import joblib
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import re
from scipy.signal import find_peaks

# --- Pengaturan Path untuk Impor Modul Proyek ---
# Ini memastikan kita bisa mengimpor fungsi dari skrip lain
SRC_PATH = Path(__file__).parent / 'src'
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

# Mengimpor fungsi pra-pemrosesan yang sudah kita buat
# Diasumsikan 'preprocess_universal.py' ada di direktori utama
try:
    from preprocess_universal import process_spectrum_file
except ImportError:
    print("❌ Gagal mengimpor 'process_spectrum_file'. Pastikan 'preprocess_universal.py' ada di direktori utama.")
    sys.exit(1)

# --- BARU: Kamus untuk rentang wavenumber gugus fungsi ---
# Sumber: Pengalaman umum spektroskopi IR, dapat disesuaikan.
FUNCTIONAL_GROUP_RANGES = {
    "R-OH": {"O-H stretch (broad)": (3200, 3550)},
    "Ar-OH": {"O-H stretch (broad)": (3200, 3550)},
    "R-COOH": {
        "O-H stretch (very broad)": (2500, 3300),
        "C=O stretch": (1700, 1725)
    },
    "R-COO-": {
        "C=O asym stretch": (1550, 1610),
        "C=O sym stretch": (1360, 1420)
    },
    "R-COOR": {
        "C=O stretch": (1735, 1750),
        "C-O stretch": (1150, 1300)
    },
    "Ar-COOR": {
        "C=O stretch": (1715, 1730),
        "C-O stretch": (1250, 1310)
    },
    "R-CHO": {
        "C=O stretch": (1720, 1740),
        "C-H stretch (aldehyde)": (2700, 2900)
    },
    "R-CO-R": {"C=O stretch": (1705, 1725)},
    "R-O-R": {"C-O stretch": (1070, 1150)},
    "Ar-O-R": {
        "Aryl C-O stretch": (1200, 1275),
        "Alkyl C-O stretch": (1020, 1075)
    },
    "R-NH2": {"N-H stretch (2 bands)": (3300, 3500)},
    "R-C≡N": {"C≡N stretch": (2210, 2260)},
    "R-NO2": {
        "N=O asym stretch": (1500, 1570),
        "N=O sym stretch": (1300, 1370)
    },
    "Ar": {"C=C stretch (aromatic)": (1400, 1600)}
}

def analyze_peak_properties(spectrum_df: pd.DataFrame, peak_range: tuple):
    """
    Menganalisis wilayah spektrum tertentu untuk menemukan puncak utama dan propertinya.

    Args:
        spectrum_df (pd.DataFrame): DataFrame format panjang dari spektrum.
        peak_range (tuple): Tuple (min_wavenumber, max_wavenumber) untuk dianalisis.

    Returns:
        dict: Kamus berisi statistik puncak, atau None jika tidak ada puncak yang ditemukan.
    """
    min_wn, max_wn = min(peak_range), max(peak_range)
    region_df = spectrum_df[(spectrum_df['x_value'] >= min_wn) & (spectrum_df['x_value'] <= max_wn)]

    if region_df.empty or len(region_df) < 3:
        return None

    # Gunakan find_peaks untuk mendapatkan detail puncak
    # prominence=0.01 berarti puncak harus setidaknya 1% lebih tinggi dari sekitarnya
    peaks_indices, properties = find_peaks(
        region_df['y_value'],
        prominence=0.01,
        width=1  # Meminta properti lebar
    )

    if len(peaks_indices) == 0:
        return None

    # Ambil puncak dengan prominence tertinggi
    main_peak_idx_local = np.argmax(properties['prominences'])
    main_peak_index = region_df.index[peaks_indices[main_peak_idx_local]]

    # Hitung lebar puncak dalam satuan cm-1
    width_points = properties['widths'][main_peak_idx_local]
    # Asumsikan jarak antar titik data seragam di wilayah kecil ini
    avg_point_spacing = np.mean(np.diff(region_df['x_value']))
    peak_width_cm = abs(width_points * avg_point_spacing)

    return {
        "location": spectrum_df.loc[main_peak_index, 'x_value'],
        "height": spectrum_df.loc[main_peak_index, 'y_value'],
        "prominence": properties['prominences'][main_peak_idx_local],
        "width_cm": peak_width_cm
    }

def create_feature_vector_from_jdx(jdx_path: Path, training_columns: list) -> pd.DataFrame:
    """
    Memproses satu file JDX dan mengubahnya menjadi vektor fitur (DataFrame baris tunggal)
    yang cocok dengan format data training.

    Args:
        jdx_path (Path): Path ke file .jdx yang baru.
        training_columns (list): Daftar nama kolom 'bin_...' dari dataset training.

    Returns:
        pd.DataFrame: DataFrame dengan satu baris yang siap untuk prediksi.
    """
    print(f"🔬 Memproses file JDX: {jdx_path.name}")

    # --- Langkah 1: Pra-pemrosesan menggunakan fungsi yang ada ---
    # Kita akan menyimpan output sementara di direktori 'temp'
    temp_dir = Path("temp_processed")
    temp_dir.mkdir(exist_ok=True)
    
    # Ubah sementara direktori output global di preprocess_universal
    # Ini adalah cara cepat, idealnya fungsi akan menerima output_dir
    import preprocess_universal
    original_output_dir = preprocess_universal.PROCESSED_SPECTRUM_DIR
    preprocess_universal.PROCESSED_SPECTRUM_DIR = temp_dir
    
    processed_csv_path_str, _ = process_spectrum_file(jdx_path)
    
    # Kembalikan path ke nilai aslinya
    preprocess_universal.PROCESSED_SPECTRUM_DIR = original_output_dir

    if not processed_csv_path_str:
        print("❌ Gagal memproses file JDX.")
        return None

    processed_csv_path = Path(processed_csv_path_str)
    df_long = pd.read_csv(processed_csv_path)

    # --- Langkah 2: Transformasi ke Format Lebar (Binning & Pivot) ---
    # Proses ini meniru 'create_training_dataset.py'
    df_long['x_bin'] = df_long['x_value'].round().astype(int)
    
    wide_df = df_long.pivot_table(
        index='molecule_id',
        columns='x_bin',
        values='y_value',
        aggfunc='max'
    ).fillna(0)

    wide_df.rename(columns={col: f"bin_{round(col)}" for col in wide_df.columns}, inplace=True)

    # --- Langkah 3: Pastikan Kolom Cocok dengan Data Training ---
    # Gunakan .reindex() untuk menyelaraskan kolom dengan data training.
    # Ini secara otomatis akan:
    # 1. Menambahkan kolom fitur yang hilang dari spektrum baru (diisi dengan NaN).
    # 2. Menghapus kolom yang ada di spektrum baru tapi tidak ada di training.
    # 3. Mengurutkan kolom agar sama persis dengan data training.
    feature_vector = wide_df.reindex(columns=training_columns).fillna(0)

    # Hapus file sementara
    processed_csv_path.unlink()
    if not any(temp_dir.iterdir()):
        temp_dir.rmdir()

    print("✅ Vektor fitur berhasil dibuat.")
    return feature_vector

def visualize_comparison(new_spectrum_df: pd.DataFrame, best_match_series: pd.Series, detected_groups: list, output_path: Path):
    """
    Membuat dan menyimpan plot perbandingan antara spektrum baru dan spektrum terbaik dari data training.

    Args:
        new_spectrum_df (pd.DataFrame): DataFrame format panjang dari spektrum baru (input).
        best_match_series (pd.Series): Series (satu baris) dari data training yang paling cocok.
        detected_groups (list): Daftar gugus fungsi yang terdeteksi untuk anotasi.
        output_path (Path): Path untuk menyimpan file gambar plot.
    """
    print(f"🎨 Membuat plot perbandingan...")

    # --- 1. Siapkan data spektrum dari sampel terbaik (format lebar ke panjang) ---
    feature_cols = [col for col in best_match_series.index if col.startswith('bin_')]
    match_spectrum_data = best_match_series[feature_cols]
    
    # Ekstrak wavenumber (x) dan absorbance (y)
    match_x = [int(re.search(r'bin_(\d+)', col).group(1)) for col in match_spectrum_data.index]
    match_y = match_spectrum_data.values
    match_id = best_match_series['molecule_id']

    # --- 2. Siapkan data spektrum dari sampel baru (sudah format panjang) ---
    new_x = new_spectrum_df['x_value']
    new_y = new_spectrum_df['y_value']
    new_id = new_spectrum_df['molecule_id'].iloc[0]

    # --- 3. Buat Plot ---
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(16, 8))

    # Plot spektrum sampel baru
    plt.plot(new_x, new_y, label=f"Sampel Input: {new_id}", color='blue', linewidth=2)

    # Plot spektrum dari data training yang cocok
    plt.plot(match_x, match_y, label=f"Sampel Mirip dari Training: {match_id}", color='red', linestyle='--', linewidth=1.5)

    # --- 4. Tandai Area Gugus Fungsi yang Terdeteksi ---
    plotted_group_labels = set()
    colors = plt.cm.viridis(np.linspace(0, 1, len(FUNCTIONAL_GROUP_RANGES)))
    color_map = {list(FUNCTIONAL_GROUP_RANGES.keys())[i]: colors[i] for i in range(len(FUNCTIONAL_GROUP_RANGES))}

    for group_info in detected_groups:
        group_name_key = group_info["group"].replace(' ', '-') # Ubah 'R-COOR' menjadi 'R-COOR'
        if group_name_key in FUNCTIONAL_GROUP_RANGES:
            for region_name, (start, end) in FUNCTIONAL_GROUP_RANGES[group_name_key].items():
                label = f"{group_name_key} ({region_name})"
                if label not in plotted_group_labels: # Hindari duplikasi label di legenda
                    plt.axvspan(start, end, color=color_map.get(group_name_key, 'gray'), alpha=0.15, label=label)
                    plotted_group_labels.add(label)
                else:
                    plt.axvspan(start, end, color=color_map.get(group_name_key, 'gray'), alpha=0.15)

    plt.title(f"Perbandingan Spektrum: {new_id} vs {match_id}", fontsize=18)
    plt.xlabel("Wavenumber (cm⁻¹)", fontsize=12)
    plt.ylabel("Absorbance (Normalized)", fontsize=12)
    plt.gca().invert_xaxis()  # Konvensi spektrum IR
    plt.legend(fontsize=9, loc='upper left')
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.tight_layout()

    # Simpan plot
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print(f"✅ Plot perbandingan berhasil disimpan di: {output_path}")


if __name__ == "__main__":
    # --- KONFIGURASI ---
    # Ganti path ini dengan file JDX yang ingin Anda analisis
    JDX_TO_PREDICT = Path("data/raw/nist_jdx/50-78-2/50-78-2_IR_SOLID_(KBr_DISC)_VS_KBr_1.jdx") # Contoh: Aspirin
    
    MODELS_DIR = Path("models")
    TRAINING_DATASET_PATH = Path("data/for_train/training_dataset.csv")
    REPORTS_DIR = Path("reports/predictions")
    CONFIDENCE_THRESHOLD = 0.50  # Hanya tampilkan hasil dengan kepercayaan >= 50%

    # --- 1. Persiapan: Dapatkan Kolom Fitur & Proses JDX menjadi Vektor Fitur ---
    print(f"📋 Mendapatkan daftar fitur dari: {TRAINING_DATASET_PATH}")
    training_df = pd.read_csv(TRAINING_DATASET_PATH) # Baca seluruh dataset training
    feature_columns = [col for col in training_df.columns if col.startswith('bin_')]

    print(f"\n🔬 Memproses file JDX target: {JDX_TO_PREDICT.name}")
    new_spectrum_features = create_feature_vector_from_jdx(JDX_TO_PREDICT, feature_columns)

    if new_spectrum_features is None or new_spectrum_features.empty:
        print("❌ Gagal membuat vektor fitur. Proses dihentikan.")
        sys.exit(1)

    # --- 2. Iterasi dan Prediksi Menggunakan Semua Model ---
    print(f"\n🤖 Menjalankan prediksi dengan semua model di direktori: {MODELS_DIR}")
    all_model_paths = sorted(list(MODELS_DIR.glob("model_*.joblib")))
    if not all_model_paths:
        print(f"❌ Tidak ada model yang ditemukan di '{MODELS_DIR}'. Pastikan Anda sudah menjalankan 'train_and_analyze_model.py'.")
        sys.exit(1)

    detected_groups = []
    for model_path in all_model_paths:
        model = joblib.load(model_path)
        target_label = model_path.stem.replace('model_', '')
        
        prediction = model.predict(new_spectrum_features)
        
        if prediction[0] == 1:
            probabilities = model.predict_proba(new_spectrum_features)
            confidence = probabilities[0][1] if probabilities.shape[1] == 2 else probabilities[0][0]

            if confidence >= CONFIDENCE_THRESHOLD:
                group_name = target_label.replace('_', '-') # Ubah 'R_OH' menjadi 'R-OH'
                detected_groups.append({"group": group_name, "confidence": confidence})

    # --- 3. Tampilkan Laporan Ringkasan ---
    print("\n" + "="*50)
    print("🎉 HASIL ANALISIS GUGUS FUNGSI 🎉")
    print(f"  - File: {JDX_TO_PREDICT.name}")
    print("="*50)

    if not detected_groups:
        print(f"  -> Tidak ada gugus fungsi yang terdeteksi dengan tingkat kepercayaan di atas {CONFIDENCE_THRESHOLD:.0%}.")
    else:
        detected_groups_sorted = sorted(detected_groups, key=lambda x: x['confidence'], reverse=True)
        print("  Gugus fungsi yang terdeteksi:")
        
        # --- BARU: Analisis Statistik Puncak ---
        # Muat data spektrum format panjang sekali saja untuk analisis
        processed_csv_path_str, _ = process_spectrum_file(JDX_TO_PREDICT)
        spectrum_long_df = pd.read_csv(processed_csv_path_str)

        for item in detected_groups_sorted:
            print(f"    - {item['group']:<25} (Kepercayaan: {item['confidence']:.2%})")
            
            # Cari properti puncak untuk setiap rentang yang relevan
            group_key = item['group']
            if group_key in FUNCTIONAL_GROUP_RANGES:
                for region_name, peak_range in FUNCTIONAL_GROUP_RANGES[group_key].items():
                    peak_stats = analyze_peak_properties(spectrum_long_df, peak_range)
                    if peak_stats:
                        print(f"      - {region_name:<25}:")
                        print(f"        - Lokasi Puncak : {peak_stats['location']:.2f} cm⁻¹")
                        print(f"        - Tinggi Puncak : {peak_stats['height']:.3f} (Absorbansi)")
                        print(f"        - Prominence    : {peak_stats['prominence']:.3f}")
                        print(f"        - Lebar Puncak  : {peak_stats['width_cm']:.2f} cm⁻¹")
                    else:
                        print(f"      - {region_name:<25}: Tidak ada puncak signifikan yang ditemukan.")
    
    print("="*50)

    # --- 4. Cari Sampel Paling Mirip dan Visualisasikan (BAGIAN BARU) ---
    if not detected_groups:
        print("\n🤷 Tidak ada gugus fungsi yang terdeteksi, perbandingan visual dilewati.")
    else:
        print("\n[+] Mencari sampel paling mirip dari data training untuk perbandingan...")
        
        # PERBAIKAN: Gunakan nama grup yang cocok dengan kolom di training_dataset.csv (misal: 'R-OH')
        detected_group_names = {item['group'] for item in detected_groups}
        
        best_match_score = -1
        best_match_index = -1

        # Iterasi melalui dataset training untuk menemukan yang paling cocok
        for index, row in training_df.iterrows():
            current_score = 0
            for group_name in detected_group_names:
                # Cek jika kolom label ada dan nilainya 1
                if group_name in row.index and row[group_name] == 1:
                    current_score += 1
            
            if current_score > best_match_score:
                best_match_score = current_score
                best_match_index = index

        if best_match_index != -1:
            best_match_series = training_df.iloc[best_match_index]
            print(f"✅ Sampel paling mirip ditemukan: '{best_match_series['molecule_id']}' (Skor kecocokan: {best_match_score})")

            # Gunakan kembali data spektrum format panjang yang sudah dimuat
            
            # Buat visualisasi
            plot_output_path = REPORTS_DIR / f"comparison_{JDX_TO_PREDICT.stem}.png"
            visualize_comparison(spectrum_long_df, best_match_series, detected_groups, plot_output_path)
        else:
            print("❌ Tidak dapat menemukan sampel yang cocok di data training.")