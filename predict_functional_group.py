import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import argparse
from pathlib import Path
from tensorflow.keras.preprocessing.sequence import pad_sequences
import sys
import yaml
import json

# --- PERBAIKAN: Menambahkan path root proyek untuk impor absolut ---
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.data.data_processing import parse_jdx, preprocess_spectrum
except ImportError as e:
    print(f"❌ Gagal mengimpor modul dari 'src'. Pastikan struktur direktori benar. Error: {e}")
    sys.exit(1)

def predict_functional_groups_from_spectrum(file_path: Path, model_path: Path, encoder_path: Path, rules_path: Path):
    """
    Memprediksi SEMUA gugus fungsi yang mungkin dari satu file spektrum dengan menganalisis segmen-segmennya.

    Args:
        file_path (Path): Path ke file input (.jdx atau .csv).
        model_path (Path): Path ke file model Keras (.keras).
        encoder_path (Path): Path ke file encoder joblib (.joblib).
        rules_path (Path): Path ke file JSON yang berisi aturan rentang gugus fungsi.
    """
    # --- 1. Muat Model dan Encoder ---
    print("🧠 Memuat model dan encoder...")
    try:
        model = tf.keras.models.load_model(model_path)
        label_encoder = joblib.load(encoder_path)
    except (IOError, ImportError) as e:
        print(f"❌ Error saat memuat model atau encoder: {e}")
        print(f"Pastikan file '{model_path.name}' dan '{encoder_path.name}' ada di direktori yang benar.")
        return

    print("✅ Model dan encoder berhasil dimuat.")
    model.summary()

    # --- 2. Dapatkan Panjang Input yang Diharapkan dari Model ---
    # Input shape model adalah (None, max_len, 1)
    # PERUBAHAN: Input shape sekarang (None, max_len, 2)
    try: 
        max_len = model.input_shape[1]
        print(f"Panjang sekuens yang diharapkan oleh model: {max_len}")
    except (TypeError, IndexError):
        print("❌ Tidak dapat menentukan panjang input dari arsitektur model.")
        return

    # --- 3. Baca File Input dan Konfigurasi ---
    print(f"🔬 Membaca dan memproses file input: {file_path.name}")
    
    # --- PERUBAHAN: Muat konfigurasi untuk pra-pemrosesan ---
    try:
        with open('main_config.yaml', 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print("❌ Error: File 'main_config.yaml' tidak ditemukan. File ini diperlukan untuk pra-pemrosesan.")
        return

    # --- PERBAIKAN: Muat aturan gugus fungsi ---
    try:
        with open(rules_path, 'r') as f:
            # Ambil aturan dari file JSON yang digunakan untuk membuat dataset training
            # Kita asumsikan struktur {cas: {detected_functional_groups: {group: {range_min: ...}}}}
            # dan kita ambil aturan dari entri pertama.
            structural_data = json.load(f)
            first_cas = next(iter(structural_data))
            functional_group_rules = structural_data[first_cas]['detected_functional_groups']
        print(f"✅ Aturan untuk {len(functional_group_rules)} gugus fungsi berhasil dimuat.")
    except (IOError, KeyError, StopIteration) as e:
        print(f"❌ Gagal memuat atau mem-parsing file aturan '{rules_path.name}'. Error: {e}")
        return

    # --- 4. Pra-pemrosesan Spektrum Penuh ---
    try:
        # Selalu proses file JDX untuk mendapatkan spektrum yang terstandarisasi
        raw_data = parse_jdx(str(file_path))
        full_spectrum_df = preprocess_spectrum(raw_data, config, 'ir', normalize=True)
        if full_spectrum_df is None or full_spectrum_df.empty:
            print("❌ Pra-pemrosesan spektrum menghasilkan data kosong.")
            return
    except Exception as e:
        print(f"❌ Gagal membaca atau memproses file input: {e}")
        return

    # --- 5. Iterasi, Ekstrak Segmen, dan Prediksi ---
    print("\n🤖 Melakukan prediksi untuk setiap segmen gugus fungsi...")
    detected_groups = []

    for group_name, rules in functional_group_rules.items():
        min_wn = rules.get("range_min")
        max_wn = rules.get("range_max")

        if min_wn is None or max_wn is None:
            continue

        # a. Ekstrak segmen spektrum sesuai aturan
        segment_df = full_spectrum_df[
            (full_spectrum_df['wavenumber'] >= min_wn) & (full_spectrum_df['wavenumber'] <= max_wn)
        ]
        
        # PERBAIKAN: Ekstrak nilai absorbansi (y) dan bilangan gelombang (x)
        absorbance_segment = segment_df['absorbance'].tolist()
        wavenumber_segment = segment_df['wavenumber'].tolist()

        if not absorbance_segment:
            continue

        # b. Pra-pemrosesan segmen untuk input 2-channel
        y_padded = pad_sequences([absorbance_segment], maxlen=max_len, dtype='float32', padding='post', truncating='post')
        x_padded = pad_sequences([wavenumber_segment], maxlen=max_len, dtype='float32', padding='post', truncating='post')
        
        # Normalisasi nilai bilangan gelombang (x) seperti saat training
        x_padded_normalized = x_padded / 4000.0

        # Gabungkan menjadi input 2-channel dengan shape (1, max_len, 2)
        X_input = np.stack([y_padded, x_padded_normalized], axis=-1)
        # c. Lakukan prediksi pada segmen ini
        y_pred_probs = model.predict(X_input, verbose=0)
        predicted_class_index = np.argmax(y_pred_probs, axis=1)[0]
        predicted_class_name = label_encoder.inverse_transform([predicted_class_index])[0]
        confidence = y_pred_probs[0][predicted_class_index]

        # d. Jika model memprediksi nama gugus yang sama dengan yang kita uji, catat hasilnya
        if predicted_class_name == group_name: # PERUBAHAN: Hapus threshold kepercayaan untuk menampilkan semua hasil
            detected_groups.append({
                "group": group_name,
                "confidence": confidence
            })

    # --- 6. Tampilkan Laporan Hasil ---
    print("\n" + "="*40)
    print("🎉 HASIL PREDIKSI GUGUS FUNGSI 🎉")
    print("="*40)

    if not detected_groups:
        print("  -> Tidak ada gugus fungsi yang terdeteksi.")
    else:
        # Urutkan hasil berdasarkan tingkat kepercayaan
        detected_groups_sorted = sorted(detected_groups, key=lambda x: x['confidence'], reverse=True)
        print("  Gugus fungsi yang terdeteksi:")
        for item in detected_groups_sorted:
            print(f"    - {item['group']:<20} (Kepercayaan: {item['confidence']:.2%})")

    print("="*40)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Prediksi Gugus Fungsi dari Spektrum menggunakan Model CNN.")
    parser.add_argument(
        '--input', 
        type=str, 
        required=True,
        help="Path ke file spektrum input (.jdx)."
    )
    args = parser.parse_args()

    # Konfigurasi path ke model dan encoder
    MODEL_PATH = Path('models/functional_group_cnn_model.keras')
    ENCODER_PATH = Path('models/functional_group_encoder.joblib')
    # PERBAIKAN: Path ke file aturan yang digunakan untuk membuat dataset training
    RULES_PATH = Path('data/reports/structural_confidence.json')
    INPUT_FILE_PATH = Path(args.input)

    if not INPUT_FILE_PATH.exists():
        print(f"❌ Error: File input tidak ditemukan di '{INPUT_FILE_PATH}'")
    elif not RULES_PATH.exists():
        print(f"❌ Error: File aturan '{RULES_PATH.name}' tidak ditemukan. File ini penting untuk ekstraksi segmen.")
    else:
        predict_functional_groups_from_spectrum(INPUT_FILE_PATH, MODEL_PATH, ENCODER_PATH, RULES_PATH)