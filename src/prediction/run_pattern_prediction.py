# src/run_pattern_prediction.py

import os
import yaml
import joblib
import numpy as np
import matplotlib.pyplot as plt

from data.data_processing import parse_jdx, preprocess_spectrum

def execute_pattern_prediction(args, config):
    """
    Memprediksi pola substitusi aromatik dari satu file spektrum IR
    menggunakan model AI yang sudah dilatih.
    """
    paths = config['paths']
    file_path = args.file

    if not os.path.exists(file_path):
        print(f"❌ Error: File tidak ditemukan di '{file_path}'")
        return

    # 1. Muat model dan encoder yang sudah dilatih
    print("🧠 Memuat model pengenalan pola...")
    try:
        model_path = os.path.join(paths['saved_models_dir'], 'pattern_recognition_model.joblib')
        encoder_path = os.path.join(paths['saved_models_dir'], 'pattern_label_encoder.joblib')
        model = joblib.load(model_path)
        le = joblib.load(encoder_path)
    except FileNotFoundError:
        print(f"❌ Error: File model atau encoder tidak ditemukan. Jalankan 'python main.py train-pattern' terlebih dahulu.")
        return

    # 2. Pra-pemrosesan spektrum input agar sesuai dengan data latih
    print(f"🔬 Memproses spektrum: {os.path.basename(file_path)}")
    
    # Definisikan grid yang sama persis seperti saat pelatihan
    FINGERPRINT_START = 650
    FINGERPRINT_STOP = 900
    FINGERPRINT_POINTS = 250
    fingerprint_grid = np.linspace(FINGERPRINT_START, FINGERPRINT_STOP, FINGERPRINT_POINTS)

    try:
        raw_data = parse_jdx(file_path)
        if not raw_data:
            print("❌ Gagal mem-parsing file JDX.")
            return

        # Gunakan preprocess_spectrum untuk pembersihan awal
        temp_df = preprocess_spectrum(raw_data, config, 'ir', normalize=False)
        if temp_df is None:
            print("❌ Gagal memproses spektrum.")
            return

        # Interpolasi ke grid sidik jari yang spesifik
        fingerprint_data = np.interp(fingerprint_grid, temp_df['wavenumber'], temp_df['absorbance'])

        # Normalisasi lokal hanya pada daerah sidik jari
        min_val, max_val = fingerprint_data.min(), fingerprint_data.max()
        if max_val > min_val:
            fingerprint_norm = (fingerprint_data - min_val) / (max_val - min_val)
        else:
            fingerprint_norm = fingerprint_data # Spektrum datar

    except Exception as e:
        print(f"❌ Terjadi error saat memproses spektrum: {e}")
        return

    # 3. Lakukan Prediksi
    # Model mengharapkan input 2D, jadi kita reshape
    prediction_encoded = model.predict(fingerprint_norm.reshape(1, -1))
    
    # 4. Ubah hasil prediksi kembali ke label teks
    predicted_pattern = le.inverse_transform(prediction_encoded)

    print("\n" + "="*40)
    print(f"✅ Prediksi Pola Substitusi: {predicted_pattern[0]}")
    print("="*40)

    # --- PERBAIKAN: Tambahkan visualisasi jika diminta ---
    if args.visualize:
        print("📊 Membuat plot visualisasi...")
        plt.figure(figsize=(12, 6))
        plt.plot(fingerprint_grid, fingerprint_norm, label='Daerah Sidik Jari (Input ke Model)')
        
        plt.title(f"Prediksi Pola: {predicted_pattern[0]}", fontsize=16)
        plt.xlabel("Wavenumber (cm⁻¹)")
        plt.ylabel("Intensitas (Dinormalisasi Lokal)")
        plt.xlim(max(fingerprint_grid), min(fingerprint_grid)) # Balik sumbu-x untuk IR
        plt.grid(True, linestyle='--')
        plt.legend()
        plt.tight_layout()
        plt.show()