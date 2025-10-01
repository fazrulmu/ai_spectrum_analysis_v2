# predict.py (Versi Gabungan Final yang Cerdas)

import os
import yaml
import joblib
import numpy as np
import tensorflow as tf
import argparse

# Impor fungsi-fungsi yang relevan dari proyek Anda
from src.data_processing import parse_jdx, preprocess_spectrum, extract_metadata_feature

def predict_single_spectrum(file_path, spectrum_type, config):
    """
    Memprediksi menggunakan HANYA SATU spektrum (IR atau UV).
    Menggunakan model dasar yang dilatih selama proses ensemble.
    """
    paths = config['paths']
    print(f"🧠 Memuat model dasar {spectrum_type.upper()}...")
    try:
        model = tf.keras.models.load_model(os.path.join(paths['saved_models_dir'], f'base_model_{spectrum_type}_final.keras'), compile=False)
        meta_encoder = joblib.load(os.path.join(paths['saved_models_dir'], f'metadata_encoder_{spectrum_type}.joblib'))
        label_binarizer = joblib.load(os.path.join(paths['saved_models_dir'], f'label_binarizer_{spectrum_type}.joblib'))
    except Exception as e:
        print(f"❌ Error: Gagal memuat model untuk {spectrum_type}. Pastikan pelatihan selesai. Detail: {e}")
        return

    print(f"Processing {spectrum_type.upper()} spectrum: {file_path}")
    raw_data = parse_jdx(file_path)
    if spectrum_type == 'ir':
        processed_df = preprocess_spectrum(raw_data, config, 'ir', normalize=True)
        spec_vec = processed_df['absorbance'].values.reshape(1, -1, 1)
    else: # uv
        processed_df = preprocess_spectrum(raw_data, config, 'uv', normalize=True)
        spec_vec = processed_df['log_epsilon'].values.reshape(1, -1, 1)

    meta_vec = meta_encoder.transform([extract_metadata_feature(raw_data['metadata'])]).toarray()
    
    # Dapatkan prediksi probabilitas
    probabilities = model.predict([spec_vec, meta_vec])[0]
    
    # Terapkan threshold
    threshold = config['prediction']['probability_threshold']
    prediction_binary = (probabilities >= threshold).astype(int).reshape(1, -1)
    predicted_labels_tup = label_binarizer.inverse_transform(prediction_binary)

    print(f"\n✅ Prediksi (Hanya Model {spectrum_type.upper()}):")
    if predicted_labels_tup[0]:
        all_classes = list(label_binarizer.classes_)
        for label in predicted_labels_tup[0]:
            idx = all_classes.index(label)
            confidence = probabilities[idx]
            print(f"- {label} (Kepercayaan: {confidence:.1%})")
    else:
        print(f"- Tidak ada fitur yang terdeteksi oleh model {spectrum_type.upper()}.")


def predict_compound_ensemble(ir_file_path, uv_file_path, config):
    """
    Memprediksi menggunakan KEDUA spektrum (IR dan UV) dengan model ensemble.
    (Ini adalah fungsi dari prediction_ensemble.py sebelumnya)
    """
    paths = config['paths']
    print("🧠 Memuat semua model terlatih untuk ENSEMBLE...")
    try:
        base_model_ir = tf.keras.models.load_model(os.path.join(paths['saved_models_dir'], 'base_model_ir_final.keras'), compile=False)
        base_model_uv = tf.keras.models.load_model(os.path.join(paths['saved_models_dir'], 'base_model_uv_final.keras'), compile=False)
        meta_model = joblib.load(os.path.join(paths['saved_models_dir'], 'meta_model_final.joblib'))
        label_binarizer_ir = joblib.load(os.path.join(paths['saved_models_dir'], 'label_binarizer_ir.joblib'))
        label_binarizer_uv = joblib.load(os.path.join(paths['saved_models_dir'], 'label_binarizer_uv.joblib'))
        meta_encoder_ir = joblib.load(os.path.join(paths['saved_models_dir'], 'metadata_encoder_ir.joblib'))
        meta_encoder_uv = joblib.load(os.path.join(paths['saved_models_dir'], 'metadata_encoder_uv.joblib'))
    except Exception as e:
        print(f"❌ Error: Gagal memuat model. Pastikan pelatihan selesai. Detail: {e}")
        return

    # Proses Spektrum IR & UV
    print(f"Processing IR spectrum: {ir_file_path}")
    raw_data_ir = parse_jdx(ir_file_path)
    processed_df_ir = preprocess_spectrum(raw_data_ir, config, 'ir', normalize=True)
    spec_vec_ir = processed_df_ir['absorbance'].values.reshape(1, -1, 1)
    meta_vec_ir = meta_encoder_ir.transform([extract_metadata_feature(raw_data_ir['metadata'])]).toarray()
    
    print(f"Processing UV spectrum: {uv_file_path}")
    raw_data_uv = parse_jdx(uv_file_path)
    processed_df_uv = preprocess_spectrum(raw_data_uv, config, 'uv', normalize=True)
    spec_vec_uv = processed_df_uv['log_epsilon'].values.reshape(1, -1, 1)
    meta_vec_uv = meta_encoder_uv.transform([extract_metadata_feature(raw_data_uv['metadata'])]).toarray()
    
    # Dapatkan Prediksi Probabilitas dari Model Dasar
    prob_ir = base_model_ir.predict([spec_vec_ir, meta_vec_ir])
    prob_uv = base_model_uv.predict([spec_vec_uv, meta_vec_uv])

    # Buat Fitur Meta & Prediksi Final
    new_meta_feature = np.concatenate([prob_ir, prob_uv], axis=1)
    final_prediction_probas_list = meta_model.predict_proba(new_meta_feature)
    positive_probas = np.array([proba[0][1] for proba in final_prediction_probas_list])
    threshold = config['prediction']['probability_threshold']
    final_prediction_binary = (positive_probas >= threshold).astype(int)

    # Tampilkan hasil terpisah
    num_ir_classes = len(label_binarizer_ir.classes_)
    pred_binary_ir = final_prediction_binary[:num_ir_classes].reshape(1, -1)
    pred_binary_uv = final_prediction_binary[num_ir_classes:].reshape(1, -1)
    probas_ir = positive_probas[:num_ir_classes]
    probas_uv = positive_probas[num_ir_classes:]

    predicted_labels_ir_tup = label_binarizer_ir.inverse_transform(pred_binary_ir)
    predicted_labels_uv_tup = label_binarizer_uv.inverse_transform(pred_binary_uv)

    print("\n✅ Prediksi Final (Model ENSEMBLE Gabungan):")
    print("\n--- Gugus Fungsi (IR) ---")
    if predicted_labels_ir_tup[0]:
        all_classes_ir = list(label_binarizer_ir.classes_)
        for label in predicted_labels_ir_tup[0]:
            idx = all_classes_ir.index(label)
            confidence = probas_ir[idx]
            print(f"- {label} (Kepercayaan: {confidence:.1%})")
    else:
        print("- Tidak ada gugus fungsi IR yang terdeteksi.")
        
    print("\n--- Kromofor (UV-Vis) ---")
    if predicted_labels_uv_tup[0]:
        all_classes_uv = list(label_binarizer_uv.classes_)
        for label in predicted_labels_uv_tup[0]:
            idx = all_classes_uv.index(label)
            confidence = probas_uv[idx]
            print(f"- {label} (Kepercayaan: {confidence:.1%})")
    else:
        print("- Tidak ada kromofor UV yang terdeteksi.")


if __name__ == '__main__':
    # Logika utama untuk memilih mode prediksi
    parser = argparse.ArgumentParser(description="Script Prediksi Spektrum Cerdas (Tunggal atau Ensemble)")
    parser.add_argument('--ir', help='Path ke file JDX spektrum IR')
    parser.add_argument('--uv', help='Path ke file JDX spektrum UV-Vis')
    args = parser.parse_args()

    # Muat konfigurasi sekali saja
    with open('main_config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    if args.ir and args.uv:
        print("Mode: ENSEMBLE (IR & UV disediakan)")
        predict_compound_ensemble(args.ir, args.uv, config)
    elif args.ir:
        print("Mode: IR TUNGGAL (hanya IR disediakan)")
        predict_single_spectrum(args.ir, 'ir', config)
    elif args.uv:
        print("Mode: UV TUNGGAL (hanya UV disediakan)")
        predict_single_spectrum(args.uv, 'uv', config)
    else:
        print("❌ Error: Harap sediakan path file spektrum menggunakan --ir dan/atau --uv.")
        print("Contoh: python predict.py --ir path/ke/file.jdx --uv path/ke/file2.jdx")