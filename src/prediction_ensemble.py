# prediction_ensemble.py (Versi BARU dengan output gabungan IR & UV)

import os
import yaml
import joblib
import numpy as np
import tensorflow as tf
from src.data_processing import parse_jdx, preprocess_spectrum, extract_metadata_feature

def predict_compound_ensemble(ir_file_path, uv_file_path, config_path='main_config.yaml'):
    """
    Memprediksi gugus fungsi (IR) dan kromofor (UV) dari sepasang file spektrum
    menggunakan sistem ensemble stacking yang sudah dilatih.
    """
    # 1. Muat Konfigurasi dan Semua Model
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    paths = config['paths']
    
    print("🧠 Memuat semua model terlatih (2x CNN, 1x XGBoost)...")
    try:
        base_model_ir = tf.keras.models.load_model(os.path.join(paths['saved_models_dir'], 'base_model_ir_final.keras'), compile=False)
        base_model_uv = tf.keras.models.load_model(os.path.join(paths['saved_models_dir'], 'base_model_uv_final.keras'), compile=False)
        meta_model = joblib.load(os.path.join(paths['saved_models_dir'], 'meta_model_final.joblib'))
        
        # Muat KEDUA binarizer
        label_binarizer_ir = joblib.load(os.path.join(paths['saved_models_dir'], 'label_binarizer_ir.joblib'))
        label_binarizer_uv = joblib.load(os.path.join(paths['saved_models_dir'], 'label_binarizer_uv.joblib'))
        meta_encoder_ir = joblib.load(os.path.join(paths['saved_models_dir'], 'metadata_encoder_ir.joblib'))
        meta_encoder_uv = joblib.load(os.path.join(paths['saved_models_dir'], 'metadata_encoder_uv.joblib'))
    except Exception as e:
        print(f"❌ Error: Gagal memuat salah satu file model. Pastikan proses pelatihan sudah selesai. Detail: {e}")
        return

    # 2. Proses Spektrum (Sama seperti sebelumnya)
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
    
    # 3. Dapatkan Prediksi Probabilitas dari Model Dasar
    prob_ir = base_model_ir.predict([spec_vec_ir, meta_vec_ir])
    prob_uv = base_model_uv.predict([spec_vec_uv, meta_vec_uv])

    # 4. Buat Fitur Meta
    new_meta_feature = np.concatenate([prob_ir, prob_uv], axis=1)

    # 5. Dapatkan Prediksi Probabilitas Final dan Terapkan Threshold
    final_prediction_probas_list = meta_model.predict_proba(new_meta_feature)
    positive_probas = np.array([proba[0][1] for proba in final_prediction_probas_list])
    threshold = config['prediction']['probability_threshold']
    final_prediction_binary = (positive_probas >= threshold).astype(int)

    # ====================================================================
    # == PERUBAHAN: Pisahkan hasil IR dan UV untuk tampilan yang rapi   ==
    # ====================================================================
    num_ir_classes = len(label_binarizer_ir.classes_)
    
    # Pisahkan prediksi biner dan probabilitas
    pred_binary_ir = final_prediction_binary[:num_ir_classes].reshape(1, -1)
    pred_binary_uv = final_prediction_binary[num_ir_classes:].reshape(1, -1)
    
    probas_ir = positive_probas[:num_ir_classes]
    probas_uv = positive_probas[num_ir_classes:]

    # Ubah kembali ke nama label
    predicted_labels_ir_tup = label_binarizer_ir.inverse_transform(pred_binary_ir)
    predicted_labels_uv_tup = label_binarizer_uv.inverse_transform(pred_binary_uv)

    # Tampilkan hasil
    print("\n✅ Prediksi Final Ensemble (Gabungan IR & UV):")
    
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
    # ====================================================================

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Prediksi Senyawa Menggunakan Model Ensemble Gabungan")
    parser.add_argument('--ir', required=True, help='Path ke file JDX spektrum IR')
    parser.add_argument('--uv', required=True, help='Path ke file JDX spektrum UV-Vis')
    args = parser.parse_args()
    
    predict_compound_ensemble(args.ir, args.uv)