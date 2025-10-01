# scripts/run_prediction.py (Logika untuk Prediksi)

import os
import numpy as np
import tensorflow as tf
import joblib

from src.data_processing import parse_jdx, preprocess_spectrum, extract_metadata_feature

def execute_prediction(args, config):
    """Fungsi utama yang dipanggil oleh main.py untuk menjalankan prediksi."""
    
    # Fungsi predict_single_spectrum dan predict_compound_ensemble dari jawaban sebelumnya disalin ke sini
    def _predict_single_spectrum(file_path, spectrum_type, config):
        # ... (kode lengkap dari fungsi predict_single_spectrum)
        pass

    def _predict_compound_ensemble(ir_file_path, uv_file_path, config):
        # ... (kode lengkap dari fungsi predict_compound_ensemble)
        pass
    
    # Tempelkan kode lengkap kedua fungsi tersebut di sini
    def _predict_single_spectrum(file_path, spectrum_type, config):
        paths = config['paths']
        print(f"🧠 Memuat model dasar {spectrum_type.upper()}...")
        model = tf.keras.models.load_model(os.path.join(paths['saved_models_dir'], f'base_model_{spectrum_type}_final.keras'), compile=False)
        meta_encoder = joblib.load(os.path.join(paths['saved_models_dir'], f'metadata_encoder_{spectrum_type}.joblib'))
        label_binarizer = joblib.load(os.path.join(paths['saved_models_dir'], f'label_binarizer_{spectrum_type}.joblib'))
        raw_data = parse_jdx(file_path)
        if spectrum_type == 'ir': processed_df = preprocess_spectrum(raw_data, config, 'ir', True); spec_vec = processed_df['absorbance'].values.reshape(1, -1, 1)
        else: processed_df = preprocess_spectrum(raw_data, config, 'uv', True); spec_vec = processed_df['log_epsilon'].values.reshape(1, -1, 1)
        meta_vec = meta_encoder.transform([extract_metadata_feature(raw_data['metadata'])]).toarray()
        probabilities = model.predict([spec_vec, meta_vec])[0]
        prediction_binary = (probabilities >= config['prediction']['probability_threshold']).astype(int).reshape(1, -1)
        predicted_labels_tup = label_binarizer.inverse_transform(prediction_binary)
        print(f"\n✅ Prediksi (Hanya Model {spectrum_type.upper()}):")
        if predicted_labels_tup[0]:
            all_classes = list(label_binarizer.classes_)
            for label in predicted_labels_tup[0]:
                idx = all_classes.index(label)
                print(f"- {label} (Kepercayaan: {probabilities[idx]:.1%})")

    def _predict_compound_ensemble(ir_file_path, uv_file_path, config):
        paths = config['paths']
        print("🧠 Memuat semua model untuk ENSEMBLE...")
        base_model_ir = tf.keras.models.load_model(os.path.join(paths['saved_models_dir'], 'base_model_ir_final.keras'), compile=False)
        base_model_uv = tf.keras.models.load_model(os.path.join(paths['saved_models_dir'], 'base_model_uv_final.keras'), compile=False)
        meta_model = joblib.load(os.path.join(paths['saved_models_dir'], 'meta_model_final.joblib'))
        label_binarizer_ir, label_binarizer_uv = joblib.load(os.path.join(paths['saved_models_dir'], 'label_binarizer_ir.joblib')), joblib.load(os.path.join(paths['saved_models_dir'], 'label_binarizer_uv.joblib'))
        meta_encoder_ir, meta_encoder_uv = joblib.load(os.path.join(paths['saved_models_dir'], 'metadata_encoder_ir.joblib')), joblib.load(os.path.join(paths['saved_models_dir'], 'metadata_encoder_uv.joblib'))
        raw_data_ir, raw_data_uv = parse_jdx(ir_file_path), parse_jdx(uv_file_path)
        processed_df_ir, processed_df_uv = preprocess_spectrum(raw_data_ir, config, 'ir', True), preprocess_spectrum(raw_data_uv, config, 'uv', True)
        spec_vec_ir, spec_vec_uv = processed_df_ir['absorbance'].values.reshape(1, -1, 1), processed_df_uv['log_epsilon'].values.reshape(1, -1, 1)
        meta_vec_ir, meta_vec_uv = meta_encoder_ir.transform([extract_metadata_feature(raw_data_ir['metadata'])]).toarray(), meta_encoder_uv.transform([extract_metadata_feature(raw_data_uv['metadata'])]).toarray()
        prob_ir, prob_uv = base_model_ir.predict([spec_vec_ir, meta_vec_ir]), base_model_uv.predict([spec_vec_uv, meta_vec_uv])
        new_meta_feature = np.concatenate([prob_ir, prob_uv], axis=1)
        final_prediction_probas_list = meta_model.predict_proba(new_meta_feature)
        positive_probas = np.array([proba[0][1] for proba in final_prediction_probas_list])
        final_prediction_binary = (positive_probas >= config['prediction']['probability_threshold']).astype(int)
        num_ir_classes = len(label_binarizer_ir.classes_)
        pred_binary_ir, pred_binary_uv = final_prediction_binary[:num_ir_classes].reshape(1, -1), final_prediction_binary[num_ir_classes:].reshape(1, -1)
        probas_ir, probas_uv = positive_probas[:num_ir_classes], positive_probas[num_ir_classes:]
        predicted_labels_ir_tup, predicted_labels_uv_tup = label_binarizer_ir.inverse_transform(pred_binary_ir), label_binarizer_uv.inverse_transform(pred_binary_uv)
        print("\n✅ Prediksi Final (Model ENSEMBLE Gabungan):")
        print("\n--- Gugus Fungsi (IR) ---")
        if predicted_labels_ir_tup[0]:
            all_classes_ir = list(label_binarizer_ir.classes_)
            for label in predicted_labels_ir_tup[0]:
                idx = all_classes_ir.index(label)
                print(f"- {label} (Kepercayaan: {probas_ir[idx]:.1%})")
        print("\n--- Kromofor (UV-Vis) ---")
        if predicted_labels_uv_tup[0]:
            all_classes_uv = list(label_binarizer_uv.classes_)
            for label in predicted_labels_uv_tup[0]:
                idx = all_classes_uv.index(label)
                print(f"- {label} (Kepercayaan: {probas_uv[idx]:.1%})")

    # Logika pemilihan mode prediksi
    if args.ir and args.uv:
        print("Mode: ENSEMBLE (IR & UV disediakan)")
        _predict_compound_ensemble(args.ir, args.uv, config)
    elif args.ir:
        print("Mode: IR TUNGGAL (hanya IR disediakan)")
        _predict_single_spectrum(args.ir, 'ir', config)
    elif args.uv:
        print("Mode: UV TUNGGAL (hanya UV disediakan)")
        _predict_single_spectrum(args.uv, 'uv', config)