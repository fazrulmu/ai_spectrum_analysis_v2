# scripts/run_training.py (Logika untuk Pelatihan)

import os
import numpy as np
import joblib
import pandas as pd
from xgboost import XGBClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report

# Impor fungsi dari src
from data_processing import prepare_ir_dataset, prepare_uv_dataset
from modeling import train_base_model_cv

def _train_base_stage(config, spectrum_type):
    # ... (logika ini sama dengan fungsi train_base_stage dari jawaban sebelumnya)
    pass

def _train_meta_stage(config):
    # ... (logika ini sama dengan fungsi train_meta_stage dari jawaban sebelumnya)
    pass

def execute_training(args, config):
    """Fungsi utama yang dipanggil oleh main.py untuk menjalankan pelatihan."""
    
    # Fungsi _train_base_stage dan _train_meta_stage dari jawaban sebelumnya disalin ke sini
    def _train_base_stage(config, spectrum_type):
        print(f"🚀 Memulai Pelatihan TAHAP DASAR untuk model: {spectrum_type.upper()}")
        paths = config['paths']
        if spectrum_type == 'ir': X_spec, X_meta, y, groups = prepare_ir_dataset(config)
        else: X_spec, X_meta, y, groups = prepare_uv_dataset(config)
        oof_preds = train_base_model_cv(X_spec, X_meta, y, groups, config, spectrum_type)
        print(f"💾 Menyimpan hasil Out-of-Fold untuk {spectrum_type.upper()}...")
        np.save(os.path.join(paths['saved_models_dir'], f'{spectrum_type}_oof_preds.npy'), oof_preds)
        np.save(os.path.join(paths['saved_models_dir'], f'{spectrum_type}_y_true.npy'), y)
        np.save(os.path.join(paths['saved_models_dir'], f'{spectrum_type}_groups.npy'), groups)
        print(f"✅ Pelatihan TAHAP DASAR untuk {spectrum_type.upper()} selesai.")

    def _train_meta_stage(config):
        print("🚀 Memulai Pelatihan TAHAP META...")
        paths = config['paths']
        try:
            ir_oof_preds, uv_oof_preds = np.load(os.path.join(paths['saved_models_dir'], 'ir_oof_preds.npy')), np.load(os.path.join(paths['saved_models_dir'], 'uv_oof_preds.npy'))
            y_ir, groups_ir = np.load(os.path.join(paths['saved_models_dir'], 'ir_y_true.npy')), np.load(os.path.join(paths['saved_models_dir'], 'ir_groups.npy'))
            y_uv, groups_uv = np.load(os.path.join(paths['saved_models_dir'], 'uv_y_true.npy')), np.load(os.path.join(paths['saved_models_dir'], 'uv_groups.npy'))
        except FileNotFoundError as e:
            print(f"❌ Error: File OOF tidak ditemukan. Pastikan sudah menjalankan tahap 'base' untuk IR dan UV. File: {e.filename}")
            return
        df_ir, df_uv = pd.DataFrame({'cas': groups_ir, 'idx_ir': range(len(groups_ir))}), pd.DataFrame({'cas': groups_uv, 'idx_uv': range(len(groups_uv))})
        merged_df = pd.merge(df_ir, df_uv, on='cas', how='inner')
        ir_indices, uv_indices = merged_df['idx_ir'].values, merged_df['idx_uv'].values
        ir_oof_aligned, uv_oof_aligned = ir_oof_preds[ir_indices], uv_oof_preds[uv_indices]
        y_ir_aligned, y_uv_aligned = y_ir[ir_indices], y_uv[uv_indices]
        y_aligned = np.concatenate([y_ir_aligned, y_uv_aligned], axis=1)
        X_meta_features = np.concatenate([ir_oof_aligned, uv_oof_aligned], axis=1)
        meta_model = XGBClassifier(objective='binary:logistic', eval_metric='logloss', use_label_encoder=False, n_estimators=250, random_state=config['modeling']['random_state'])
        wrapper_model = MultiOutputClassifier(meta_model)
        wrapper_model.fit(X_meta_features, y_aligned)
        joblib.dump(wrapper_model, os.path.join(paths['saved_models_dir'], 'meta_model_final.joblib'))
        print("✅ Pelatihan TAHAP META selesai.")

    # Logika pemilihan stage
    if args.stage == "base":
        _train_base_stage(config, args.type)
    elif args.stage == "meta":
        _train_meta_stage(config)
    elif args.stage == "all":
        _train_base_stage(config, 'ir')
        _train_base_stage(config, 'uv')
        _train_meta_stage(config)