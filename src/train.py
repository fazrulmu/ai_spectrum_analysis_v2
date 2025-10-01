# train.py (Entry Point Utama untuk Pelatihan)

import os
import yaml
import numpy as np
import joblib
import pandas as pd
from xgboost import XGBClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
import warnings

# Impor fungsi-fungsi dari src
from src.data_processing import prepare_ir_dataset, prepare_uv_dataset
from src.modeling import train_base_model_cv

warnings.filterwarnings("ignore", category=UserWarning)

def main():
    print("🚀 Memulai alur kerja pelatihan ensemble (IR & UV)...")
    
    with open('main_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    paths = config['paths']
    os.makedirs(paths['saved_models_dir'], exist_ok=True)
    os.makedirs(paths['reports_dir'], exist_ok=True)
    
    print("\n[Tahap 1/5] Memuat dan memproses dataset...")
    X_spec_ir, X_meta_ir, y_ir, groups_ir = prepare_ir_dataset(config)
    X_spec_uv, X_meta_uv, y_uv, groups_uv = prepare_uv_dataset(config)
    
    print("\n[Tahap 2/5] Menyelaraskan sampel antara dataset IR dan UV...")
    df_ir = pd.DataFrame({'cas': groups_ir, 'idx_ir': range(len(groups_ir))})
    df_uv = pd.DataFrame({'cas': groups_uv, 'idx_uv': range(len(groups_uv))})
    merged_df = pd.merge(df_ir, df_uv, on='cas', how='inner')
    
    if len(merged_df) == 0:
        raise ValueError("Tidak ada sampel yang cocok antara dataset IR dan UV.")
    print(f"Ditemukan {len(merged_df)} sampel yang cocok untuk dilatih.")
    
    ir_indices = merged_df['idx_ir'].values
    uv_indices = merged_df['idx_uv'].values
    
    X_spec_ir_aligned, X_meta_ir_aligned = X_spec_ir[ir_indices], X_meta_ir[ir_indices]
    X_spec_uv_aligned, X_meta_uv_aligned = X_spec_uv[uv_indices], X_meta_uv[uv_indices]
    groups_aligned = merged_df['cas'].values
    
    y_ir_aligned, y_uv_aligned = y_ir[ir_indices], y_uv[uv_indices]
    y_aligned = np.concatenate([y_ir_aligned, y_uv_aligned], axis=1)
    
    print("\n[Tahap 3/5] Melatih model-model dasar dengan Cross-Validation...")
    ir_oof_preds = train_base_model_cv(X_spec_ir_aligned, X_meta_ir_aligned, y_ir_aligned, groups_aligned, config, 'ir')
    uv_oof_preds = train_base_model_cv(X_spec_uv_aligned, X_meta_uv_aligned, y_uv_aligned, groups_aligned, config, 'uv')

    print("\n[Tahap 4/5] Melatih Meta-Model (XGBoost)...")
    X_meta_features = np.concatenate([ir_oof_preds, uv_oof_preds], axis=1)
    meta_model = XGBClassifier(objective='binary:logistic', eval_metric='logloss', use_label_encoder=False, n_estimators=250, random_state=config['modeling']['random_state'])
    wrapper_model = MultiOutputClassifier(meta_model)
    wrapper_model.fit(X_meta_features, y_aligned)
    joblib.dump(wrapper_model, os.path.join(paths['saved_models_dir'], 'meta_model_final.joblib'))
    
    print("\n[Tahap 5/5] Mengevaluasi dan menyimpan laporan...")
    y_pred_meta = wrapper_model.predict(X_meta_features)
    label_binarizer_ir = joblib.load(os.path.join(paths['saved_models_dir'], 'label_binarizer_ir.joblib'))
    label_binarizer_uv = joblib.load(os.path.join(paths['saved_models_dir'], 'label_binarizer_uv.joblib'))
    combined_class_names = list(label_binarizer_ir.classes_) + list(label_binarizer_uv.classes_)
    report = classification_report(y_aligned, y_pred_meta, target_names=combined_class_names, zero_division=0)
    
    print("\n--- Laporan Klasifikasi Final Ensemble Model (IR & UV) ---")
    print(report)
    
    report_path = os.path.join(paths['reports_dir'], 'final_ensemble_report.txt')
    with open(report_path, 'w') as f: f.write(report)
    print(f"✅ Laporan final disimpan di {report_path}")
    print("\n🏁 Alur kerja pelatihan ensemble selesai!")

if __name__ == "__main__":
    main()