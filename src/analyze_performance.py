# analyze_performance.py

import yaml
import os
import numpy as np
import joblib
import pandas as pd
import argparse
from sklearn.metrics import recall_score, f1_score, multilabel_confusion_matrix

def analyze_results(config, spectrum_type):
    """
    Menganalisis hasil prediksi model untuk memberikan rekomendasi.
    """
    paths = config['paths']
    
    # Muat file-file yang diperlukan
    try:
        y_test = np.load(os.path.join(paths['reports_dir'], f'y_test_{spectrum_type}.npy'))
        y_pred = np.load(os.path.join(paths['reports_dir'], f'y_pred_{spectrum_type}.npy'))
        mlb = joblib.load(os.path.join(paths['saved_models_dir'], f'label_binarizer_{spectrum_type}.joblib'))
        class_names = mlb.classes_
    except FileNotFoundError:
        print(f"Error: File hasil (y_test_{spectrum_type}.npy, dll.) tidak ditemukan.")
        print("Pastikan Anda sudah menjalankan mode 'train' terlebih dahulu dengan kode modeling.py yang sudah diperbarui.")
        return

    print(f"\n--- Menganalisis Performa Model {spectrum_type.upper()} ---")

    # --- 1. Analisis Recall: Kelas mana yang aturannya perlu dilonggarkan? ---
    recalls = recall_score(y_test, y_pred, average=None, zero_division=0)
    low_recall_classes = sorted([(name, score) for name, score in zip(class_names, recalls) if score < 0.5], key=lambda x: x[1])

    print("\n## 🎯 Rekomendasi: Longgarkan Aturan (Recall Rendah)")
    print("Kelas-kelas ini paling sering dilewatkan oleh model. Pertimbangkan untuk menurunkan `min_prominence` di auto-labeler.")
    if not low_recall_classes:
        print("  -> Semua kelas memiliki recall di atas 50%. Hasil bagus!")
    else:
        for name, score in low_recall_classes:
            print(f"  - {name:<30} | Recall: {score:.2f}")

    # --- 2. Analisis F1-Score: Kelas mana yang paling sulit dipelajari? ---
    f1_scores = f1_score(y_test, y_pred, average=None, zero_division=0)
    hardest_classes = sorted([(name, score) for name, score in zip(class_names, f1_scores)], key=lambda x: x[1])

    print("\n## 📚 Kelas Paling Sulit Dipelajari (F1-Score Terendah)")
    print("Ini adalah kelas dengan performa keseluruhan terburuk, gabungan dari precision dan recall.")
    for name, score in hardest_classes[:5]: # Tampilkan 5 terburuk
        print(f"  - {name:<30} | F1-Score: {score:.2f}")

    # --- 3. Analisis Kesalahan: Bagaimana model membuat kesalahan? ---
    mlcm = multilabel_confusion_matrix(y_test, y_pred)
    
    # False Negatives (FN): Model seharusnya bilang "YA" tapi bilang "TIDAK"
    fn_counts = sorted([(class_names[i], mlcm[i][1, 0]) for i in range(len(class_names))], key=lambda x: x[1], reverse=True)
    
    # False Positives (FP): Model seharusnya bilang "TIDAK" tapi bilang "YA"
    fp_counts = sorted([(class_names[i], mlcm[i][0, 1]) for i in range(len(class_names))], key=lambda x: x[1], reverse=True)

    print("\n## ❗️ Kelas yang Paling Sering Dilewatkan (False Negatives Terbanyak)")
    print("Model kesulitan menemukan kelas-kelas ini. Solusi: perbanyak data atau longgarkan aturan label.")
    for name, count in fn_counts[:5]:
        print(f"  - {name:<30} | Dilewatkan sebanyak: {count} kali")

    print("\n## ❓ Kelas yang Paling Sering Salah Tebak (False Positives Terbanyak)")
    print("Model terlalu sering menebak kelas ini padahal salah. Solusi: perketat aturan label atau periksa kualitas label.")
    for name, count in fp_counts[:5]:
        print(f"  - {name:<30} | Salah tebak sebanyak: {count} kali")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analisis Performa Model Spektral")
    parser.add_argument("--type", choices=["ir", "uv"], required=True, help="Tipe model yang akan dianalisis (ir atau uv)")
    args = parser.parse_args()

    with open("configs/main_config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    analyze_results(config, args.type)