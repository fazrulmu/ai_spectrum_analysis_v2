# src/run_full_prediction.py

import os
import yaml
import joblib
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image

# Impor dari RDKit untuk menggambar struktur
from rdkit import Chem
from rdkit.Chem import Draw

# Impor fungsi-fungsi yang sudah ada
from data.data_processing import parse_jdx, preprocess_spectrum, extract_metadata_feature
from data.harness_spectra import get_smiles_from_cas, get_molform_from_cas
from labeling.auto_labeler import load_rules_from_json # Impor fungsi pemuat aturan

# --- PERBAIKAN: Muat aturan IR secara eksplisit dari file JSON ---
def get_ir_rules():
    """Helper untuk memuat aturan IR."""
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    ir_rules_path = os.path.join(project_root, 'ir_rules.json')
    return load_rules_from_json(ir_rules_path)

IR_RULES = get_ir_rules()


def execute_full_prediction(args, config):
    """
    Menjalankan pipeline prediksi lengkap: gugus fungsi, pola substitusi,
    dan menghasilkan laporan visual yang komprehensif.
    """
    paths = config['paths']
    file_path = args.file

    if not os.path.exists(file_path):
        print(f"❌ Error: File tidak ditemukan di '{file_path}'")
        return

    # --- 1. Muat Semua Model dan Encoder ---
    print("🧠 Memuat semua model dan encoder...")
    try:
        # Model gugus fungsi
        model_ir = tf.keras.models.load_model(os.path.join(paths['saved_models_dir'], 'base_model_ir_final.keras'), compile=False)
        meta_encoder_ir = joblib.load(os.path.join(paths['saved_models_dir'], 'metadata_encoder_ir.joblib'))
        label_binarizer_ir = joblib.load(os.path.join(paths['saved_models_dir'], 'label_binarizer_ir.joblib'))
        
        # Model pola substitusi
        model_pattern = joblib.load(os.path.join(paths['saved_models_dir'], 'pattern_recognition_model.joblib'))
        encoder_pattern = joblib.load(os.path.join(paths['saved_models_dir'], 'pattern_label_encoder.joblib'))
    except FileNotFoundError as e:
        print(f"❌ Error: Gagal memuat file model. Pastikan semua model sudah dilatih. Detail: {e}")
        return

    # --- 2. Proses Spektrum Input ---
    print(f"🔬 Memproses spektrum: {os.path.basename(file_path)}")
    raw_data = parse_jdx(file_path)
    if not raw_data:
        print("❌ Gagal mem-parsing file JDX.")
        return

    # Pra-pemrosesan untuk model gugus fungsi (spektrum penuh)
    processed_df_full = preprocess_spectrum(raw_data, config, 'ir', normalize=True)
    spec_vec_full = processed_df_full['absorbance'].values.reshape(1, -1, 1)
    meta_vec_full = meta_encoder_ir.transform([extract_metadata_feature(raw_data['metadata'])]).toarray()

    # Pra-pemrosesan untuk model pola (hanya daerah sidik jari)
    FINGERPRINT_START, FINGERPRINT_STOP, FINGERPRINT_POINTS = 650, 900, 250
    fingerprint_grid = np.linspace(FINGERPRINT_START, FINGERPRINT_STOP, FINGERPRINT_POINTS)
    temp_df = preprocess_spectrum(raw_data, config, 'ir', normalize=False)
    fingerprint_data = np.interp(fingerprint_grid, temp_df['wavenumber'], temp_df['absorbance'])
    min_val, max_val = fingerprint_data.min(), fingerprint_data.max()
    fingerprint_norm = (fingerprint_data - min_val) / (max_val - min_val) if max_val > min_val else fingerprint_data

    # --- 3. Jalankan Semua Prediksi ---
    print("🤖 Menjalankan prediksi AI...")
    # Prediksi gugus fungsi
    probas_ir = model_ir.predict([spec_vec_full, meta_vec_full])[0]
    pred_binary_ir = (probas_ir >= config['prediction']['probability_threshold']).astype(int).reshape(1, -1)
    predicted_labels_ir = label_binarizer_ir.inverse_transform(pred_binary_ir)[0]

    # Prediksi pola substitusi
    pred_encoded_pattern = model_pattern.predict(fingerprint_norm.reshape(1, -1))
    predicted_pattern = encoder_pattern.inverse_transform(pred_encoded_pattern)[0]

    # --- 4. Dapatkan Struktur Kimia ---
    print("🧪 Mengambil struktur kimia...")
    cas_no = raw_data['metadata'].get('cas registry no', '').strip() or os.path.basename(os.path.dirname(file_path))
    
    # --- PERBAIKAN: Logika pengambilan molform yang lebih andal ---
    molform_from_file = raw_data['metadata'].get('molform', '')
    if molform_from_file:
        molform = molform_from_file
    else:
        molform = get_molform_from_cas(cas_no) if cas_no else "N/A"

    smiles = get_smiles_from_cas(cas_no) if cas_no else None
    mol_image = None
    if smiles and "not_found" not in smiles and "error" not in smiles:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            mol_image = Draw.MolToImage(mol, size=(250, 250))

    # --- 5. Buat Laporan Visual Gabungan ---
    print("📊 Membuat laporan visual...")
    fig, ax = plt.subplots(figsize=(18, 9))

    # Plot spektrum utama
    ax.plot(processed_df_full['wavenumber'], processed_df_full['absorbance'], color='black', linewidth=1.5, label='Spektrum IR')

    # Sorot daerah penting
    ax.axvspan(900, 650, color='lightblue', alpha=0.4, label='Daerah Sidik Jari (Pola Substitusi)')
    ax.axvspan(2000, 1665, color='lightgreen', alpha=0.4, label='Daerah Overtone')

    # Anotasi puncak gugus fungsi yang terdeteksi
    for label in predicted_labels_ir:
        rule = next((r for r in label_binarizer_ir.classes_ if r == label), None)
        # Logika sederhana untuk menempatkan anotasi di tengah rentang aturan
        # Ini bisa dipercanggih dengan deteksi puncak aktual
        rule_def = next((r for r in label_binarizer_ir.classes_ if r == label), None)
        if rule_def:
            # Cari puncak tertinggi di rentang aturan
            # Gunakan variabel IR_RULES yang sudah dimuat, bukan config
            rule_range = next((r['range'] for r in IR_RULES if r['group'] == label), None)
            if rule_range is None: continue # Lewati jika aturan tidak ditemukan
            region = processed_df_full[(processed_df_full['wavenumber'] >= min(rule_range)) & (processed_df_full['wavenumber'] <= max(rule_range))]
            if not region.empty:
                peak_idx = region['absorbance'].idxmax()
                peak_wavenumber = region.loc[peak_idx, 'wavenumber']
                peak_height = region.loc[peak_idx, 'absorbance']
                ax.plot(peak_wavenumber, peak_height, 'x', color='red', markersize=8)
                ax.text(peak_wavenumber, peak_height + 0.03, label, rotation=45, ha='left', fontsize=10, color='darkred')

    # Pengaturan plot utama
    ax.set_title(f"Laporan Analisis Spektrum Otomatis untuk: {raw_data['metadata'].get('title', 'N/A')}", fontsize=18, pad=20)
    ax.set_xlabel("Wavenumber (cm⁻¹)", fontsize=12)
    ax.set_ylabel("Absorbance (Normalized)", fontsize=12)
    ax.invert_xaxis()
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc='upper left')

    # Buat kotak teks untuk hasil prediksi
    text_results = "--- HASIL ANALISIS AI ---\n\n"
    text_results += f"Rumus Molekul: {molform}\n\n"
    text_results += "Prediksi Gugus Fungsi:\n"
    if predicted_labels_ir:
        for label in predicted_labels_ir:
            text_results += f"- {label}\n"
    else:
        text_results += "- Tidak ada yang terdeteksi.\n"
    
    text_results += "\nPrediksi Pola Substitusi:\n"
    text_results += f"- {predicted_pattern}\n"

    fig.text(0.75, 0.85, text_results, transform=ax.transAxes, fontsize=11,
             verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))

    # Tambahkan gambar struktur molekul jika ada
    if mol_image:
        # Buat axes baru di dalam figure untuk gambar
        # [left, bottom, width, height] dalam koordinat figure (0-1)
        img_axes = fig.add_axes([0.78, 0.15, 0.2, 0.2], anchor='SE', zorder=1)
        img_axes.imshow(mol_image)
        img_axes.axis('off') # Sembunyikan sumbu

    plt.tight_layout(rect=[0, 0, 0.95, 1]) # Beri ruang untuk teks dan gambar

    # Simpan atau tampilkan plot
    if args.output:
        output_path = args.output
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=150)
        print(f"\n✅ Laporan visual disimpan di: {output_path}")
    else:
        plt.show()