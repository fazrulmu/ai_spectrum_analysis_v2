# src/data_processing.py (Versi FINAL dengan fallback UV & IR)

import jcamp
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
import re
import os
import glob
from tqdm import tqdm
from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer
import joblib
from scipy.sparse.linalg import spsolve
from scipy import sparse
from scipy.ndimage import grey_opening
from scipy.signal import find_peaks, peak_widths

from scipy import sparse
from scipy.sparse.linalg import spsolve



from src.auto_labeler import autogenerate_functional_groups, autogenerate_chromophores


from scipy.spatial import ConvexHull



def smooth_signal(y, window_length=11, polyorder=3):
    """
    Smoothing spektrum dengan Savitzky-Golay filter.
    - window_length: harus ganjil, biasanya 7–15.
    - polyorder: 2 atau 3 biasanya cukup.
    """
    if len(y) < window_length:
        return y
    return savgol_filter(y, window_length=window_length, polyorder=polyorder)



def poly_baseline(x, y, order=3):
    """
    Hitung baseline menggunakan fitting polynomial sederhana.
    """
    coeffs = np.polyfit(x, y, deg=order)
    baseline = np.polyval(coeffs, x)
    return baseline

def morph_baseline(y, size=101):
    """
    Baseline correction dengan morphological opening (grey opening).
    Parameter:
        size: window smoothing (ganjil, lebih besar = baseline lebih halus).
    """
    size = min(size, len(y)//2 * 2 + 1)  # pastikan ganjil & <= len(y)
    baseline = grey_opening(y, size=size)
    return baseline

def estimate_morph_size(y, target_factor=3):
    """
    Estimasi ukuran structuring element untuk morph baseline
    berdasarkan lebar puncak median.
    """
    peaks, _ = find_peaks(y, prominence=np.max(y)*0.02 if np.max(y)>0 else 0.01)
    if len(peaks) == 0:
        return 101
    results = peak_widths(y, peaks, rel_height=0.5)
    widths = results[0]
    median_w = int(np.median(widths))
    size = max(31, min(len(y)//2, median_w * target_factor | 1))  # pastikan ganjil
    return size

def choose_baseline_and_correct(x, y_resampled, morph_size=101, poly_order=3):
    """
    Pilih baseline terbaik:
    - Morphological baseline (default)
    - Fallback ke Hybrid Poly+ALS bila morph over-correct
    """
    # baseline morphological
    baseline_morph = morph_baseline(y_resampled, size=morph_size)
    corrected_morph = np.clip(y_resampled - baseline_morph, 0, None)

    # integrity check
    orig_peaks, _ = find_peaks(y_resampled, prominence=np.max(y_resampled)*0.02)
    morph_peaks, _ = find_peaks(corrected_morph, prominence=np.max(corrected_morph)*0.02 if corrected_morph.max()>0 else 0.01)

    peak_frac = len(morph_peaks)/len(orig_peaks) if len(orig_peaks)>0 else 1.0

    # fallback jika morph menghapus terlalu banyak puncak
    if peak_frac < 0.7:
        baseline = hybrid_poly_als_baseline(x, y_resampled, poly_order=poly_order)
        corrected = np.clip(y_resampled - baseline, 0, None)
        method = "Hybrid Poly+ALS (fallback)"
    else:
        baseline = baseline_morph
        corrected = corrected_morph
        method = "Morphological"

    return baseline, corrected, method

def hybrid_poly_als_baseline(x, y, poly_order=2):
    """
    Baseline correction gabungan:
    1. Polynomial fit (untuk cekungan global/parabola)
    2. ALS modular (untuk baseline halus)
    """
    # Step 1: Polynomial baseline
    baseline_poly = poly_baseline(x, y, order=poly_order)
    residual = y - baseline_poly

    # Step 2: ALS pada residual
    baseline_als_part = baseline_als(residual)

    # Gabungan baseline
    baseline_final = baseline_poly + baseline_als_part
    return baseline_final


def rubberband_baseline(x, y):
    """Baseline dengan convex hull (rubberband)."""
    pts = np.column_stack([x, y])
    hull = ConvexHull(pts)
    mask = np.sort(hull.vertices)
    baseline = np.interp(x, x[mask], y[mask])
    return baseline

def hybrid_baseline(x, y):
    """
    Hybrid baseline correction:
    1. Rubberband baseline → hilangkan cekungan besar
    2. ALS modular → perbaikan halus
    """
    # Step 1: rubberband
    baseline_rb = rubberband_baseline(x, y)

    # Step 2: ALS pada data yang sudah dikoreksi rubberband
    residual = y - baseline_rb
    baseline_als_part = baseline_als(residual)

    # Gabungan
    baseline_final = baseline_rb + baseline_als_part
    return baseline_final


# -------------------------------
# Augmentasi Data (BARU)
# -------------------------------

# =================================================================
# == PARSER JDX MENGGUNAKAN LIBRARY KHUSUS YANG ANDAL            ==
# =================================================================
def parse_jdx(file_path):
    try:
        data = jcamp.jcamp_readfile(file_path)
        metadata = {k.lower().replace(' ', '_').replace('/','_'): v for k, v in data.items() if not isinstance(v, (list, np.ndarray))}
        x_vals = np.array(data.get('x', []), dtype=float)
        y_vals = np.array(data.get('y', []), dtype=float)
        if x_vals.size == 0 or y_vals.size == 0 or x_vals.size != y_vals.size:
            return None
        return {"x": x_vals, "y": y_vals, "metadata": metadata}
    except Exception:
        return None

# =================================================================
# == FUNGSI KOREKSI BASELINE (STABIL)                            ==
# =================================================================
def baseline_als(y, lam_range=(1e5, 1e7), p_range=(0.01, 0.1), niter=10):
    """
    Adaptive Asymmetric Least Squares baseline correction.
    - lam (λ): smoothing parameter (semakin besar, baseline makin kaku).
    - p: asymmetry parameter (semakin besar, baseline makin permisif).
    
    Fungsi ini memilih lam & p otomatis berdasarkan karakteristik spektrum.
    """
    L = len(y)
    if L < 10:
        return np.zeros_like(y)  # fallback kalau data terlalu pendek

    # --- analisis spektrum ---
    y_range = np.max(y) - np.min(y)
    y_std = np.std(y)

    # --- pilih parameter secara adaptif ---
    # lam: dipengaruhi oleh "keramaian/noise" → std besar → lam besar (lebih kaku)
    lam = np.clip(1e5 * (1 + y_std * 5), lam_range[0], lam_range[1])

    # p: dipengaruhi oleh rentang intensitas → range besar → p kecil (lebih nempel ke baseline)
    if y_range > 2:
        p = 0.01
    elif y_range > 0.5:
        p = 0.03
    else:
        p = 0.07
    p = np.clip(p, p_range[0], p_range[1])

    # --- ALS baseline ---
    D = sparse.diags([1, -2, 1], [-1, 0, 1], shape=(L, L - 2))
    D = lam * D.dot(D.T)
    w = np.ones(L)
    for _ in range(niter):
        W = sparse.spdiags(w, 0, L, L)
        Z = W + D + sparse.eye(L) * 1e-9
        z = spsolve(Z, w * y)
        w = p * (y > z) + (1 - p) * (y < z)
    return z

# =================================================================
# == FUNGSI PREPROCESSING UTAMA DENGAN LOGIKA ADAPTIF YANG BENAR   ==
# =================================================================
def preprocess_spectrum(data, config, spectrum_type, normalize=True):
    # ... (Bagian awal fungsi: parsing, cleaning, unit standardization, dll. tidak berubah)
    if not data or 'x' not in data or 'y' not in data or len(data['x']) < 20: return None
    x, y, metadata = data['x'], data['y'], data['metadata']
    df = pd.DataFrame({'x': x, 'y': y}).dropna()
    if df.empty or len(df) < 20: return None
    df_cleaned = df.groupby('x', as_index=False).mean()
    df_sorted = df_cleaned.sort_values(by='x').reset_index(drop=True)
    if len(df_sorted) < 20: return None
    x_clean, y_clean = df_sorted['x'].values, df_sorted['y'].values
    
    x_units = metadata.get('xunits', '').lower()
    y_units = metadata.get('yunits', '').lower()
    
    if spectrum_type == 'ir':
        if 'micrometer' in x_units or 'micron' in x_units:
            x_clean[x_clean == 0] = 1e-9; x_standard = 10000 / x_clean
        else: x_standard = x_clean
        is_transmittance = 'transmittance' in y_units or np.median(y_clean) > 1.1
        if is_transmittance: y_standard = 2 - np.log10(np.clip(y_clean, 1e-9, 100))
        else: y_standard = y_clean
        if x_standard[0] < x_standard[-1]: x_final, y_final = x_standard[::-1], y_standard[::-1]
        else: x_final, y_final = x_standard, y_standard
        grid_config = config['preprocessing']['ir_grid']
        output_col_name = 'absorbance'; x_col_name = 'wavenumber'
    
    elif spectrum_type == 'uv':
        is_transmittance_uv = 'transmittance' in y_units or np.median(y_clean) > 1.1
        if is_transmittance_uv: y_final = 2 - np.log10(np.clip(y_clean, 1e-9, 100))
        elif 'absorbance' in y_units: y_final = y_clean
        else: y_final = y_clean
        x_final = x_clean
        grid_config = config['preprocessing']['uv_grid']
        output_col_name = 'log_epsilon'; x_col_name = 'wavelength'
    else: raise ValueError("Tipe spektrum tidak valid")
        
    grid = np.linspace(grid_config['start'], grid_config['stop'], grid_config['num_points'])
    f = interp1d(x_final, y_final, bounds_error=False, fill_value=0.0)
    y_resampled = f(grid)
    
    # --- Smoothing standar untuk semua tipe data ---
    if len(y_resampled) > 7: y_smoothed = savgol_filter(y_resampled, window_length=7, polyorder=2)
    else: y_smoothed = y_resampled
    
    state = metadata.get('state', '').lower()
    
    # --- BLOK KOREKSI BASELINE DENGAN PENANGANAN KHUSUS UNTUK FASA GAS ---
        # --- BLOK KOREKSI BASELINE (STABIL & ADAPTIF) ---
    try:
        y_for_baseline_calc = y_resampled.copy()

        # cek seberapa cekung baseline dengan fitting polinomial orde 2
        baseline_test = poly_baseline(grid, y_for_baseline_calc, order=2)
        residual_var = np.var(y_for_baseline_calc - baseline_test)

        if residual_var > 0.05:
            # baseline cekung → pakai hybrid poly + ALS
            baseline = hybrid_poly_als_baseline(grid, y_for_baseline_calc, poly_order=2)
        else:
            # baseline normal → cukup ALS modular
            baseline = baseline_als(y_for_baseline_calc)

        # kurangkan baseline
        y_corrected = y_for_baseline_calc - baseline

        # smoothing ringan
        if len(y_corrected) > 7:
            y_final_processed = savgol_filter(y_corrected, window_length=7, polyorder=2)
        else:
            y_final_processed = y_corrected

        # clip ke nol
        y_final_processed = np.clip(y_final_processed, 0, None)

    except Exception:
        y_final_processed = y_smoothed

    
    # --- Normalisasi (tidak berubah) ---
    output_df = pd.DataFrame({x_col_name: grid})
    if normalize:
        min_val, max_val = np.min(y_final_processed), np.max(y_final_processed)
        if max_val > min_val: y_out = (y_final_processed - min_val) / (max_val - min_val)
        else: y_out = y_final_processed
    else: y_out = y_final_processed
    
    output_df[output_col_name] = y_out
    return output_df
# -------------------------------
# Metadata feature
# -------------------------------
def extract_metadata_feature(metadata):
    state = metadata.get('state', 'unknown').lower()
    if 'solid' in state:
        phase = 'solid'
    elif 'gas' in state:
        phase = 'gas'
    elif 'liquid' in state:
        phase = 'liquid'
    elif 'solution' in state:
        phase = 'solution'
    else:
        phase = 'unknown'
    return [phase]

# -------------------------------
# Dataset IR
# -------------------------------
def prepare_ir_dataset(config):
    paths = config['paths']

    all_jdx = glob.glob(os.path.join(paths['raw_data_dir'], '**', '*.jdx'), recursive=True)
    all_files = [f for f in all_jdx if re.search(r'(ftir|ft-ir|ir)', os.path.basename(f), re.IGNORECASE)]

    spectra, metas, labels, groups = [], [], [], []
    failed_files = []  # <-- simpan daftar gagal

    print("Mempersiapkan dataset IR...")
    print("Ditemukan file IR:", len(all_files))

    for file_path in tqdm(all_files):
        cas_no = os.path.basename(os.path.dirname(file_path))
        raw_data = parse_jdx(file_path)
        if not raw_data:
            failed_files.append(file_path)  # simpan file gagal
            continue

        processed_df = preprocess_spectrum(raw_data, config, 'ir', normalize=True)
        if processed_df is None or processed_df.empty:
            failed_files.append(file_path)
            continue

        if 'absorbance' not in processed_df.columns:
            failed_files.append(file_path)
            continue

        gen_labels = autogenerate_functional_groups(processed_df)
        if not gen_labels:
            gen_labels = ["no_functional_group"]

        try:
            spectra.append(processed_df['absorbance'].values)
            metas.append(extract_metadata_feature(raw_data['metadata']))
            labels.append(gen_labels)
            groups.append(cas_no)
        except Exception:
            failed_files.append(file_path)

    # --- simpan daftar gagal ke file ---
    if failed_files:
        log_path = os.path.join(paths['saved_models_dir'], "failed_ir_files.txt")
        with open(log_path, "w") as f:
            f.write("\n".join(failed_files))
        print(f"⚠️ {len(failed_files)} file IR gagal diparse. Daftar disimpan di {log_path}")

    if not spectra:
        raise ValueError("Tidak ada file IR valid ditemukan. Semua gagal diparse atau kosong.")

    # Encoder metadata & label
    meta_encoder = OneHotEncoder(handle_unknown='ignore')
    X_meta = meta_encoder.fit_transform(metas).toarray()

    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(labels)

    print("\nKelas IR ditemukan:", mlb.classes_)
    os.makedirs(paths['saved_models_dir'], exist_ok=True)
    joblib.dump(meta_encoder, os.path.join(paths['saved_models_dir'], 'metadata_encoder_ir.joblib'))
    joblib.dump(mlb, os.path.join(paths['saved_models_dir'], 'label_binarizer_ir.joblib'))
    print("Encoder IR disimpan.")

    X_spec = np.array(spectra).reshape(len(spectra), -1, 1)
    return X_spec, X_meta, y, np.array(groups)


# -------------------------------
# Dataset UV
# -------------------------------
def prepare_uv_dataset(config):
    paths = config['paths']

    all_jdx = glob.glob(os.path.join(paths['raw_data_dir'], '**', '*.jdx'), recursive=True)
    all_files = [f for f in all_jdx if re.search(r'uv[\-\s]?vis', os.path.basename(f), re.IGNORECASE)]

    spectra, metas, labels, groups = [], [], [], []
    failed_files = []  # <-- simpan daftar gagal

    print("Mempersiapkan dataset UV-Vis...")
    print("Ditemukan file UV:", len(all_files))

    for file_path in tqdm(all_files):
        cas_no = os.path.basename(os.path.dirname(file_path))
        raw_data = parse_jdx(file_path)
        if not raw_data:
            failed_files.append(file_path)
            continue

        unnormalized_df = preprocess_spectrum(raw_data, config, 'uv', normalize=False)
        if unnormalized_df is None or unnormalized_df.empty:
            failed_files.append(file_path)
            continue

        gen_labels = autogenerate_chromophores(unnormalized_df)
        if not gen_labels:
            gen_labels = ["no_chromophore"]

        normalized_df = preprocess_spectrum(raw_data, config, 'uv', normalize=True)
        if normalized_df is None or normalized_df.empty:
            failed_files.append(file_path)
            continue

        try:
            spectra.append(normalized_df['log_epsilon'].values)
            metas.append(extract_metadata_feature(raw_data['metadata']))
            labels.append(gen_labels)
            groups.append(cas_no)
        except Exception:
            failed_files.append(file_path)

    # --- simpan daftar gagal ke file ---
    if failed_files:
        log_path = os.path.join(paths['saved_models_dir'], "failed_uv_files.txt")
        with open(log_path, "w") as f:
            f.write("\n".join(failed_files))
        print(f"⚠️ {len(failed_files)} file UV gagal diparse. Daftar disimpan di {log_path}")

    if not spectra:
        raise ValueError("Tidak ada file UV-Vis valid ditemukan. Semua gagal diparse atau kosong.")

    # Encoder metadata & label
    meta_encoder = OneHotEncoder(handle_unknown='ignore')
    X_meta = meta_encoder.fit_transform(metas).toarray()

    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(labels)

    print("\nKelas UV ditemukan:", mlb.classes_)
    os.makedirs(paths['saved_models_dir'], exist_ok=True)
    joblib.dump(meta_encoder, os.path.join(paths['saved_models_dir'], 'metadata_encoder_uv.joblib'))
    joblib.dump(mlb, os.path.join(paths['saved_models_dir'], 'label_binarizer_uv.joblib'))
    print("Encoder UV disimpan.")

    X_spec = np.array(spectra).reshape(len(spectra), -1, 1)
    return X_spec, X_meta, y, np.array(groups)
