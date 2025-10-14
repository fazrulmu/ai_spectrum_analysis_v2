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
import yaml
from scipy import sparse
from scipy.sparse.linalg import spsolve

import datetime



from scipy.spatial import ConvexHull

from src.auto_labeler import autogenerate_functional_groups, autogenerate_chromophores 

# Paste this entire function at the end of src/data_processing.py





    # (lanjutan kode robust parser yang sebelumnya kamu pasang)
    
def load_and_standardize_spectrum(
    file_path,
    target_resolution: float = 2.0,
    x_range: tuple = (4000.0, 400.0),
    baseline_correction: bool = True,
    normalize_intensity: bool = True,
    smoothing: bool = True
):
    """
    Pipeline penuh untuk membaca, menyeragamkan, dan menstandarkan spektrum IR/UV.

    Parameters
    ----------
    file_path : str
        Path ke file .jdx atau DataFrame.
    target_resolution : float
        Resolusi target X (misal 2 cm⁻¹ untuk IR).
    x_range : tuple
        Rentang X yang diinginkan (default: 4000–400 cm⁻¹).
    baseline_correction : bool
        Jika True, kurangi baseline (min intensity).
    normalize_intensity : bool
        Jika True, skala intensitas ke 0–1.
    smoothing : bool
        Jika True, lakukan smoothing ringan (Savitzky–Golay).

    Returns
    -------
    dict
        {
            "meta": {info file, tanggal, dsb},
            "df": DataFrame uniform (x, y),
            "numpy": array [x, y],
            "params": konfigurasi pemrosesan
        }
    """

    # --- Coba parse file
    try:
        from jcamp import jcamp_readfile
        jdx = jcamp_readfile(file_path)
        x = np.array(jdx.get("x", jdx.get("wavenumber")))
        y = np.array(jdx.get("y", jdx.get("transmittance", jdx.get("absorbance"))))
    except Exception as e:
        raise RuntimeError(f"Gagal membaca file {file_path}: {e}")

    if x is None or y is None:
        raise ValueError(f"File {file_path} tidak berisi data X/Y valid")

    # --- Pastikan urutan X menurun (IR biasanya 4000 → 400)
    if x[0] < x[-1]:
        x = x[::-1]
        y = y[::-1]

    # --- Buat grid seragam
    start, end = x_range
    x_uniform = np.arange(start, end - target_resolution, -target_resolution)

    # --- Interpolasi (cubic)
    f = interp1d(x, y, kind="cubic", fill_value="extrapolate")
    y_uniform = f(x_uniform)

    # --- Baseline correction (optional)
    if baseline_correction:
        y_uniform = y_uniform - np.min(y_uniform)

    # --- Normalisasi intensitas (optional)
    if normalize_intensity:
        y_uniform = y_uniform / np.max(np.abs(y_uniform))

    # --- Smoothing ringan (opsional)
    if smoothing:
        from scipy.signal import savgol_filter
        try:
            y_uniform = savgol_filter(y_uniform, window_length=11, polyorder=3)
        except Exception:
            pass

    df_uniform = pd.DataFrame({"x": x_uniform, "y": y_uniform})

    return {
        "meta": {
            "source_file": os.path.basename(file_path),
            "created_at": datetime.now().isoformat(),
            "resolution_target": target_resolution,
            "range": x_range,
        },
        "df": df_uniform,
        "numpy": np.column_stack((x_uniform, y_uniform)),
        "params": {
            "baseline_correction": baseline_correction,
            "normalize_intensity": normalize_intensity,
            "smoothing": smoothing,
        },
    }


# Ganti fungsi yang ada di src/data_processing.py dengan ini

def standardize_for_pattern_recognition(raw_data):
    """
    Menyiapkan spektrum IR untuk VISUALISASI dan LABELING MANUAL.
    TIDAK melakukan resampling.
    Output selalu Transmitans (puncak ke bawah) pada grid data asli.
    """
    if (
        raw_data is None or
        'x' not in raw_data or
        'y' not in raw_data or
        len(raw_data['x']) == 0 or
        len(raw_data['y']) == 0
    ):
        return None

    x, y, metadata = raw_data['x'], raw_data['y'], raw_data['metadata']
    df = pd.DataFrame({'x': x, 'y': y}).dropna()
    df_sorted = df.groupby('x', as_index=False).mean().sort_values(by='x').reset_index(drop=True)
    x_clean, y_clean = df_sorted['x'].values, df_sorted['y'].values

    # --- Logika Konversi ke Transmitans ---
    y_units = metadata.get('yunits', '').lower()
    if 'transmittance' in y_units:
        # Jika sudah transmittance, normalisasi ke rentang 0-1
        y_as_transmittance = y_clean / 100.0
    else: 
        # Jika absorbans, konversi ke transmittance
        y_as_transmittance = 10**(-y_clean)
        
    # Pastikan rentang data valid (0 hingga 1) dan lakukan normalisasi sederhana
    y_as_transmittance = np.clip(y_as_transmittance, 0, 1)

    # --- Logika Standardisasi Sumbu-X ---
    x_units = metadata.get('xunits', '').lower()
    if 'micrometer' in x_units:
        x_clean[x_clean == 0] = 1e-9
        x_standard = 10000 / x_clean
    else:
        x_standard = x_clean
        
    # Pastikan sumbu-x selalu menurun sesuai konvensi IR
    if x_standard[0] < x_standard[-1]:
        x_final, y_final = x_standard[::-1], y_as_transmittance[::-1]
    else:
        x_final, y_final = x_standard, y_as_transmittance

    # --- PERBAIKAN UTAMA: Kembalikan dictionary dengan DataFrame yang benar ---
    full_spectrum_df = pd.DataFrame({'wavenumber': x_final, 'transmittance': y_final})
    
    # Bagian lain dari fungsi ini (overtone, fingerprint) bisa ditambahkan di sini jika perlu
    # Untuk saat ini, kita hanya butuh spektrum penuh
    return {'full_spectrum': full_spectrum_df}
def augment_spectrum(spectrum_data, spectrum_type='ir'):
    """
    Applies a series of augmentations to a single spectrum to create a new,
    realistic training example.

    Args:
        spectrum_data (np.ndarray): A 1D numpy array representing the spectrum's y-values (absorbance/log_epsilon).
        spectrum_type (str): 'ir' or 'uv'. This allows for different noise levels.

    Returns:
        np.ndarray: The augmented spectrum data.
    """
    augmented_spectrum = spectrum_data.copy()

    # --- 1. Add Random Noise ---
    # Simulates electronic detector noise. IR spectra are often noisier.
    if spectrum_type == 'ir':
        # Higher noise level for IR
        noise_level = np.random.uniform(0.001, 0.015)
    else: # uv
        # Lower noise level for the smoother UV spectra
        noise_level = np.random.uniform(0.0005, 0.005)

    noise = np.random.normal(0, noise_level, augmented_spectrum.shape)
    augmented_spectrum += noise

    # --- 2. Baseline Shift ---
    # Simulates baseline drift during measurement.
    # We can add a simple vertical shift or a slight tilt.
    if np.random.rand() < 0.5: # Apply this augmentation 50% of the time
        baseline_shift = np.random.uniform(-0.05, 0.05)
        augmented_spectrum += baseline_shift
    else:
        # Add a slight tilt to the baseline
        tilt_factor = np.random.uniform(-0.0001, 0.0001)
        tilt = tilt_factor * np.arange(len(augmented_spectrum))
        augmented_spectrum += tilt

    # --- 3. Intensity Scaling ---
    # Simulates variations in sample concentration or instrument sensitivity.
    if np.random.rand() < 0.7: # Apply this augmentation 70% of the time
        scaling_factor = np.random.uniform(0.9, 1.1)
        augmented_spectrum *= scaling_factor
        
    # --- 4. Horizontal (Wavenumber/Wavelength) Shift ---
    # Simulates slight instrument calibration differences.
    if np.random.rand() < 0.5: # Apply this augmentation 50% of the time
        shift_amount = np.random.randint(-3, 4) # Shift by -3, -2, -1, 0, 1, 2, or 3 points
        augmented_spectrum = np.roll(augmented_spectrum, shift_amount)

    # Ensure data is clipped between 0 and a max value to avoid negative absorbance
    augmented_spectrum = np.clip(augmented_spectrum, 0, 2.0)

    return augmented_spectrum

def load_processed_spectrum(file_path):
    """
    Memuat data spektrum yang sudah diproses dari file CSV.
    Fungsi inilah yang dicari oleh visualization.py.
    """
    if not os.path.exists(file_path):
        print(f"Error: File data yang diproses tidak ditemukan di {file_path}")
        return None
    return pd.read_csv(file_path)

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




from jcamp import jcamp_readfile

def parse_jdx(file_path):
    """
    Membaca file JCAMP-DX (.jdx) dari berbagai sumber (NIST, Sigma, FTIR) dan 
    mengeluarkan hasil dengan format seragam:
    { 'x': np.array, 'y': np.array, 'metadata': dict }

    Fungsi ini robust terhadap format berbeda (##XYDATA, ##PEAK TABLE, ##XYPOINTS).
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File tidak ditemukan: {file_path}")

    def _clean_line(line):
        # Hilangkan simbol atau pemisah aneh
        return line.strip().replace(',', ' ').replace(';', ' ').replace('\t', ' ')

    try:
        # Coba baca dengan library jcamp
        jdx = jcamp_readfile(file_path)
        x = np.array(jdx.get("x", jdx.get("wavenumber", [])), dtype=float)
        y = np.array(jdx.get("y", jdx.get("transmittance", jdx.get("absorbance", []))), dtype=float)
        metadata = {
            "title": jdx.get("title", os.path.basename(file_path)),
            "xunits": jdx.get("xunits", "1/cm"),
            "yunits": jdx.get("yunits", "transmittance"),
            "data_type": jdx.get("data type", "unknown"),
            "xfactor": float(jdx.get("xfactor", 1.0)),
            "yfactor": float(jdx.get("yfactor", 1.0)),
        }
        # Apply XFACTOR and YFACTOR if present
        x = x * metadata["xfactor"]
        y = y * metadata["yfactor"]
        if len(x) > 0 and len(y) > 0:
            return {"x": x, "y": y, "metadata": metadata}

    except Exception:
        # Jika gagal, lakukan parsing manual
        x, y = [], []
        metadata = {}
        with open(file_path, "r", errors="ignore") as f:
            lines = f.readlines()

        data_started = False
        for line in lines:
            line = line.strip()
            if not line:
                continue

            if line.startswith("##"):
                # Metadata JCAMP
                parts = line.split("=", 1)
                if len(parts) == 2:
                    key = parts[0].strip(" #").lower()
                    val = parts[1].strip()
                    metadata[key] = val
                if "##xydata" in line.lower() or "##peak table" in line.lower():
                    data_started = True
                continue

            # Ambil data XY jika sudah di bagian data
            if data_started:
                cleaned = _clean_line(line)
                nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", cleaned)
                if len(nums) >= 2:
                    try:
                        x.append(float(nums[0]))
                        y.append(float(nums[1]))
                    except:
                        pass

        x, y = np.array(x), np.array(y)
        # Apply XFACTOR and YFACTOR if present in metadata (manual parse)
        xfactor = float(metadata.get("xfactor", 1.0))
        yfactor = float(metadata.get("yfactor", 1.0))
        x = x * xfactor
        y = y * yfactor
        if len(x) == 0 or len(y) == 0:
            raise ValueError("Gagal mem-parse data XY dari file JCAMP")

        # Deteksi unit secara otomatis jika tidak tersedia
        if "xunits" not in metadata:
            metadata["xunits"] = "1/cm"
        if "yunits" not in metadata:
            metadata["yunits"] = "transmittance" if np.max(y) <= 1.5 else "absorbance"

        metadata["title"] = metadata.get("title", os.path.basename(file_path))
        metadata["data_type"] = metadata.get("data type", "manual parse")

        return {"x": x, "y": y, "metadata": metadata}


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
    """
    Menyiapkan spektrum untuk TRAINING.
    Melakukan konversi ke Absorbans, resampling ke grid standar, dan normalisasi.
    Output selalu Absorbans (puncak ke atas).
    """
    if not data or 'x' not in data or 'y' not in data or len(data['x']) < 20:
        return None

    x, y, metadata = data['x'], data['y'], data['metadata']
    df = pd.DataFrame({'x': x, 'y': y}).dropna()
    if df.empty: return None
        
    df_sorted = df.groupby('x', as_index=False).mean().sort_values(by='x').reset_index(drop=True)
    x_clean, y_clean = df_sorted['x'].values, df_sorted['y'].values

    if spectrum_type == 'ir':
        grid_config = config['preprocessing']['ir_grid']
        grid = np.linspace(grid_config['stop'], grid_config['start'], grid_config['num_points'])
        x_col_name, output_col_name = 'wavenumber', 'absorbance'
        
        y_units = metadata.get('yunits', '').lower()
        if 'transmittance' in y_units or np.median(y_clean) > 1.1:
            y_in_absorbance = 2 - np.log10(np.clip(y_clean, 1e-9, 100))
        else:
            y_in_absorbance = y_clean
        
        x_units = metadata.get('xunits', '').lower()
        if 'micrometer' in x_units:
            x_clean[x_clean == 0] = 1e-9
            x_standard = 10000 / x_clean
        else:
            x_standard = x_clean
            
        if x_standard[0] > x_standard[-1]:
            x_final, y_final = x_standard[::-1], y_in_absorbance[::-1]
        else:
            x_final, y_final = x_standard, y_in_absorbance
    
    # ... (logika untuk UV tidak berubah)
    elif spectrum_type == 'uv':
        grid_config = config['preprocessing']['uv_grid']
        grid = np.linspace(grid_config['start'], grid_config['stop'], grid_config['num_points'])
        x_col_name, output_col_name = 'wavelength', 'log_epsilon'
        # ... (logika konversi unit UV)
        y_final = y_clean # Disederhanakan, asumsikan sudah absorbans
        x_final = x_clean

    else:
        raise ValueError("Tipe spektrum tidak valid")

    f = interp1d(x_final, y_final, bounds_error=False, fill_value=0.0)
    y_resampled = f(grid)
    y_processed = np.clip(y_resampled, 0, None)

    if normalize:
        max_val = np.max(y_processed)
        if max_val > 0: y_out = y_processed / max_val
        else: y_out = y_processed
    else:
        y_out = y_processed
    
    return pd.DataFrame({x_col_name: grid, output_col_name: y_out})


# -------------------------------
# Metadata feature
# -------------------------------
def extract_metadata_feature(metadata):
    """
    Ambil metadata umum (tanpa mempertimbangkan fase fisik)
    agar hasil spektrum tidak bias oleh keadaan padatan/cair/gas.
    """
    keys = ['instrument', 'operator', 'date', 'title', 'source']
    features = []
    for k in keys:
        val = metadata.get(k, '').strip().lower()
        features.append(val if val else 'unknown')
    return features


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



# =================================================================
# == BLOK MAIN UNTUK ORCHESTRATOR                              ==
# =================================================================
def main(output_dir=".", config_path="main_config.yaml"):
    """
    Fungsi utama untuk menjalankan pemrosesan data IR dan UV.
    """
    print("--- Menjalankan Modul Pemrosesan Data ---")
    
    # Memuat konfigurasi

    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: File konfigurasi '{config_path}' tidak ditemukan.")
        return

    # Menentukan direktori output spesifik untuk setiap dataset
    output_dir_ir = os.path.join(output_dir, "ir_dataset")
    output_dir_uv = os.path.join(output_dir, "uv_dataset")
    os.makedirs(output_dir_ir, exist_ok=True)
    os.makedirs(output_dir_uv, exist_ok=True)

    # Memproses dataset IR
    try:
        print("\nMemulai pemrosesan dataset IR...")
        prepare_ir_dataset(config, output_dir=output_dir_ir)
    except Exception as e:
        print(f"Gagal memproses dataset IR: {e}")

    # Memproses dataset UV
    try:
        print("\nMemulai pemrosesan dataset UV-Vis...")
        prepare_uv_dataset(config, output_dir=output_dir_uv)
    except Exception as e:
        print(f"Gagal memproses dataset UV-Vis: {e}")
        
    print("--- Modul Pemrosesan Data Selesai ---")

if __name__ == '__main__':
    """
    Blok ini akan dieksekusi jika file dijalankan secara langsung.
    """
    default_folder = "default_output/data_processing"
    print(f"Menjalankan {__file__} secara mandiri untuk pengujian...")
    main(output_dir=default_folder)
