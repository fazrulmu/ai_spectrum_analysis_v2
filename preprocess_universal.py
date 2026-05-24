import pandas as pd
import numpy as np
import os
import json
from pathlib import Path
from tqdm import tqdm
import sys
import yaml

# --- Pengaturan Path untuk Impor Modul Proyek ---
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.data.data_processing import parse_jdx, preprocess_spectrum

import re
from scipy.signal import find_peaks, peak_widths

def load_structural_confidence(path):
    """
    Load structural confidence data for functional group labeling.
    Returns a dictionary mapping CAS number to functional groups.
    """
    try:
        with open(path, 'r') as f:
            data = json.load(f)
        
        # Create a simplified lookup: CAS -> list of group names
        lookup = {}
        for cas, info in data.items():
            groups = list(info.get('detected_functional_groups', {}).keys())
            lookup[cas] = {
                'smiles': info.get('smiles', ''),
                'groups': groups
            }
        return lookup
    except FileNotFoundError:
        print(f"⚠️ Warning: Structural confidence file not found at {path}")
        return {}
    except Exception as e:
        print(f"⚠️ Error loading structural confidence: {e}")
        return {}

def parse_advanced_metadata(state_string, spectrum_type):
    """
    Parsing canggih untuk metadata STATE, mengekstrak matrix dan range.
    """
    if not state_string:
        return ('UNKNOWN', '', '')
    
    state_raw = str(state_string).upper()
    
    # 1. Deteksi State Utama
    state = 'UNKNOWN'
    if 'GAS' in state_raw: state = 'GAS'
    elif 'LIQUID' in state_raw: state = 'LIQUID'
    elif 'SOLID' in state_raw: state = 'SOLID'
    elif 'NEAT' in state_raw: state = 'NEAT'
    elif 'SOLUTION' in state_raw: state = 'SOLUTION'
        
    # 2. Ekstrak Matrix dan Range
    matrices = []
    ranges = []
    
    known_matrices = ['CCL4', 'CS2', 'KBR', 'NUJOL', 'FLUOROLUBE', 'HALOCARBON', 'CHCL3', 'N2', 'CSI', 'NEAT']
    
    pattern1 = r'([A-Z0-9]+)\s+FOR\s+(\d+[\s-]+\d+)'
    matches1 = re.findall(pattern1, state_raw)
    
    for mat, rng in matches1:
        rng_clean = rng.replace(' ', '')
        if mat not in ['AND', 'IN', 'FOR', 'OF', 'WITH']:
            if mat == 'S2': mat = 'CS2'
            matrices.append(mat)
            ranges.append(rng_clean)
            
    if not matrices:
        for km in known_matrices:
            if km in state_raw:
                if km not in matrices:
                    matrices.append(km)
                    
    if state == 'GAS' and not matrices:
        matrices.append('GAS_INERT')
        
    matrix_sample_str = '[' + ','.join(matrices) + ']' if matrices else ''
    abs_matrix_str = '[' + ','.join(ranges) + ']' if ranges else ''
    
    return state, matrix_sample_str, abs_matrix_str

def detect_spectrum_type(file_path):
    file_str = str(file_path).upper()
    if '_IR_' in file_str or 'INFRARED' in file_str: return 'IR'
    elif '_UV_' in file_str or 'UV-VIS' in file_str or 'ULTRAVIOLET' in file_str: return 'UV'
    else: return 'UNKNOWN'

def calculate_spectrum_features(x, y, spectrum_type):
    features = {}
    peaks, properties = find_peaks(y, height=0.01, distance=10)
    num_peaks = len(peaks)
    features['num_peaks'] = num_peaks
    
    fwhm = 0.0
    if num_peaks > 0:
        widths = peak_widths(y, peaks, rel_height=0.5)
        fwhm = float(np.mean(widths[0]))
    features['fwhm'] = fwhm
    
    if spectrum_type == 'UV':
        max_idx = np.argmax(y)
        features['lambda_max'] = float(x[max_idx])
        features['log_epsilon'] = float(y[max_idx])
        
    return features

def create_universal_dataset(raw_data_dir: str, config_path: str, output_path: str, structural_conf_path: str):
    print("🚀 Memulai proses pembuatan dataset universal (JSON Output)...")

    # --- 1. Muat Konfigurasi & Structural Confidence ---
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"❌ Config not found: {config_path}")
        return

    structural_data = load_structural_confidence(structural_conf_path)
    print(f"📚 Loaded structural confidence data for {len(structural_data)} CAS numbers.")

    # --- 2. Proses Setiap File Spektrum ---
    json_records = []
    
    file_path_map = {}
    for path in Path(raw_data_dir).rglob('*.jdx'):
        file_path_map[path.stem] = path
    
    print(f"🔄 Memproses {len(file_path_map)} file spektrum...")
    
    # --- Load CAS Whitelist (Optional) ---
    whitelist_cas = set()
    if os.path.exists('uv_cas_list.txt'):
        print("ℹ️  Loading CAS whitelist from uv_cas_list.txt...")
        with open('uv_cas_list.txt', 'r') as f:
            whitelist_cas = set(line.strip() for line in f if line.strip())
        print(f"   Whitelist loaded: {len(whitelist_cas)} CAS numbers.")

    debug_count = 0
    for file_id, file_path in tqdm(file_path_map.items(), desc="Processing"):
        # Check CAS before parsing (optimization)
        folder_cas = file_path.parent.name
        if whitelist_cas and folder_cas not in whitelist_cas:
             continue

        try:
            # a. Parse JDX
            raw_spectrum = parse_jdx(str(file_path))
            if raw_spectrum is None: continue
            
            metadata = raw_spectrum.get('metadata', {})
            
            # b. Deteksi Tipe
            spectrum_type = detect_spectrum_type(file_path)
            if spectrum_type == 'UNKNOWN':
                dt = metadata.get('data type', '').upper()
                if 'INFRARED' in dt or 'IR' in dt: spectrum_type = 'IR'
                elif 'UV' in dt: spectrum_type = 'UV'
            
            # c. Ekstrak Metadata Lanjut
            state_raw = metadata.get('state', '')
            state, matrix_sample, abs_matrix = parse_advanced_metadata(state_raw, spectrum_type)
            
            cas_number = metadata.get('cas registry no', '')
            if not cas_number: cas_number = file_path.parent.name
            
            # d. Preprocess Spectrum (Resampling)
            spec_type_key = spectrum_type.lower() if spectrum_type in ['IR', 'UV'] else 'ir'
            processed_df = preprocess_spectrum(raw_spectrum, config, spec_type_key, normalize=True)
            if processed_df is None or processed_df.empty: continue
            
            # Tentukan kolom x dan y
            if 'wavenumber' in processed_df.columns:
                x_col, y_col = 'wavenumber', 'absorbance'
            elif 'wavelength' in processed_df.columns:
                x_col, y_col = 'wavelength', 'log_epsilon'
            else: continue
            
            x_vals = processed_df[x_col].values
            y_vals = processed_df[y_col].values
            
            # --- ENFORCE DESCENDING ORDER FOR X (High -> Low) ---
            # This applies to both IR (4000->400) and UV (e.g. 800->200) as requested.
            if len(x_vals) > 1 and x_vals[0] < x_vals[-1]:
                x_vals = x_vals[::-1]
                y_vals = y_vals[::-1]
            
            # e. Hitung Fitur Tambahan
            features = calculate_spectrum_features(x_vals, y_vals, spectrum_type)
            
            # f. Get Labels from Structural Confidence
            struct_info = structural_data.get(cas_number, {})
            labels = struct_info.get('groups', [])
            smiles = struct_info.get('smiles', '')
            
            # g. Susun Data Record (JSON Structure)
            record = {
                'file_id': file_id,
                'cas_number': cas_number,
                'smiles': smiles,
                'spectrum_type': spectrum_type,
                'x': x_vals.tolist(), # Convert numpy array to list for JSON
                'y': y_vals.tolist(),
                'labels': labels,
                'metadata': {
                    'state': state,
                    'matrix_sample': matrix_sample,
                    'abs_matrix': abs_matrix,
                    'state_raw': state_raw,
                    'num_peaks': int(features.get('num_peaks', 0)),
                    'fwhm': float(features.get('fwhm', 0.0)),
                    'lambda_max': float(features.get('lambda_max', 0.0)) if spectrum_type == 'UV' else None,
                    'log_epsilon': float(features.get('log_epsilon', 0.0)) if spectrum_type == 'UV' else None
                }
            }
            
            json_records.append(record)

        except Exception as e:
            tqdm.write(f"⚠️ Error {file_path.name}: {e}")

    # --- 3. Simpan JSONL ---
    print(f"📊 Menyimpan {len(json_records)} record ke JSONL...")
    
    if not json_records:
        print("⚠️ Peringatan: Tidak ada data yang berhasil diproses.")
        return

    # Pisahkan IR dan UV
    ir_records = [r for r in json_records if r['spectrum_type'] == 'IR']
    uv_records = [r for r in json_records if r['spectrum_type'] == 'UV']
    
    if ir_records:
        base_name = str(Path(output_path).with_suffix(''))
        ir_out = f"{base_name}_IR.jsonl"
        with open(ir_out, 'w') as f:
            for record in ir_records:
                json.dump(record, f)
                f.write('\n')
        print(f"💾 Dataset IR disimpan: {ir_out} ({len(ir_records)} records)")
        
    if uv_records:
        base_name = str(Path(output_path).with_suffix(''))
        uv_out = f"{base_name}_UV.jsonl"
        with open(uv_out, 'w') as f:
            for record in uv_records:
                json.dump(record, f)
                f.write('\n')
        print(f"💾 Dataset UV disimpan: {uv_out} ({len(uv_records)} records)")

    print("✅ Proses selesai.")

if __name__ == '__main__':
    RAW_JDX_DIR = "data/raw/nist_jdx"
    CONFIG_FILE = "main_config.yaml"
    OUTPUT_FILE = "data/for_train/universal_training_dataset.jsonl"
    STRUCTURAL_CONF_FILE = "data/for_train/structural_confidence.json"
    
    Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)
    create_universal_dataset(RAW_JDX_DIR, CONFIG_FILE, OUTPUT_FILE, STRUCTURAL_CONF_FILE)