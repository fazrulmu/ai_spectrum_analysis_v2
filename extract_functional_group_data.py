import pandas as pd
import numpy as np
import os
import json
from pathlib import Path
from tqdm import tqdm
import sys
import yaml

# --- Path Setup ---
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.data.data_processing import parse_jdx, preprocess_spectrum

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_structural_confidence(path):
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"❌ Structural confidence file not found: {path}")
        return {}

def load_cas_whitelist(path):
    if not os.path.exists(path):
        print(f"⚠️ Whitelist file not found: {path}")
        return set()
    with open(path, 'r') as f:
        return set(line.strip() for line in f if line.strip())

def detect_spectrum_type(file_path, metadata):
    # Priority 1: Filename/Folder conventions (if reliable)
    # Priority 2: Metadata
    dt = metadata.get('data type', '').upper()
    if 'INFRARED' in dt or 'IR' in dt: return 'IR'
    if 'UV' in dt or 'ULTRAVIOLET' in dt: return 'UV'
    
    # Fallback checks
    fname = str(file_path).upper()
    if '_IR_' in fname: return 'IR'
    if '_UV_' in fname: return 'UV'
    
    return 'UNKNOWN'

def extract_data(raw_dir, config_path, struct_conf_path, whitelist_path, output_path):
    print("🚀 Starting Functional Group Data Extraction...")
    
    # 1. Load Resources
    config = load_config(config_path)
    struct_data = load_structural_confidence(struct_conf_path)
    whitelist = load_cas_whitelist(whitelist_path)
    
    print(f"📚 Structural Data: {len(struct_data)} CAS entries")
    print(f"📋 Whitelist: {len(whitelist)} CAS numbers")
    
    # 2. Find Files
    all_files = list(Path(raw_dir).rglob('*.jdx'))
    print(f"📂 Found {len(all_files)} JDX files total.")
    
    extracted_records = []
    
    for file_path in tqdm(all_files, desc="Processing"):
        # --- Filter 1: CAS Whitelist ---
        # Assuming folder name is CAS, or we check metadata later. 
        # Checking folder first is faster.
        folder_cas = file_path.parent.name
        if whitelist and folder_cas not in whitelist:
            continue
            
        try:
            # Parse JDX
            raw_spectrum = parse_jdx(str(file_path))
            if raw_spectrum is None: continue
            
            metadata = raw_spectrum.get('metadata', {})
            cas_number = metadata.get('cas registry no', folder_cas)
            
            # Double check CAS in whitelist (in case folder name was wrong)
            if whitelist and cas_number not in whitelist:
                continue

            # --- Filter 2: IR Only ---
            spec_type = detect_spectrum_type(file_path, metadata)
            if spec_type != 'IR':
                continue
                
            # --- Preprocess ---
            # We use 'ir' config to ensure consistent grid
            processed_df = preprocess_spectrum(raw_spectrum, config, 'ir', normalize=True)
            if processed_df is None or processed_df.empty: continue
            
            # Get X and Y
            if 'wavenumber' in processed_df.columns:
                x_vals = processed_df['wavenumber'].values
                y_vals = processed_df['absorbance'].values
            else:
                # Should not happen for IR if preprocess succeeded, but safety check
                continue
                
            # --- ENFORCE DESCENDING X (High -> Low) ---
            # User Requirement: "range must same as preprocess_spectrum for ir wich is high x to low x"
            if len(x_vals) > 1 and x_vals[0] < x_vals[-1]:
                x_vals = x_vals[::-1]
                y_vals = y_vals[::-1]
            
            # --- Extract Functional Groups ---
            if cas_number in struct_data:
                groups_info = struct_data[cas_number].get('detected_functional_groups', {})
                
                for group_name, group_details in groups_info.items():
                    r_min = group_details.get('range_min')
                    r_max = group_details.get('range_max')
                    
                    if r_min is None or r_max is None: continue
                    
                    # Extract Slice
                    # Since X is descending (e.g. 4000 ... 400), 
                    # we want x where r_min <= x <= r_max
                    mask = (x_vals >= r_min) & (x_vals <= r_max)
                    
                    if not np.any(mask):
                        continue
                        
                    x_slice = x_vals[mask]
                    y_slice = y_vals[mask]
                    
                    # Ensure slice is also descending (it should be if original is descending)
                    # But just to be absolutely sure for the user output
                    sort_idx = np.argsort(x_slice)[::-1]
                    x_slice = x_slice[sort_idx]
                    y_slice = y_slice[sort_idx]
                    
                    record = {
                        'cas': cas_number,
                        'file_id': file_path.stem,
                        'group_name': group_name,
                        'range_min': r_min,
                        'range_max': r_max,
                        'x_values': x_slice.tolist(),
                        'y_values': y_slice.tolist()
                    }
                    extracted_records.append(record)
                    
        except Exception as e:
            # tqdm.write(f"Error processing {file_path.name}: {e}")
            continue
            
    # 3. Save Output
    print(f"📊 Extracted {len(extracted_records)} functional group samples.")
    
    if extracted_records:
        with open(output_path, 'w') as f:
            for rec in extracted_records:
                json.dump(rec, f)
                f.write('\n')
        print(f"✅ Saved to: {output_path}")
    else:
        print("⚠️ No data extracted.")

if __name__ == "__main__":
    RAW_DIR = "data/raw/nist_jdx"
    CONFIG = "main_config.yaml"
    STRUCT_CONF = "data/for_train/structural_confidence.json"
    WHITELIST = "uv_cas_list.txt"
    OUTPUT = "data/for_train/functional_group_spectral_data.jsonl"
    
    extract_data(RAW_DIR, CONFIG, STRUCT_CONF, WHITELIST, OUTPUT)
