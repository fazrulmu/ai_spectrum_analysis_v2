#!/usr/bin/env python3
"""
Demo sederhana untuk mengekstrak metadata dari satu file JDX
"""
import sys
from pathlib import Path

# Setup path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.data.data_processing import parse_jdx

# Test dengan file yang kita sudah lihat
test_file = "data/raw/nist_jdx/50-00-0/50-00-0_IR_GAS_(HEATING_PARAFORMALDEHYDE_0.jdx"

print("=" * 70)
print(f"Membaca file: {test_file}")
print("=" * 70)

try:
    result = parse_jdx(test_file)
    
    if result:
        metadata = result.get('metadata', {})
        
        print("\n📋 METADATA YANG DIEKSTRAK:")
        print("-" * 70)
        
        # Print metadata penting
        important_keys = ['title', 'data type', 'state', 'cas registry no', 
                         'xunits', 'yunits', 'origin', 'class']
        
        for key in important_keys:
            value = metadata.get(key, 'N/A')
            print(f"{key.upper():20s}: {value}")
        
        # Test parse state
        state_raw = metadata.get('state', '')
        print("\n🔍 PARSE STATE:")
        print("-" * 70)
        print(f"Raw STATE: {state_raw}")
        
        # Manual parse state seperti di fungsi kita
        state_string = str(state_raw).upper()
        
        # Deteksi state
        state = 'UNKNOWN'
        if 'GAS' in state_string:
            state = 'GAS'
        elif 'LIQUID' in state_string:
            state = 'LIQUID'
        elif 'SOLID' in state_string:
            state = 'SOLID'
        
        # Deteksi matrix
        matrix_sample = ''
        matrix_keywords = ['KBR', 'CCL4', 'CHCL3', 'N2', 'S2']
        for keyword in matrix_keywords:
            if keyword in state_string:
                matrix_sample = keyword
                break
        
        # Fallback state logic
        if state == 'UNKNOWN' and state_raw:
            state = state_raw
            print(f"State fallback triggered! New State: {state}")

        print(f"Parsed STATE: {state}")
        print(f"Matrix Sample: {matrix_sample if matrix_sample else '(none)'}")
        
        # CAS Number extraction
        cas_number = metadata.get('cas registry no', '')
        if not cas_number:
            # Mocking file path parent for demo
            cas_number = Path(test_file).parent.name
        print(f"CAS Number: {cas_number}")
        
        # Detect spectrum type from filename
        print("\n📊 SPECTRUM TYPE:")
        print("-" * 70)
        if '_IR_' in test_file:
            print("Detected: IR")
        elif '_UV_' in test_file:
            print("Detected: UV")
        else:
            data_type = metadata.get('data type', '').upper()
            if 'INFRARED' in data_type:
                print("Detected from metadata: IR")
            elif 'UV' in data_type:
                print("Detected from metadata: UV")
        
        print("\n✅ Metadata berhasil diekstrak!")
        
    else:
        print("❌ Gagal parse file")
        
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("=" * 70)
