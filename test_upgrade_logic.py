import re
import numpy as np
# from scipy.signal import find_peaks, peak_widths

# ... (parsing logic remains) ...

print("\nTesting Peak Finding (SKIPPED due to env issues):")
# x = np.linspace(0, 100, 1000)
# y = np.sin(x) + np.random.normal(0, 0.1, 1000)
# peaks, _ = find_peaks(y, height=0.5)
# print(f"Found {len(peaks)} peaks.")

def parse_advanced_metadata(state_string, spectrum_type):
    """
    Parsing canggih untuk metadata STATE, mengekstrak matrix dan range.
    """
    if not state_string:
        return ('UNKNOWN', '', '')
    
    state_raw = str(state_string).upper()
    
    # 1. Deteksi State Utama
    state = 'UNKNOWN'
    if 'GAS' in state_raw:
        state = 'GAS'
    elif 'LIQUID' in state_raw:
        state = 'LIQUID'
    elif 'SOLID' in state_raw:
        state = 'SOLID'
    elif 'NEAT' in state_raw:
        state = 'NEAT'
    elif 'SOLUTION' in state_raw:
        state = 'SOLUTION'
        
    matrices = []
    ranges = []
    
    # Pola 1: [MATRIX] FOR [RANGE]
    pattern1 = r'([A-Z0-9]+)\s+FOR\s+(\d+[\s-]+\d+)'
    matches1 = re.findall(pattern1, state_raw)
    
    for mat, rng in matches1:
        rng_clean = rng.replace(' ', '')
        if mat not in ['AND', 'IN', 'FOR', 'OF', 'WITH']:
            if mat == 'S2': mat = 'CS2'
            matrices.append(mat)
            ranges.append(rng_clean)
            
    known_matrices = ['CCL4', 'CS2', 'KBR', 'NUJOL', 'FLUOROLUBE', 'HALOCARBON', 'CHCL3', 'N2', 'CSI', 'NEAT']
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

# Test Cases
test_strings = [
    "SOLUTION (10% IN CCl4 FOR 3800-1300, 10% IN CS2 FOR 1300-650, 10% IN CCl4 FOR 650-250 CM^-^1) VERSUS SOLVENT",
    "SOLID (SPLIT MULL, FLUOROLUBE FOR 3800-1330 AND NUJOL FOR 1330-400 CM ^-^1)",
    "SOLUTION SATURATED (CS2 FOR 2-15 microns)",
    "GAS (150 mmHg DILUTED TO A TOTAL PRESSURE OF 600 mmHg WITH N2)",
    "GAS"
]

print("Testing Metadata Parsing:")
for s in test_strings:
    print(f"Input: {s}")
    print(f"Output: {parse_advanced_metadata(s, 'IR')}")
    print("-" * 20)

print("\nTesting Peak Finding:")
x = np.linspace(0, 100, 1000)
y = np.sin(x) + np.random.normal(0, 0.1, 1000)
peaks, _ = find_peaks(y, height=0.5)
print(f"Found {len(peaks)} peaks.")
