#!/usr/bin/env python3
"""
Test script untuk memverifikasi fungsi ekstraksi metadata
"""
import sys
from pathlib import Path

# Setup path
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Import fungsi dari preprocess_universal
from preprocess_universal import parse_state_metadata, detect_spectrum_type

# Test 1: parse_state_metadata
print("=" * 60)
print("TEST 1: parse_state_metadata")
print("=" * 60)

test_states = [
    "GAS (HEATING PARAFORMALDEHYDE; CONCENTRATION UNKNOWN)",
    "SOLID (KBR PELLET)",
    "LIQUID (CCL4 SOLUTION)",
    "SOLUTION (CHCL3)",
    "NEAT",
    "",
    "SOLID (NUJOL MULL)"
]

for state_str in test_states:
    state, matrix = parse_state_metadata(state_str)
    print(f"Input: '{state_str}'")
    print(f"  → State: {state}, Matrix: {matrix}")
    print()

# Test 2: detect_spectrum_type
print("=" * 60)
print("TEST 2: detect_spectrum_type")
print("=" * 60)

test_paths = [
    "50-00-0_IR_GAS_(HEATING_PARAFORMALDEHYDE_0.jdx",
    "123-45-6_UV_SOLUTION.jdx",
    "data/raw/nist_jdx/50-00-0/50-00-0_IR_GAS_(HEATING_PARAFORMALDEHYDE_0.jdx",
    "some_INFRARED_spectrum.jdx",
    "some_ULTRAVIOLET_spectrum.jdx",
    "unknown_file.jdx"
]

for path in test_paths:
    spec_type = detect_spectrum_type(path)
    print(f"Path: {path}")
    print(f"  → Type: {spec_type}")
    print()

print("=" * 60)
print("✅ Test selesai!")
print("=" * 60)
