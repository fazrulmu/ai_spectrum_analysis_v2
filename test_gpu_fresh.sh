#!/bin/bash
# This script runs the GPU verification in a fresh login shell
# to ensure group membership changes (render, video) are active

export HSA_OVERRIDE_GFX_VERSION=11.0.3
export HSA_ENABLE_SDMA=0

cd ~/Desktop/ai_spectrum_analysis_v2
source ~/venv-rocm/bin/activate
python3 verify_final.py
