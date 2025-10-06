# src/auto_labeler.py (Versi Lengkap dengan Rules IR & UV)

import numpy as np
from scipy.signal import find_peaks
import os
# =================================================================
# == ATURAN UNTUK SPEKTROSKOPI INFRAMERAH (IR)                    ==
# =================================================================
SPECTRAL_RULES = [
    # --- Stretching (Ikatan Tunggal dengan Hidrogen) ---
    {"group": "alcohol_phenol_oh", "range": (3200, 3600), "min_prominence": 0.15, "vibration_type": "O-H stretch, broad"},
    {"group": "carboxylic_acid_oh_broad", "range": (2500, 3300), "min_prominence": 0.1, "vibration_type": "O-H stretch, very broad"},
    {"group": "amine_nh_primary", "range": (3300, 3500), "min_prominence": 0.05, "vibration_type": "N-H stretch (2 peaks)"},
    {"group": "amine_nh_secondary", "range": (3300, 3500), "min_prominence": 0.05, "vibration_type": "N-H stretch (1 peak)"},
    {"group": "amide_nh", "range": (3100, 3500), "min_prominence": 0.05, "vibration_type": "N-H stretch"},
    {"group": "alkane_ch_sp3", "range": (2850, 2960), "min_prominence": 0.1, "vibration_type": "sp³ C-H stretch"},
    {"group": "alkene_ch_sp2", "range": (3010, 3100), "min_prominence": 0.1, "vibration_type": "sp² C-H stretch"},
    {"group": "aromatic_ch", "range": (3000, 3100), "min_prominence": 0.05, "vibration_type": "Aromatic C-H stretch"},
    {"group": "alkyne_ch_sp", "range": (3300, 3320), "min_prominence": 0.05, "vibration_type": "sp C-H stretch, sharp"},
    {"group": "thiol_sh", "range": (2550, 2700), "min_prominence": 0.1, "vibration_type": "S-H stretch, weak"},
    {"group": "aldehyde_co", "range": (1720, 1740), "min_prominence": 0.2, "vibration_type": "C=O stretch"},
    {"group": "ketone_co", "range": (1705, 1725), "min_prominence": 0.2, "vibration_type": "C=O stretch"},
    {"group": "ester_co", "range": (1735, 1750), "min_prominence": 0.25, "vibration_type": "C=O stretch"},
    {"group": "amide_co", "range": (1640, 1690), "min_prominence": 0.25, "vibration_type": "C=O stretch"},
    {"group": "carboxylic_acid_co", "range": (1700, 1725), "min_prominence": 0.25, "vibration_type": "C=O stretch"},
    {"group": "alkene_cc", "range": (1620, 1680), "min_prominence": 0.1, "vibration_type": "C=C stretch"},
    {"group": "aromatic_cc", "range": (1450, 1600), "min_prominence": 0.1, "vibration_type": "Aromatic C=C stretch"},
    {"group": "alkyne_cc", "range": (2100, 2260), "min_prominence": 0.1, "vibration_type": "C≡C stretch"},
    {"group": "nitrile_cn", "range": (2210, 2260), "min_prominence": 0.1, "vibration_type": "C≡N stretch"},
    {"group": "isocyanate_nco", "range": (2250, 2270), "min_prominence": 0.1, "vibration_type": "N=C=O asymmetric stretch"},
    {"group": "amine_cn", "range": (1020, 1230), "min_prominence": 0.05, "vibration_type": "C-N stretch"},
    {"group": "amide_cn", "range": (1180, 1360), "min_prominence": 0.06, "vibration_type": "C-N stretch"},
    {"group": "ether_co", "range": (1050, 1150), "min_prominence": 0.04, "vibration_type": "C-O stretch"},
    {"group": "ester_co_c", "range": (1150, 1250), "min_prominence": 0.05, "vibration_type": "C-O stretch"},
    {"group": "anhydride_co", "range": (1740, 1850), "min_prominence": 0.13, "vibration_type": "C=O stretch (2 bands)"},
    {"group": "alkyl_halide_ccl", "range": (600, 800), "min_prominence": 0.03, "vibration_type": "C-Cl stretch"},
    {"group": "alkyl_halide_cbr", "range": (500, 650), "min_prominence": 0.03, "vibration_type": "C-Br stretch"},
    {"group": "alkyl_halide_cI", "range": (450, 600), "min_prominence": 0.05, "vibration_type": "C-I stretch"},
    {"group": "aromatic_out_of_plane_ch", "range": (690, 900), "min_prominence": 0.03, "vibration_type": "Aromatic C-H bend (out-of-plane)"},
    {"group": "alkene_out_of_plane_ch", "range": (900, 990), "min_prominence": 0.04, "vibration_type": "Alkene C-H bend (out-of-plane)"},
    {"group": "benzene_ring_breathing", "range": (1000, 1075), "min_prominence": 0.04, "vibration_type": "Ring breathing"},
    {"group": "nitro_no2_asym", "range": (1500, 1570), "min_prominence": 0.14, "vibration_type": "Asymmetric N-O stretch"},
    {"group": "nitro_no2_sym", "range": (1300, 1370), "min_prominence": 0.06, "vibration_type": "Symmetric N-O stretch"},
    {"group": "sulfone_so2", "range": (1300, 1350), "min_prominence": 0.08, "vibration_type": "S=O stretch"},
    {"group": "sulfate_so4", "range": (1050, 1150), "min_prominence": 0.04, "vibration_type": "S=O stretch"}
]

# =================================================================
# == ATURAN UNTUK SPEKTROSKOPI UV-VIS (BARU)                       ==
# =================================================================
UV_SPECTRAL_RULES = [
    # --- Kromofor Terisolasi ---
    {"group": "alkene_isolated", "range": (170, 195), "min_prominence": 0.15, "min_log_epsilon": 3.8, "comment": "Transisi π -> π* pada C=C terisolasi."},
    {"group": "carbonyl_pi_pi_star", "range": (180, 195), "min_prominence": 0.1, "min_log_epsilon": 3.0, "comment": "Transisi π -> π* yang kuat pada C=O."},
    {"group": "carbonyl_n_pi_star", "range": (270, 300), "min_prominence": 0.24, "min_log_epsilon": 2.92, "comment": "Transisi n -> π* yang lemah pada C=O."},
    # --- Sistem Dien Terkonjugasi ---
    {"group": "diene_conjugated_acyclic", "range": (215, 230), "min_prominence": 0.52, "min_log_epsilon": 2.59, "comment": "Transisi π -> π* untuk diena terkonjugasi."},
    {"group": "diene_conjugated_cyclic", "range": (250, 270), "min_prominence": 0.14, "min_log_epsilon": 2.45, "comment": "Diena terkonjugasi dalam cincin."},
    # --- Sistem Karbonil Terkonjugasi (Enon) ---
    {"group": "enone_pi_pi_star", "range": (210, 250), "min_prominence": 0.42, "min_log_epsilon": 2.92, "comment": "Transisi π -> π* (pita K) yang kuat pada enon (C=C-C=O)."},
    {"group": "enone_n_pi_star", "range": (310, 330), "min_prominence": 0.24, "min_log_epsilon": 1.72, "comment": "Transisi n -> π* (pita R) yang lemah, bergeser ke λ lebih panjang."},
    # --- Sistem Aromatik dan Substituennya ---
    {"group": "benzene_primary_E2_band", "range": (200, 210), "min_prominence": 0.03, "min_log_epsilon": 3.51, "comment": "Pita E2 primer benzena."},
    {"group": "benzene_secondary_B_band", "range": (250, 270), "min_prominence": 0.14, "min_log_epsilon": 2.45, "comment": "Pita B sekunder benzena (struktur halus)."},
    {"group": "phenol_or_aniline", "range": (270, 285), "min_prominence": 0.27, "min_log_epsilon": 2.86, "comment": "Gugus -OH atau -NH2 menggeser pita B."},
    {"group": "styrene_conjugated_aromatic", "range": (245, 260), "min_prominence": 0.21, "min_log_epsilon": 3.35, "comment": "Konjugasi cincin benzena dengan C=C."},
    {"group": "benzaldehyde_conjugated_aromatic", "range": (240, 255), "min_prominence": 0.24, "min_log_epsilon": 3.55, "comment": "Konjugasi cincin benzena dengan C=O."},
    {"group": "naphthalene_polycyclic_aromatic", "range": (300, 320), "min_prominence": 0.24, "min_log_epsilon": 1.67, "comment": "Sistem polisiklik aromatik seperti naftalena."},
]


# =================================================================
# == FUNGSI AUTO-LABELER UNTUK IR                                  ==
# =================================================================
def autogenerate_functional_groups(spectrum_df):
    # ... (kode fungsi ini tidak berubah) ...
    detected_groups = set()
    wavenumbers = spectrum_df['wavenumber'].values
    absorbances = spectrum_df['absorbance'].values
    all_peaks, properties = find_peaks(absorbances, prominence=0.02, width=1)
    for rule in SPECTRAL_RULES:
        for i, peak_idx in enumerate(all_peaks):
            wavenumber_at_peak = wavenumbers[peak_idx]
            prominence_at_peak = properties['prominences'][i]
            if rule['range'][0] <= wavenumber_at_peak <= rule['range'][1]:
                if prominence_at_peak >= rule['min_prominence']:
                    detected_groups.add(rule['group'])
                    break
    return list(detected_groups)

# =================================================================
# == FUNGSI AUTO-LABELER UNTUK UV-VIS                              ==
# =================================================================
def autogenerate_chromophores(spectrum_df):
    # ... (kode fungsi ini tidak berubah) ...
    detected_groups = set()
    wavelengths = spectrum_df['wavelength'].values
    log_epsilons = spectrum_df['log_epsilon'].values
    all_peaks, properties = find_peaks(log_epsilons, prominence=0.01, width=3)
    for rule in UV_SPECTRAL_RULES:
        for i, peak_idx in enumerate(all_peaks):
            wavelength_at_peak = wavelengths[peak_idx]
            prominence_at_peak = properties['prominences'][i]
            log_epsilon_at_peak = log_epsilons[peak_idx]
            if rule['range'][0] <= wavelength_at_peak <= rule['range'][1]:
                if (prominence_at_peak >= rule['min_prominence'] and log_epsilon_at_peak >= rule['min_log_epsilon']):
                    detected_groups.add(rule['group'])
                    break
    return list(detected_groups)