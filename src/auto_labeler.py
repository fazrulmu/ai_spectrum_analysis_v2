# src/auto_labeler.py (Versi Lengkap dengan Rules IR & UV)

import numpy as np
from scipy.signal import find_peaks
import os
import re


def get_elements_from_molform(molform_string):
    """
    Mengekstrak elemen-elemen unik dari string rumus molekul (MOLFORM).
    Contoh: 'C6H6O' -> {'C', 'H', 'O'}
    """
    if not isinstance(molform_string, str):
        return set() # Kembalikan set kosong jika input tidak valid
        
    # Menemukan semua simbol elemen (huruf kapital diikuti huruf kecil opsional)
    elements = re.findall('[A-Z][a-z]*', molform_string)
    return set(elements)
# =================================================================
# == ATURAN UNTUK SPEKTROSKOPI INFRAMERAH (IR)                    ==
# =================================================================
SPECTRAL_RULES = [
    # --- Stretching (Ikatan Tunggal dengan Hidrogen) ---
    

    
    
    {"group": "amine_nh_secondary", "range": (3300, 3500), "min_prominence": 0.05, "vibration_type": "N-H stretch (1 peak)"},
    
    {"group": "alkane_ch_sp3", "range": (2850, 2960), "min_prominence": 0.1, "vibration_type": "sp³ C-H stretch"},
    {"group": "alkene_ch_sp2", "range": (3010, 3100), "min_prominence": 0.1, "vibration_type": "sp² C-H stretch"},
    
    {"group": "alkyne_ch_sp", "range": (3300, 3320), "min_prominence": 0.05, "vibration_type": "sp C-H stretch, sharp"},
    {"group": "thiol_sh", "range": (2550, 2700), "min_prominence": 0.1, "vibration_type": "S-H stretch, weak"},
    
    

    
    
    
    
    {"group": "carboxylic_acid_co", "range": (1700, 1725), "min_prominence": 0.25, "vibration_type": "C=O stretch"},
    {"group": "nitrile_cn", "range": (2210, 2260), "min_prominence": 0.1, "vibration_type": "C≡N stretch"},
    {"group": "isocyanate_nco", "range": (2250, 2270), "min_prominence": 0.1, "vibration_type": "N=C=O asymmetric stretch"},
    
    #yang terdapat dalam database
    
    {"group": "ester_co_c", "range": (1150, 1250), "min_prominence": 0.0527, "min_height": 0.0591, "min_width": 2.73, "max_width": 107.60, "vibration_type": "C-O stretch", "required_atoms": {'C', 'O'}},
    {"group": "benzene_ring_breathing", "range": (1000, 1075), "min_prominence": 0.0502, "min_height": 0.0731, "min_width": 0.84, "max_width": 62.64, "vibration_type": "Ring breathing", "required_atoms": {'C'}},
    {"group": "alkyl_halide_ccl", "range": (600, 800), "min_prominence": 0.0745, "min_height": 0.1120, "min_width": 3.72, "max_width": 32.50, "vibration_type": "C-Cl stretch", "required_atoms": {'C', 'Cl'}},
    {"group": "alkyl_halide_cbr", "range": (500, 650), "min_prominence": 0.0375, "min_height": 0.0379, "min_width": 3.72, "max_width": 11.46, "vibration_type": "C-Br stretch", "required_atoms": {'C', 'Br'}},
    {"group": "ether_co", "range": (1050, 1150), "min_prominence": 0.0442, "min_height": 0.0580, "min_width": 2.11, "max_width": 54.27, "vibration_type": "C-O stretch", "required_atoms": {'C', 'O'}},
    {"group": "alkene_out_of_plane_ch", "range": (900, 990), "min_prominence": 0.0836, "min_height": 0.1016, "min_width": 3.70, "max_width": 13.26, "vibration_type": "Alkene C-H bend (out-of-plane)", "required_atoms": {'C', 'H'}},
    {"group": "amine_cn", "range": (1020, 1230), "min_prominence": 0.0907, "min_height": 0.1094, "min_width": 3.75, "max_width": 9.53, "vibration_type": "C-N stretch", "required_atoms": {'C', 'N'}},
    {"group": "aromatic_out_of_plane_ch", "range": (690, 900), "min_prominence": 0.0362, "min_height": 0.0530, "min_width": 2.61, "max_width": 104.67, "vibration_type": "Aromatic C-H bend (out-of-plane)", "required_atoms": {'C', 'H'}},
    {"group": "amide_cn", "range": (1180, 1360), "min_prominence": 0.0684, "min_height": 0.0730, "min_width": 3.55, "max_width": 11.39, "vibration_type": "C-N stretch", "required_atoms": {'C', 'N'}},
    {"group": "anhydride_co", "range": (1740, 1850), "min_prominence": 0.1425, "min_height": 0.1425, "min_width": 4.33, "max_width": 7.56, "vibration_type": "C=O stretch (2 bands)", "required_atoms": {'C', 'O'}},
    {"group": "aromatic_cc", "range": (1450, 1600), "min_prominence": 0.1134, "min_height": 0.1377, "min_width": 3.63, "max_width": 12.17, "vibration_type": "Aromatic C=C stretch", "required_atoms": {'C'}},
    {"group": "ketone_co", "range": (1705, 1725), "min_prominence": 0.3390, "min_height": 0.5757, "min_width": 3.86, "max_width": 6.54, "vibration_type": "C=O stretch", "required_atoms": {'C', 'O'}},
    {"group": "carboxylic_acid_oh_broad", "range": (2500, 3300), "min_prominence": 0.1164, "min_height": 0.1433, "min_width": 3.15, "max_width": 18.52, "vibration_type": "O-H stretch, very broad", "required_atoms": {'C', 'O', 'H'}},
    {"group": "amide_co", "range": (1640, 1690), "min_prominence": 0.3359, "min_height": 0.4792, "min_width": 3.72, "max_width": 7.63, "vibration_type": "C=O stretch", "required_atoms": {'C', 'O', 'N'}},
    {"group": "aldehyde_co", "range": (1720, 1740), "min_prominence": 0.4566, "min_height": 0.6306, "min_width": 4.04, "max_width": 7.60, "vibration_type": "C=O stretch", "required_atoms": {'C', 'O', 'H'}},
    {"group": "alkyl_halide_cI", "range": (450, 600), "min_prominence": 0.0663, "min_height": 0.1105, "min_width": 0.81, "max_width": 117.25, "vibration_type": "C-I stretch", "required_atoms": {'C', 'I'}},
    {"group": "ester_co", "range": (1735, 1750), "min_prominence": 0.4228, "min_height": 0.6303, "min_width": 4.04, "max_width": 8.61, "vibration_type": "C=O stretch", "required_atoms": {'C', 'O'}},
    {"group": "alkene_cc", "range": (1620, 1680), "min_prominence": 0.1229, "min_height": 0.1330, "min_width": 3.90, "max_width": 11.33, "vibration_type": "C=C stretch", "required_atoms": {'C'}},
    {"group": "aromatic_ch", "range": (3000, 3100), "min_prominence": 0.0534, "min_height": 0.0675, "min_width": 2.37, "max_width": 8.60, "vibration_type": "Aromatic C-H stretch", "required_atoms": {'C', 'H'}},
    {"group": "amine_nh_primary", "range": (3300, 3500), "min_prominence": 0.0557, "min_height": 0.0644, "min_width": 3.80, "max_width": 31.72, "vibration_type": "N-H stretch (2 peaks)", "required_atoms": {'N', 'H'}},
    {"group": "alcohol_phenol_oh", "range": (3200, 3600), "min_prominence": 0.1784, "min_height": 0.2137, "min_width": 3.93, "max_width": 79.67, "vibration_type": "O-H stretch, broad", "required_atoms": {'O', 'H'}},
    {"group": "nitro_no2_sym", "range": (1300, 1370), "min_prominence": 0.0966, "min_height": 0.1685,"min_width": 3.82,"max_width": 6.11,"vibration_type": "Symmetric N-O stretch","required_atoms": {'N', 'O'}},
    {"group": "amide_nh", "range": (3100, 3500), "min_prominence": 0.0535, "min_height": 0.0535, "min_width": 5.38, "max_width": 12.06, "vibration_type": "N-H stretch", "required_atoms": {'C', 'O', 'N', 'H'}},
    {"group": "alkyne_cc", "range": (2100, 2260), "min_prominence": 0.1227, "min_height": 0.1433, "min_width": 3.82, "max_width": 7.58, "vibration_type": "C≡C stretch", "required_atoms": {'C'}},
    #

    
    
    {"group": "sulfone_so2", "range": (1300, 1350), "min_prominence": 0.08, "vibration_type": "S=O stretch"},
    {"group": "sulfate_so4", "range": (1050, 1150), "min_prominence": 0.04, "vibration_type": "S=O stretch"},
    {"group": "nitro_no2_asym", "range": (1500, 1570), "min_prominence": 0.0966,  "vibration_type": "Asymmetric N-O stretch", "required_atoms": {'N', 'O'}}
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