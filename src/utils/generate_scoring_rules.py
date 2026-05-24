# generate_scoring_rules.py

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from scipy.signal import find_peaks, peak_widths
from tqdm import tqdm
import json
from pathlib import Path

# --- 1. Konfigurasi Input ---
# Definisikan semua gugus fungsi yang akan dianalisis di sini.
CONFIG = {
    "Ester": {
        "smarts": "[CX3](=O)[OX2H0][#6]",
        "peaks": {
            "ESTER_CO_STRETCH": {"search_window": [1720, 1760]},
            "ESTER_C_O_STRETCH": {"search_window": [1150, 1300]}
        }
    },
    "Carboxylic Acid": {
        "smarts": "C(=O)[O;H1]",
        "peaks": {
            "CA_OH_STRETCH": {"search_window": [2500, 3300]},
            "CA_CO_STRETCH": {"search_window": [1680, 1720]}
        }
    },
    "Alkene": {
        "smarts": "[#6]=[#6]",
        "peaks": {
            "ALKENE_CH_STRETCH": {"search_window": [3010, 3100]},
            "ALKENE_CC_STRETCH": {"search_window": [1620, 1680]}
        }
    },
    "Ketone": {
        "smarts": "[#6][CX3](=O)[#6]",
        "peaks": {
            "KETONE_CO_STRETCH": {"search_window": [1705, 1725]}
        }
    }
}

DATASET_PATH = Path("data/standarize/dataset.csv")
OUTPUT_JSON_PATH = Path("ir_scoring_rules.json")

def get_smiles_from_cas(cas_no):
    """Dummy function to simulate getting SMILES. In a real scenario, this would query a database."""
    # This is a placeholder. You would replace this with your actual SMILES lookup logic.
    # For this example, we'll generate a dummy SMILES based on the CAS.
    # A real implementation would use a lookup table or an API like PubChem.
    # Let's assume some CAS numbers for our CONFIG groups.
    cas_map = {
        "141-78-6": "CCOC(=O)C",  # Ethyl acetate (Ester)
        "109-94-4": "CCOC=O",     # Ethyl formate (Ester)
        "64-19-7": "CC(=O)O",     # Acetic acid (Carboxylic Acid)
        "79-09-4": "CCC(=O)O",    # Propanoic acid (Carboxylic Acid)
        "110-83-8": "C1=CCCCC1",  # Cyclohexene (Alkene)
        "67-64-1": "CC(=O)C",     # Acetone (Ketone)
    }
    return cas_map.get(cas_no.split('_')[0], None)

def main():
    """
    Fungsi utama untuk menghasilkan file aturan skoring IR dari dataset spektral.
    """
    print(f"🚀 Memulai pembuatan aturan skoring dari '{DATASET_PATH}'...")

    try:
        dataset_df = pd.read_csv(DATASET_PATH)
        # Asumsi 'molecule_id' mengandung CAS number di awal.
        dataset_df['smiles'] = dataset_df['molecule_id'].apply(get_smiles_from_cas)
        dataset_df = dataset_df.dropna(subset=['smiles'])
        print(f"✅ Dataset berhasil dimuat. Ditemukan {len(dataset_df)} entri dengan SMILES.")
    except FileNotFoundError:
        print(f"❌ Error: File dataset '{DATASET_PATH}' tidak ditemukan.")
        return

    # --- 2. Struktur Data Output ---
    output_rules = {}

    # --- 3. Loop Pemrosesan Utama ---
    for group_name, details in CONFIG.items():
        print(f"\n🔬 Memproses gugus fungsi: {group_name}")
        
        # a. Saring dataset untuk molekul yang cocok
        pattern = Chem.MolFromSmarts(details["smarts"])
        
        matching_molecules = []
        for _, row in dataset_df.iterrows():
            mol = Chem.MolFromSmiles(row['smiles'])
            if mol and mol.HasSubstructMatch(pattern):
                matching_molecules.append(row)
        
        matching_df = pd.DataFrame(matching_molecules)
        print(f"  -> Ditemukan {len(matching_df)} molekul yang cocok dengan SMARTS.")

        if matching_df.empty:
            continue

        output_rules[group_name] = {"smarts": details["smarts"], "peaks": {}}

        # b. Iterasi pada setiap definisi puncak
        peak_definitions = details["peaks"]
        
        # Logika untuk peak_weight
        num_peaks = len(peak_definitions)
        peak_weights = {}
        if num_peaks == 1:
            peak_weights[list(peak_definitions.keys())[0]] = 1.0
        elif num_peaks > 1:
            # Asumsi puncak dengan frekuensi lebih tinggi lebih unik
            sorted_peaks = sorted(peak_definitions.items(), key=lambda item: item[1]['search_window'][1], reverse=True)
            peak_weights[sorted_peaks[0][0]] = 0.6
            for i in range(1, num_peaks):
                peak_weights[sorted_peaks[i][0]] = 0.4 / (num_peaks - 1)

        for peak_name, peak_details in peak_definitions.items():
            search_window = peak_details["search_window"]
            
            # i. Kumpulkan data dari semua molekul yang relevan
            positions, intensities, fwhms = [], [], []
            for _, row in tqdm(matching_df.iterrows(), total=len(matching_df), desc=f"   - Menganalisis {peak_name}"):
                try:
                    spectrum_df = pd.read_csv(row['processed_filepath'])
                    # Ganti nama kolom jika perlu, sesuaikan dengan output pra-pemrosesan Anda
                    spectrum_df.rename(columns={'wavenumber_cm-1': 'wavenumber', 'absorbance_normalized': 'absorbance'}, inplace=True)

                    window_df = spectrum_df[(spectrum_df['wavenumber'] >= search_window[0]) & (spectrum_df['wavenumber'] <= search_window[1])]
                    if window_df.empty: continue

                    peaks, props = find_peaks(window_df['absorbance'], height=0.1, prominence=0.05)
                    if len(peaks) == 0: continue

                    # Ambil puncak paling menonjol
                    most_prominent_idx = peaks[np.argmax(props['prominences'])]
                    
                    # Hitung FWHM
                    widths, _, _, _ = peak_widths(window_df['absorbance'].values, [most_prominent_idx], rel_height=0.5)
                    
                    positions.append(window_df['wavenumber'].iloc[most_prominent_idx])
                    intensities.append(props['peak_heights'][np.argmax(props['prominences'])])
                    if len(widths) > 0:
                        fwhms.append(widths[0])

                except Exception:
                    continue
            
            if not positions:
                print(f"     - ⚠️ Tidak ada puncak valid yang ditemukan untuk {peak_name}.")
                continue

            # ii. Hitung parameter statistik
            stats = {
                "position": {"mean": np.mean(positions), "std": np.std(positions), "p10": np.percentile(positions, 10), "p50": np.percentile(positions, 50), "p90": np.percentile(positions, 90)},
                "intensity": {"mean": np.mean(intensities), "std": np.std(intensities), "p10": np.percentile(intensities, 10), "p50": np.percentile(intensities, 50), "p90": np.percentile(intensities, 90)},
                "fwhm": {"mean": np.mean(fwhms), "std": np.std(fwhms), "p10": np.percentile(fwhms, 10), "p50": np.percentile(fwhms, 50), "p90": np.percentile(fwhms, 90)}
            }

            # iii. Simpan hasil ke struktur output
            output_rules[group_name]["peaks"][peak_name] = {
                "search_window": search_window,
                "peak_weight": peak_weights.get(peak_name, 0.0),
                "stats": stats
            }

    # --- 5. Tulis File JSON ---
    if not output_rules:
        print("\n❌ Tidak ada aturan yang dihasilkan. Periksa konfigurasi dan dataset Anda.")
        return

    print(f"\n💾 Menulis hasil ke '{OUTPUT_JSON_PATH}'...")
    with open(OUTPUT_JSON_PATH, 'w') as f:
        json.dump(output_rules, f, indent=2)

    print("🎉 Pembuatan file aturan skoring selesai!")

if __name__ == "__main__":
    main()