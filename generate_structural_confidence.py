import json
from pathlib import Path
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger
from tqdm import tqdm

# Menonaktifkan logging error RDKit yang terlalu verbose di konsol
RDLogger.DisableLog('rdApp.*')

# --- Konfigurasi Path ---
SMILES_CACHE_PATH = Path("data/standarize/smiles_cache.json")
RULES_FILE_PATH = Path("data/standarize/data_labels.csv")
OUTPUT_JSON_PATH = Path("data/reports/structural_confidence.json")

def generate_structural_confidence(smiles_cache_path: Path, rules_path: Path, output_path: Path):
    """
    Menganalisis setiap molekul dalam cache SMILES terhadap setiap aturan SMARTS
    dan menghasilkan file JSON dengan label struktural dan skor confidence (1, 0.5, 0).
    """
    # --- 1. Validasi dan Pemuatan File Input ---
    if not smiles_cache_path.exists():
        print(f"❌ Error: File cache SMILES tidak ditemukan di '{smiles_cache_path}'.")
        return

    if not rules_path.exists():
        print(f"❌ Error: File aturan label tidak ditemukan di '{rules_path}'.")
        return

    with open(smiles_cache_path, 'r') as f:
        original_cache = json.load(f)

    rules_df = pd.read_csv(rules_path).drop_duplicates(subset=['name'])

    print(f"🔬 Menganalisis {len(original_cache)} molekul terhadap {len(rules_df)} aturan...")

    # --- 2. Pra-kompilasi Pola SMARTS untuk Efisiensi ---
    rules_df['pattern'] = rules_df['SMARTS'].apply(Chem.MolFromSmarts)
    # Pola khusus untuk logika 0.5
    aromatic_pattern = Chem.MolFromSmarts('[c]')

    # --- 3. Proses Setiap Molekul ---
    new_cache = {}
    for cas, data in tqdm(original_cache.items(), desc="Processing Molecules"):
        smiles = data if isinstance(data, str) else data.get("smiles")
        if not smiles:
            continue

        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            continue

        is_aromatic = mol.HasSubstructMatch(aromatic_pattern)
        detected_groups = {}

        for _, rule in rules_df.iterrows():
            # --- PERBAIKAN: Normalisasi nama grup ---
            # Mengganti karakter 'EN DASH' (–) dengan 'HYPHEN' (-) untuk konsistensi.
            # Ini memastikan nama grup di output JSON cocok dengan file CSV lainnya.
            group_name = rule['name'].replace('\u2013', '-')
            smarts_str = rule['SMARTS']
            pattern = rule['pattern']
            
            if not pattern:
                continue

            has_match = mol.HasSubstructMatch(pattern)

            if has_match: # Jika ada kecocokan SMARTS
                confidence = 1 # Default confidence adalah 1
                # Logika khusus untuk confidence 0.5
                # Jika pola adalah C=C umum dan molekulnya aromatik, turunkan confidence menjadi 0.5
                if smarts_str == '[#6]=[#6]' and is_aromatic:
                    confidence = 0.5

                detected_groups[group_name] = {
                    "SMARTS": smarts_str,
                    "range_min": rule['range_min'],
                    "range_max": rule['range_max'],
                    "confidence": confidence
                }

        new_cache[cas] = {
            "smiles": smiles,
            "detected_functional_groups": detected_groups
        }

    # --- 4. Tulis File Output ---
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(new_cache, f, indent=2)

    print(f"\n✅ Analisis kepercayaan struktural selesai. File disimpan di: '{output_path}'")

if __name__ == "__main__":
    generate_structural_confidence(SMILES_CACHE_PATH, RULES_FILE_PATH, OUTPUT_JSON_PATH)