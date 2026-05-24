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
OUTPUT_JSON_PATH = Path("data/reports/structural_labels.json")

def generate_structural_labels(smiles_cache_path: Path, rules_path: Path, output_path: Path):
    """
    Menganalisis setiap molekul dalam cache SMILES terhadap setiap aturan SMARTS
    dan menghasilkan file JSON dengan label struktural biner (1/0).
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

    rules_df = pd.read_csv(rules_path)
    # Pastikan tidak ada SMARTS duplikat, ambil yang pertama jika ada
    unique_smarts_list = rules_df["SMARTS"].dropna().unique().tolist()

    print(f"🔬 Menganalisis {len(original_cache)} molekul terhadap {len(unique_smarts_list)} aturan SMARTS unik...")

    # --- 2. Pra-kompilasi Pola SMARTS untuk Efisiensi ---
    smarts_patterns = {smarts: Chem.MolFromSmarts(smarts) for smarts in unique_smarts_list}
    # Hapus pola yang tidak valid
    valid_patterns = {smarts: pattern for smarts, pattern in smarts_patterns.items() if pattern}

    # --- 3. Proses Setiap Molekul ---
    new_cache = {}
    for cas, data in tqdm(original_cache.items(), desc="Processing Molecules"):
        # Menangani format cache lama (string) dan baru (dict)
        smiles = data if isinstance(data, str) else data.get("smiles")

        if not smiles:
            continue

        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            continue

        detected_groups = {}
        for smarts, pattern in valid_patterns.items():
            # Periksa kecocokan substruktur dan tetapkan label 1 atau 0
            detected_groups[smarts] = 1 if mol.HasSubstructMatch(pattern) else 0

        new_cache[cas] = {
            "smiles": smiles,
            "detected_functional_groups": detected_groups
        }

    # --- 4. Tulis File Output ---
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(new_cache, f, indent=2)

    print(f"\n✅ Analisis struktural selesai. File label disimpan di: '{output_path}'")

if __name__ == "__main__":
    generate_structural_labels(SMILES_CACHE_PATH, RULES_FILE_PATH, OUTPUT_JSON_PATH)