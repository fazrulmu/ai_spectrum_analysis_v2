import json
from pathlib import Path
from rdkit import Chem
from rdkit import RDLogger
from tqdm import tqdm

# Menonaktifkan logging error RDKit yang terlalu verbose di konsol
RDLogger.DisableLog('rdApp.*')

# --- Konfigurasi ---
SMILES_CACHE_PATH = Path("data/standarize/smiles_cache.json")

# Pola SMARTS yang akan diperiksa, sesuai permintaan Anda
SMARTS_PATTERNS = {
    # ======= C - O / C = O / O - H =========
    "Alcohol (R-OH)": "[#6;!$(C=O)]-[OX2H1]",                     # alkohol alifatik
    "Phenol (Ar-OH)": "[c]-[OX2H1]",                              # fenol
    "Carboxylic acid (R-COOH)": "[CX3](=O)[OX2H1]",               # asam karboksilat
    "Ester (R-COOR)": "[CX3](=O)[OX2][#6]",                       # ester umum
    "Aromatic ester (Ar-COOR)": "[c][CX3](=O)[OX2][#6]",          # ester aromatik
    "Ketone/Aromatic Ketone": "[#6,a][CX3](=O)[#6,a]",            # keton alifatik atau aromatik
    "Aldehyde (R-CHO)": "[CX3H1](=O)[#6,#1,a]",                   # aldehida
    "Carboxylate (R-COO-)": "[CX3](=O)[O-]",                      # garam karboksilat
    "Acid anhydride (R-CO-O-CO-R)": "[CX3](=O)O[CX3](=O)",        # anhidrida
    "Acid chloride (R-COCl)": "[CX3](=O)Cl",                      # asil klorida

    # ======= C - O - C ==========
    "Ether (R-O-R)": "[#6,a]-[OX2;!$(O(C=O))]-[#6,a]",            # eter umum
    "Aryl ether (Ar-O-R)": "[c]-[OX2;!$(O(C=O))]-[#6]",           # eter aromatik

    # ======= N-containing =========
    "Amine primary (R-NH2)": "[NX3;H2][#6]",                      # amina primer
    "Amine secondary (R2-NH)": "[NX3;H1][#6][#6]",                # amina sekunder
    "Amine tertiary (R3-N)": "[NX3;H0][#6][#6][#6]",              # amina tersier
    "Amide (R-CONH2)": "[CX3](=O)[NX3H2]",                        # amida
    "Nitrile (R-C≡N)": "[CX2]#[NX1]",                             # nitril
    "Nitro (R-NO2)": "[NX3](=O)=O",                               # gugus nitro

    # ======= C - H / C = C / Aromatik =======
    "Alkane (C-H)": "[CH2,CH3]",                                  # C–H jenuh
    "Alkene (C=C)": "[CX3]=[CX3]",                                # alkena
    "Aromatic (Ar)": "[c]",                                       # aromatik
    "Aromatic C=C (in ring)": "[cX3]=[cX3]",                      # konjugasi aromatik

    # ======= Halogen ==========
    "Alkyl fluoride (R-F)": "[CX4][F]",                           # alkil fluorida
    "Alkyl chloride (R-Cl)": "[CX4][Cl]",                         # alkil klorida
    "Alkyl bromide (R-Br)": "[CX4][Br]",                          # alkil bromida
    "Alkyl iodide (R-I)": "[CX4][I]",                             # alkil iodida
    "Aryl halide (Ar-X)": "[c][F,Cl,Br,I]",                       # halogen aromatik

    # ======= Sulfur ==========
    "Thiol (R-SH)": "[#16X2H]",                                   # thiol
    "Thioether (R-S-R)": "[#16X2][#6]",                           # sulfida
    "Sulfoxide (R-S(=O)-R)": "[#16X3](=O)[#6]",                   # sulfoxida
    "Sulfone (R-S(=O)2-R)": "[#16X4](=O)(=O)[#6]",                # sulfon
    "Carbon disulfide (CS2)": "[#16]=[#6]=[#16]",                 # karbon disulfida

    # ======= Phosphorus & Silicon ==========
    "Phosphate ester": "[PX4](=O)(O)(O)O",                        # fosfat organik
    "Si-Cl compound": "[Si][Cl]",                                 # senyawa silil klorida

    # ======= Miscellaneous ==========
    "Peroxide (ROOR)": "[OX2][OX2]",                              # peroksida
    "Epoxide (C-O-C triangle)": "[OX2r3]",                        # epoksida
    "Carbamate (R-OC(=O)NR2)": "[CX3](=O)[NX3][OX2]",             # karbamat
}


def update_cache_with_possibilities(cache_path: Path):
    """
    Memuat cache SMILES, menganalisis setiap molekul untuk kemungkinan gugus fungsi
    berdasarkan pola SMARTS, dan menulis ulang file cache dengan struktur baru.
    """
    if not cache_path.exists():
        print(f"❌ Error: File cache tidak ditemukan di '{cache_path}'.")
        return

    with open(cache_path, 'r') as f:
        original_cache = json.load(f)

    new_cache = {}
    print(f"🔬 Menganalisis {len(original_cache)} molekul untuk menambahkan kemungkinan gugus fungsi...")

    # Buat objek MolFromSmarts sekali saja untuk efisiensi
    patterns = {name: Chem.MolFromSmarts(smarts) for name, smarts in SMARTS_PATTERNS.items()}

    for cas, data in tqdm(original_cache.items(), desc="Processing SMILES"):
        # PERBAIKAN: Tangani format cache lama (string) dan baru (dict)
        # Ini membuat skrip aman untuk dijalankan berulang kali.
        smiles = data if isinstance(data, str) else data.get("smiles")

        if not smiles or not isinstance(smiles, str):
            continue

        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            continue

        possibilities = []
        for name, pattern in patterns.items():
            if mol.HasSubstructMatch(pattern):
                possibilities.append(SMARTS_PATTERNS[name]) # Tambahkan string SMARTS asli

        new_cache[cas] = {
            "smiles": smiles,
            "possibility": possibilities
        }

    # Tulis kembali file JSON dengan struktur baru dan format yang rapi
    with open(cache_path, 'w') as f:
        json.dump(new_cache, f, indent=2)

    print(f"\n✅ File '{cache_path}' berhasil diperbarui dengan struktur baru.")

if __name__ == "__main__":
    update_cache_with_possibilities(SMILES_CACHE_PATH)