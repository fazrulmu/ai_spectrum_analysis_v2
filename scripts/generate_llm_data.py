import json
import csv
import os
import random

# ==========================================
# 1. BAGIAN LOAD DATA
# ==========================================

def load_json(filepath):
    """Memuat file JSON."""
    if not os.path.exists(filepath):
        print(f"Warning: File {filepath} tidak ditemukan.")
        return {}
    with open(filepath, 'r') as f:
        return json.load(f)

def load_csv_as_dict(filepath, key_column):
    """Memuat CSV ke dalam dictionary dengan key tertentu."""
    if not os.path.exists(filepath):
        print(f"Warning: File {filepath} tidak ditemukan.")
        return {}
    
    data = {}
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = row[key_column]
            if key not in data:
                data[key] = []
            data[key].append(row)
    return data

def load_structural_confidence(filepath):
    """Memuat data structural confidence."""
    if not os.path.exists(filepath):
        return {}
    with open(filepath, 'r') as f:
        return json.load(f)

# ==========================================
# 2. GENERATOR IR QUALITATIVE
# ==========================================

def generate_ir_qualitative(cas, smiles_data, labels_lookup):
    """
    Fokus: Vibrasi ikatan, Gugus Fungsi, Momen Dipol.
    """
    examples = []
    
    # Ambil data SMILES dan kemungkinan gugus fungsi
    smiles = smiles_data.get('smiles', '')
    possibilities = smiles_data.get('possibility', [])
    
    # Kumpulkan info gugus fungsi dari data_labels.csv
    molecule_groups = []
    for smarts in possibilities:
        if smarts in labels_lookup:
            molecule_groups.extend(labels_lookup[smarts])
            
    if not molecule_groups:
        return []

    # --- MEMBUAT NARASI DESKRIPTIF GABUNGAN ---
    # Menggabungkan semua gugus fungsi menjadi satu penjelasan struktur
    desc_list = []
    for group in molecule_groups:
        name = group.get('name', 'Gugus tak dikenal')
        r_min = group.get('range_min', '?')
        r_max = group.get('range_max', '?')
        vib_type = group.get('vibration_type', 'vibrasi') # misal: stretching/bending
        desc_list.append(f"- **{name}**: Menunjukkan {vib_type} karakteristik pada bilangan gelombang {r_min}-{r_max} cm-1.")

    joined_desc = "\n".join(desc_list)
    
    # Template 1: Analisis Hubungan Struktur-Spektrum (Forward)
    examples.append({
        "instruction": f"Lakukan analisis kualitatif spektrum inframerah (IR) untuk senyawa dengan CAS {cas} berdasarkan struktur kimianya.",
        "input": f"CAS: {cas}\nSMILES: {smiles}",
        "output": (
            f"Berdasarkan struktur molekulnya ({smiles}), senyawa ini memiliki beberapa gugus fungsi kunci yang aktif secara inframerah. "
            f"Analisis kualitatif memprediksi serapan berikut:\n\n{joined_desc}\n\n"
            f"Keberadaan puncak-puncak ini mengonfirmasi kerangka struktur senyawa tersebut melalui vibrasi ikatan spesifik."
        )
    })

    # Template 2: Identifikasi Gugus dari Data Spektrum (Reverse)
    # Mengambil satu sampel gugus secara acak untuk detail pertanyaan
    for group in molecule_groups:
        name = group['name']
        r_min = group['range_min']
        r_max = group['range_max']
        desc = group['description']
        
        examples.append({
            "instruction": "Identifikasi gugus fungsi kimia yang bertanggung jawab atas sinyal spektrum IR berikut.",
            "input": f"Terdeteksi pita serapan {desc} di daerah {r_min}-{r_max} cm-1 pada senyawa CAS {cas}.",
            "output": (
                f"Sinyal pada area {r_min}-{r_max} cm-1 tersebut merupakan karakteristik dari gugus **{name}**. "
                f"Secara kimiawi, ini menunjukkan adanya perubahan momen dipol pada ikatan tersebut saat bervibrasi."
            )
        })

    return examples

# ==========================================
# 3. GENERATOR UV QUALITATIVE
# ==========================================

def generate_uv_qualitative(cas, uv_data_list, smiles):
    """
    Fokus: Transisi Elektronik, Kromofor, Konjugasi, Lambda Max.
    """
    if not uv_data_list:
        return []

    examples = []
    
    # Kita asumsikan uv_data_list berisi list of rows dari CSV UV
    # Format row diharapkan: {'lambda_max': '254', 'log_epsilon': '3.5', ...}
    
    summary_uv = []
    has_conjugation = False
    
    for row in uv_data_list:
        l_max = row.get('lambda_max', 'N/A')
        eps = row.get('log_epsilon', 'N/A')
        try:
            if float(l_max) > 200: has_conjugation = True
        except:
            pass
        summary_uv.append(f"λmax {l_max} nm (log ε {eps})")

    joined_uv = "; ".join(summary_uv)
    
    # Narasi tambahan tentang struktur
    structural_comment = ""
    if has_conjugation:
        structural_comment = "Data ini mengindikasikan adanya sistem terkonjugasi (ikatan rangkap selang-seling) atau gugus kromofor aromatik yang memfasilitasi transisi elektronik π → π* atau n → π*."
    else:
        structural_comment = "Absorpsi pada panjang gelombang ini umumnya terkait dengan transisi elektronik pada gugus fungsi sederhana atau auksokrom."

    # Template 1: Laporan Kualitatif UV
    examples.append({
        "instruction": f"Berikan tinjauan kualitatif mengenai spektrum UV-Vis untuk senyawa CAS {cas}.",
        "input": f"CAS: {cas}\nSMILES: {smiles}",
        "output": (
            f"Analisis spektroskopi UV-Vis untuk senyawa ini menunjukkan serapan maksimum pada: {joined_uv}. "
            f"\n\nPenjelasan Struktur Kimia: {structural_comment} "
            f"Struktur SMILES {smiles} mendukung hal ini dengan keberadaan gugus-gugus elektron yang dapat tereksitasi."
        )
    })
    
    # Template 2: Prediksi Kromofor
    examples.append({
        "instruction": "Jelaskan hubungan antara struktur elektronik senyawa berikut dengan data UV yang teramati.",
        "input": f"Senyawa: {cas}. Data UV: {joined_uv}.",
        "output": (
            f"Nilai λmax dan intensitas absorpsi (log ε) tersebut merefleksikan konfigurasi elektronik molekul. "
            f"{structural_comment} Hal ini memungkinkan identifikasi keberadaan kromofor spesifik dalam kerangka molekul."
        )
    })

    return examples

# ==========================================
# 4. GENERATOR STRUCTURAL CONFIDENCE
# ==========================================

def generate_structural_confidence_instructions(cas, data):
    examples = []
    detected_groups = data.get('detected_functional_groups', {})
    
    if not detected_groups:
        return []

    # Create a list of formatted strings for each group
    group_descriptions = []
    for name, details in detected_groups.items():
        r_min = details.get('range_min')
        r_max = details.get('range_max')
        conf = details.get('confidence')
        
        # Format: "- Name: Min-Max cm-1 (Confidence: X)"
        group_descriptions.append(f"- {name}: {r_min}-{r_max} cm-1 (Confidence: {conf})")
    
    if not group_descriptions:
        return []

    output_text = f"Untuk CAS {cas}, gugus fungsi yang terdeteksi adalah:\n" + "\n".join(group_descriptions)

    # Template 1: Standard Identification
    examples.append({
        "instruction": f"Identifikasi gugus fungsi dan rentang IR untuk CAS {cas} berdasarkan analisis struktural.",
        "input": f"CAS {cas}",
        "output": output_text
    })

    # Template 2: List Request
    examples.append({
        "instruction": f"Sebutkan daftar gugus fungsi yang terdeteksi pada senyawa dengan CAS {cas}.",
        "input": f"Senyawa CAS {cas}",
        "output": output_text
    })

    # Template 3: Analysis Report
    examples.append({
        "instruction": f"Buatkan laporan analisis struktural singkat mengenai gugus fungsi untuk CAS {cas}.",
        "input": f"Data CAS {cas}",
        "output": output_text
    })

    # Template 4: Confidence Inquiry
    examples.append({
        "instruction": f"Apa saja gugus fungsi yang memiliki tingkat kepercayaan (confidence) tertentu pada CAS {cas}?",
        "input": f"CAS {cas}",
        "output": output_text
    })

    # Template 5: Plotting Data Request
    examples.append({
        "instruction": f"Berikan data yang diperlukan untuk menandai plot spektrum IR dari CAS {cas}.",
        "input": f"CAS {cas}",
        "output": output_text
    })
    
    return examples

# ==========================================
# 5. FUNGSI UTAMA (MAIN)
# ==========================================

def main():
    # Setup Path
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # File Input
    smiles_path = os.path.join(base_dir, 'data', 'standarize', 'smiles_cache.json')
    labels_path = os.path.join(base_dir, 'data', 'standarize', 'data_labels.csv')
    confidence_path = os.path.join(base_dir, 'data', 'reports', 'structural_confidence.json')
    
    # File Input Tambahan (UV)
    uv_path = os.path.join(base_dir, 'data', 'for_train', 'universal_training_dataset_UV.csv')
    
    # File Output
    output_path = os.path.join(base_dir, 'data_llm.jsonl')

    print("--- Memuat Data ---")
    smiles_cache = load_json(smiles_path)
    labels_lookup = load_csv_as_dict(labels_path, 'SMARTS')
    confidence_data = load_structural_confidence(confidence_path)
    
    # Load UV Data (Key = cas_number)
    uv_lookup = load_csv_as_dict(uv_path, 'cas_number') 
    
    all_examples = []
    
    print("--- Generating Datasets ---")
    
    count_ir = 0
    count_uv = 0
    count_conf = 0
    
    for cas, data in smiles_cache.items():
        smiles = data.get('smiles', '')
        
        # 1. Generate IR Qualitative
        ir_ex = generate_ir_qualitative(cas, data, labels_lookup)
        all_examples.extend(ir_ex)
        count_ir += len(ir_ex)
        
        # 2. Generate UV Qualitative (Jika data tersedia)
        if cas in uv_lookup:
            uv_data = uv_lookup[cas]
            uv_ex = generate_uv_qualitative(cas, uv_data, smiles)
            all_examples.extend(uv_ex)
            count_uv += len(uv_ex)

    # 3. Generate Structural Confidence
    for cas, data in confidence_data.items():
        conf_ex = generate_structural_confidence_instructions(cas, data)
        all_examples.extend(conf_ex)
        count_conf += len(conf_ex)

    print(f"Hasil Generate:\n- IR Qualitative Examples: {count_ir}\n- UV Qualitative Examples: {count_uv}\n- Structural Confidence Examples: {count_conf}")
    print(f"Total: {len(all_examples)}")
    print(f"Menulis ke {output_path}...")
    
    with open(output_path, 'w') as f:
        for example in all_examples:
            f.write(json.dumps(example) + '\n')
            
    print("Selesai.")

if __name__ == "__main__":
    main()
