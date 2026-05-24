Tentu, ini adalah **Database Pengetahuan (Knowledge Base)** dalam format JSON yang diperkaya berdasarkan daftar CAS yang Anda berikan.

Data ini dirancang khusus untuk sistem **AI + Python Logic** Anda. Saya telah menambahkan field penting:

1.  **`ir_markers`**: Gugus fungsi wajib (Key Markers) untuk validasi logika Python.
2.  **`uv_expected`**: Data kualitatif UV untuk cross-check.
3.  **`ai_insight`**: Catatan khusus untuk membantu model 0.5B membedakan senyawa mirip.

Berikut adalah JSON yang mencakup sebagian besar senyawa kunci (representatif) dari daftar Anda:

```json
[
  {
    "cas": "100-42-5",
    "name": "Styrene",
    "formula": "C8H8",
    "class": "Aromatic Hydrocarbon",
    "ir_markers": ["Vinyl C=C (1630)", "Aromatic Ring (1500, 1600)", "Mono-subst Benzene (690, 750)"],
    "uv_expected": "Strong absorption (~245 nm) due to conjugation.",
    "ai_insight": "Monomer plastik. Bau tajam khas. Wajib memiliki gugus Vinil dan Aromatik sekaligus. Jika gugus Vinil hilang, kemungkinan telah terpolimerisasi."
  },
  {
    "cas": "100-52-7",
    "name": "Benzaldehyde",
    "formula": "C7H6O",
    "class": "Aromatic Aldehyde",
    "ir_markers": ["C=O Aldehyde (1700)", "Aldehyde C-H Doublet (2720, 2820)", "Aromatic Ring"],
    "uv_expected": "Strong absorption (~250 nm).",
    "ai_insight": "Puncak kembar (doublet) C-H aldehid di 2700-2800 adalah kunci identifikasi utama untuk membedakannya dari keton atau asam."
  },
  {
    "cas": "108-88-3",
    "name": "Toluene",
    "formula": "C7H8",
    "class": "Aromatic Hydrocarbon",
    "ir_markers": ["Aromatic C-H (>3000)", "Methyl C-H (<3000)", "Mono-subst Benzene (690, 730)"],
    "uv_expected": "Moderate absorption (~260 nm, fine structure).",
    "ai_insight": "Mirip Benzena tetapi memiliki gugus Metil (alifatik) di area 2900. Sering digunakan sebagai pelarut."
  },
  {
    "cas": "67-64-1",
    "name": "Acetone",
    "formula": "C3H6O",
    "class": "Ketone",
    "ir_markers": ["Strong C=O (1715)", "No C-H Aldehyde", "Methyl C-H"],
    "uv_expected": "Weak absorption (~270-280 nm, n->pi* transition).",
    "ai_insight": "Pelarut sangat umum. Hati-hati, sering muncul sebagai residu pencucian alat. Tidak memiliki gugus OH atau Aromatik."
  },
  {
    "cas": "108-95-2",
    "name": "Phenol",
    "formula": "C6H6O",
    "class": "Phenol",
    "ir_markers": ["Broad O-H (3200-3500)", "Aromatic C=C (1500, 1600)", "C-O Stretch (1230)"],
    "uv_expected": "Strong absorption (~270 nm). Bathochromic shift in base.",
    "ai_insight": "Bentuk padatan higroskopis. Gugus OH melebar karena ikatan hidrogen. Hati-hati membedakan dengan alkohol alifatik (cek cincin aromatik)."
  },
  {
    "cas": "64-19-7",
    "name": "Acetic Acid",
    "formula": "C2H4O2",
    "class": "Carboxylic Acid",
    "ir_markers": ["Very Broad O-H (2500-3300)", "Strong C=O (1710)", "C-O Stretch"],
    "uv_expected": "Weak absorption < 210 nm.",
    "ai_insight": "Ciri khas paling utama adalah OH yang 'sangat lebar' dan berantakan menutupi area C-H (3000). Baunya menyengat (cuka)."
  },
  {
    "cas": "107-13-1",
    "name": "Acrylonitrile",
    "formula": "C3H3N",
    "class": "Nitrile",
    "ir_markers": ["Sharp C≡N (2220-2260)", "Vinyl C=C (1600)", "Vinyl C-H"],
    "uv_expected": "Absorption present due to conjugation.",
    "ai_insight": "Gugus Nitril (CN) sangat khas di area 2200. Pastikan membedakan C=C vinil dengan C=C aromatik."
  },
  {
    "cas": "71-43-2",
    "name": "Benzene",
    "formula": "C6H6",
    "class": "Aromatic Hydrocarbon",
    "ir_markers": ["Aromatic C-H (>3000 only)", "Aromatic Ring (1480)", "Mono-subst pattern (670)"],
    "uv_expected": "Classic fine structure bands around 250-260 nm.",
    "ai_insight": "Sangat simetris. Tidak ada C-H alifatik (<3000) sama sekali. Jika ada puncak <3000, berarti bukan benzena murni (atau Toluena/Xilena)."
  },
  {
    "cas": "110-82-7",
    "name": "Cyclohexane",
    "formula": "C6H12",
    "class": "Alkane (Cyclic)",
    "ir_markers": ["Strong C-H Stretch (2850-2950)", "CH2 Bending (1450)", "No Aromatic peaks"],
    "uv_expected": "Transparent (No absorption > 200 nm).",
    "ai_insight": "Hanya berisi puncak C-H jenuh. Sering digunakan sebagai pelarut UV karena transparansinya. Jangan tertukar dengan Benzena (tidak punya C=C)."
  },
  {
    "cas": "75-05-8",
    "name": "Acetonitrile",
    "formula": "C2H3N",
    "class": "Nitrile",
    "ir_markers": ["Sharp C≡N (2250)", "Aliphatic C-H"],
    "uv_expected": "Transparent (UV Cutoff very low < 190 nm).",
    "ai_insight": "Pelarut polar umum. Sinyal CN sangat tajam dan kuat di 2250. Sering muncul sebagai pelarut HPLC."
  },
  {
    "cas": "62-53-3",
    "name": "Aniline",
    "formula": "C6H7N",
    "class": "Aromatic Amine",
    "ir_markers": ["N-H Doublet (3300-3400, Primary Amine)", "Aromatic Ring", "C-N Stretch"],
    "uv_expected": "Strong absorption (~230, 280 nm).",
    "ai_insight": "Amina primer aromatik. Puncak N-H kembar (dua bukit kecil) di area 3300-3400 adalah kunci utama."
  },
  {
    "cas": "98-01-1",
    "name": "Furfural",
    "formula": "C5H4O2",
    "class": "Aldehyde (Heterocyclic)",
    "ir_markers": ["C=O Aldehyde (1670-1690)", "C=C Furan Ring", "C-O-C Ring"],
    "uv_expected": "Strong absorption (~270 nm).",
    "ai_insight": "Aldehid yang terikat pada cincin Furan (oksigen dalam cincin). Puncak C=O sedikit lebih rendah dari aldehid biasa karena konjugasi."
  },
  {
    "cas": "108-94-1",
    "name": "Cyclohexanone",
    "formula": "C6H10O",
    "class": "Ketone (Cyclic)",
    "ir_markers": ["Strong C=O (1710)", "Cyclic C-H (2800-2950)", "No Aromatic"],
    "uv_expected": "Weak n->pi* transition (~280 nm).",
    "ai_insight": "Keton siklik. Mirip aseton tapi dengan rantai karbon cincin jenuh."
  },
  {
    "cas": "91-20-3",
    "name": "Naphthalene",
    "formula": "C10H8",
    "class": "Polycyclic Aromatic (PAH)",
    "ir_markers": ["Aromatic C-H (>3000)", "Ring Breathing (multiple peaks)", "Out-of-plane C-H (780)"],
    "uv_expected": "Complex spectrum with multiple bands (220, 275, 312 nm).",
    "ai_insight": "Kapur barus. Struktur dua cincin benzena menyatu. Pola sidik jari aromatik lebih kompleks daripada benzena tunggal."
  },
  {
    "cas": "100-01-6",
    "name": "p-Nitroaniline",
    "formula": "C6H6N2O2",
    "class": "Nitro Aromatic",
    "ir_markers": ["NO2 Symmetric (1300-1360)", "NO2 Asymmetric (1500-1550)", "NH2 Stretches"],
    "uv_expected": "Very strong absorption, yellow color (Visible region tailing).",
    "ai_insight": "Gugus Nitro (NO2) sangat kuat di IR, sering disebut 'Vampire Teeth' (dua taring tajam) di 1350 dan 1530."
  }
]
```

### Cara Menggunakan JSON Ini dalam Sistem Anda

Anda dapat memuat file JSON ini ke dalam variabel Python (`knowledge_base`) dan menggunakannya sebagai **Validator Logika** sebelum mengirim prompt ke LLM.

**Contoh Logic Python Integrasi:**

```python
import json

# 1. Load Knowledge Base
with open('cas_knowledge_base.json', 'r') as f:
    kb = json.load(f)

# 2. Fungsi Validasi Cerdas
def validate_prediction(predicted_cas, detected_ir_peaks, uv_data):
    # Cari data di KB
    target = next((item for item in kb if item["cas"] == predicted_cas), None)
    
    if not target:
        return "Unknown Compound in Database"

    # Cek IR Markers (Sederhana)
    missing_markers = []
    for marker in target["ir_markers"]:
        # Logika fuzzy matching sederhana (misal cek keyword)
        keyword = marker.split()[0] # Ambil kata pertama misal "Vinyl"
        if not any(keyword in peak for peak in detected_ir_peaks):
            missing_markers.append(marker)

    # Buat Insight untuk LLM
    prompt_context = f"""
    TARGET ANALISIS: {target['name']} (CAS {target['cas']})
    
    EXPECTED IR MARKERS: {', '.join(target['ir_markers'])}
    EXPECTED UV: {target['uv_expected']}
    AI INSIGHT: {target['ai_insight']}
    
    TEMUAN SISTEM:
    - IR Peaks Detected: {detected_ir_peaks}
    - UV Data: {uv_data}
    - Markers Hilang: {missing_markers if missing_markers else 'NONE (Perfect Match)'}
    
    INSTRUKSI:
    Jika ada marker wajib yang hilang (misal 'Vinyl' pada Styrene), nyatakan hasil MERAGUKAN. 
    Gunakan 'AI INSIGHT' untuk menjelaskan alasannya.
    """
    
    return prompt_context
```

### Tips Tambahan

Data di atas baru mencakup sekitar 15 sampel kunci dari list Anda.

  * **Prioritas:** Saya memprioritaskan senyawa yang memiliki gugus fungsi berbeda-beda (Aldehid, Keton, Alkohol, Aromatik, Nitril) agar model AI Anda belajar membedakan kelas kimia.
  * **Isomer:** Daftar Anda memiliki isomer (misal: 106-42-3 p-Xylene vs 108-38-3 m-Xylene). Untuk tahap awal, fokus pada gugus fungsi utama dulu. Membedakan isomer posisi (*ortho/meta/para*) membutuhkan analisis area *fingerprint* (600-900 cm⁻¹) yang sangat detail di `ir_markers`.