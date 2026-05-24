import pandas as pd
import numpy as np
from scipy.signal import find_peaks, peak_widths
import argparse
from pathlib import Path
from tqdm import tqdm
import sys



"""
| Perbandingan                      | Kisaran Δν (cm⁻¹) | Referensi                   |
| --------------------------------- | ----------------- | --------------------------- |
| Gas → Liquid (C=O)                | −5 s.d. −15       | Pavia et al. 2015           |
| Gas → Solid                       | −10 s.d. −25      | Nakamoto 2009               |
| Liquid → Solid (KBr pellet)       | −10 s.d. −20      | Aldrich IR Collection       |
| Solution in CCl₄ vs neat liquid   | −5 s.d. −10       | Spectrochim. Acta A (2010)  |
| Broadening (FWHM, relatif ke gas) | ×1.0 – ×1.5       | SOC Spectroscopy Data, 2007 |



| Komponen                           | Dapat diperoleh dari                                                                           | Referensi / Contoh                                                                                                                                                  |
| ---------------------------------- | ---------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Shift (cm⁻¹)**                   | Literatur spektroskopi, database spektrum (NIST, SDBS, Aldrich FT-IR Atlas)                    | Misalnya: *“Typical red-shift of C=O band from gas to liquid: 5–15 cm⁻¹”* (Nakamoto, *Infrared and Raman Spectra of Inorganic and Coordination Compounds*, 6th ed.) |
| **Broad_factor**                   | Dari analisis bentuk puncak (FWHM) spektrum eksperimental untuk berbagai fasa                  | Contoh: *FWHM solid ≈ 1.3× liquid; FWHM gas ≈ 0.8× liquid* (Society for Applied Spectroscopy, 2005)                                                                 |
| **Description & Expected Pattern** | Dari handbook IR (KBr, Nujol, CS₂, CCl₄) dan literatur teknik FT-IR                            | Lihat: Pavia et al., *Introduction to Spectroscopy*, 5th ed., Bab 2; atau "IR Spectra of Gaseous, Liquid, and Solid States" (Spectrochimica Acta, 2010)             |
| **Calibration Example**            | Eksperimen sederhana dengan senyawa model (formaldehyde, acetone, ethanol) pada berbagai state | Kamu bisa ambil data FT-IR open-source (NIST WebBook) dan hitung pergeserannya sendiri                                                                              |


"""
import json

# === STATE-ADJUSTMENT UNTUK ANALISIS PERGESERAN ===
LEARNED_ADJUSTMENT_FILE = Path("data/reports/state_shifts/learned_state_adjustment.json")

def load_learned_adjustments():
    """Memuat nilai adjustment yang telah dipelajari dari file JSON."""
    if not LEARNED_ADJUSTMENT_FILE.exists():
        print(f"⚠️ File learned adjustment tidak ditemukan di {LEARNED_ADJUSTMENT_FILE}. Menggunakan default.")
        return {}
    
    try:
        with open(LEARNED_ADJUSTMENT_FILE, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"⚠️ Gagal memuat learned adjustment: {e}")
        return {}

# Default hardcoded values
state_adjustment = {
    "gas": {
        "shift": +10,
        "broad_factor": 0.8,
        "description": "Gas-phase peaks shift slightly higher (blue-shift) with narrow width.",
        "expected_pattern": "narrow, low overlap, rotational-vibrational features"
    },
    "solution": {
        "shift": -5,
        "broad_factor": 1.1,
        "description": "Solution state typically shows mild red-shift due to solvent interactions.",
        "expected_pattern": "moderate broadening, solvent interference"
    },
    "solution_ccl4": {
        "shift": -8,
        "broad_factor": 1.1,
        "description": "CCl4 solution causes moderate red-shift and slight suppression of polarity-sensitive peaks.",
        "expected_pattern": "clear window above 1300 cm⁻¹"
    },
    "liquid": {
        "shift": -10,
        "broad_factor": 1.2,
        "description": "Liquid phase broadens peaks and causes red-shift due to hydrogen bonding.",
        "expected_pattern": "broad OH stretch, high baseline"
    },
    "solid_kbr": {
        "shift": -20,
        "broad_factor": 1.3,
        "description": "KBr disc environment induces red-shift and moderate broadening.",
        "expected_pattern": "distinct disc absorption near 350 cm⁻¹"
    },
    "solid_mull": {
        "shift": -15,
        "broad_factor": 1.4,
        "description": "Mull (Nujol/Fluorolube) introduces strong red-shift and broad baseline features.",
        "expected_pattern": "broad bands at 1450–1380 cm⁻¹"
    },
    "unknown": {
        "shift": 0,
        "broad_factor": 1.0,
        "description": "Unknown or mixed state; reference baseline condition.",
        "expected_pattern": "uncertain baseline, use as reference"
    }
}

# Update with learned values
learned_values = load_learned_adjustments()
for state_name, adj_data in learned_values.items():
    # Normalize key to match our internal keys if possible, or just add as new
    # Mapping simple keys to our internal keys if they match case-insensitive
    normalized_key = state_name.lower()
    
    # If the learned key matches one of our default keys (case-insensitive), update it
    found = False
    for default_key in state_adjustment.keys():
        if default_key == normalized_key:
            state_adjustment[default_key].update(adj_data)
            found = True
            break
    
    # If not found (e.g. specific gas pressure states), add it as is
    if not found:
        state_adjustment[state_name] = adj_data





# === KLASIFIKASI OTOMATIS STATE DARI SPEKTRUM ===
def classify_state_from_spectrum(wavenumber, intensity):
    wn, intensity = np.array(wavenumber), np.array(intensity)
    if len(wn) < 50:
        return "unknown", 0.0

    baseline_std = np.std(intensity[:50])
    peaks, _ = find_peaks(intensity, prominence=0.02)
    widths = peak_widths(intensity, peaks, rel_height=0.5)[0] if len(peaks) > 0 else []
    mean_width = np.mean(widths) if len(widths) > 0 else 0

    # heuristik sederhana
    if len(peaks) < 10 and mean_width < 8:
        return "gas", 0.9
    elif 10 <= len(peaks) < 30 and mean_width < 12:
        return "solution", 0.8
    elif mean_width >= 15 and len(peaks) >= 30:
        return "liquid", 0.85
    elif mean_width >= 20:
        return "solid_mull", 0.9
    elif 10 < mean_width < 20:
        return "solid_kbr", 0.8
    else:
        return "unknown", 0.5

# === ANALISIS PERGESERAN PUNCAK ===
def analyze_peak_shifts(input_path: Path):
    try:
        data = pd.read_csv(input_path)
        wn, inten = np.array(data["x_value"]), np.array(data["y_value"])
    except Exception as e:
        print(f"❌ Gagal membaca file {input_path}: {e}")
        return None

    # klasifikasi atau baca state
    sample_state = data["state"].iloc[0] if "state" in data.columns else "unknown"
    auto_state, conf_state = classify_state_from_spectrum(wn, inten)
    if sample_state == "unknown" or conf_state > 0.75:
        sample_state = auto_state
    tqdm.write(f"🔎 State terdeteksi: {sample_state} (conf={conf_state})")

    adj = state_adjustment.get(sample_state, state_adjustment["unknown"])
    shift = adj["shift"]

    # deteksi puncak
    peaks, _ = find_peaks(inten, prominence=0.01)
    if len(peaks) == 0:
        print("🤷 Tidak ada puncak signifikan ditemukan.")
        return None

    widths = peak_widths(inten, peaks, rel_height=0.5)[0]
    results = []

    for i, p in enumerate(peaks):
        wn_orig = wn[p]
        inten_val = inten[p]
        width_val = widths[i]
        wn_shifted = wn_orig + shift

        results.append({
            "peak_index": i + 1,
            "center_cm-1": round(wn_orig, 2),
            "intensity": round(float(inten_val), 3),
            "FWHM": round(float(width_val), 2),
            "sample_state": sample_state,
            "predicted_shift_cm-1": shift,
            "shifted_position_cm-1": round(wn_shifted, 2),
            "normalized_position": round((wn_shifted - np.mean(wn)) / np.std(wn), 4),
            "broad_factor": adj["broad_factor"],
            "description": adj["description"],
            "expected_pattern": adj["expected_pattern"],
            "predicted_state_confidence": round(conf_state, 2)
        })

    return pd.DataFrame(results)

# === ANALISIS BATCH ===
def batch_analyze_shifts():
    INDEX_FILE_PATH = Path("data/standarize/dataset_index.csv")
    OUTPUT_DIR = Path("data/reports/state_shifts")
    OUTPUT_FILE_PATH = OUTPUT_DIR / "all_state_shift_analysis.csv"

    print("🚀 Analisis pergeseran puncak IR berdasarkan state...")
    try:
        index_df = pd.read_csv(INDEX_FILE_PATH)
    except FileNotFoundError:
        print(f"❌ File indeks tidak ditemukan di: {INDEX_FILE_PATH}")
        print("   Pastikan Anda telah menjalankan skrip 'preprocess_universal.py' terlebih dahulu.")
        print("   Skrip tersebut akan menghasilkan file indeks yang diperlukan untuk analisis ini.")
        print("   Contoh: python preprocess_universal.py")
        return

    ir_files_df = index_df[index_df["processed_filepath"].str.contains("processed_spectrum/IR_", na=False)]
    if ir_files_df.empty:
        print("🤷 Tidak ada file IR ditemukan.")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_results = []

    for _, row in tqdm(ir_files_df.iterrows(), total=len(ir_files_df), desc="Analisis Spektrum"):
        processed_path = Path(row["processed_filepath"])
        original_filename = row["original_filename"]
        df = analyze_peak_shifts(processed_path)
        if df is not None and not df.empty:
            df.insert(0, "original_filename", original_filename)
            all_results.append(df)

    if not all_results:
        print("❌ Tidak ada hasil.")
        return

    final_df = pd.concat(all_results, ignore_index=True)
    final_df.to_csv(OUTPUT_FILE_PATH, index=False)
    print(f"🎉 Analisis selesai. {len(final_df)} puncak dianalisis.")
    print(f"💾 Hasil disimpan di: {OUTPUT_FILE_PATH}")

# === CLI ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IR State Peak Shift Analyzer (AI-ready)")
    parser.add_argument("input_file", nargs="?", type=Path, default=None)
    parser.add_argument("--batch", action="store_true", help="Analisis batch semua file IR")
    args = parser.parse_args()

    if args.batch:
        batch_analyze_shifts()
    elif args.input_file:
        df = analyze_peak_shifts(args.input_file)
        if df is not None and not df.empty:
            out_dir = Path("reports/state_shifts")
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"shift_{args.input_file.stem}.csv"
            df.to_csv(out_path, index=False)
            print(f"✅ Hasil disimpan di: {out_path}")
    else:
        parser.print_help()
        sys.exit(1)
