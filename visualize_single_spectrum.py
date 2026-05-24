import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def visualize_ir_spectrum_from_csv(filepath: Path, output_dir: Path = Path("reports/single_spectrum_visualizations")):
    """
    Memvisualisasikan spektrum IR dari file CSV yang diberikan.

    Args:
        filepath (Path): Path ke file CSV spektrum IR.
        output_dir (Path): Direktori untuk menyimpan plot yang dihasilkan.
    """
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"❌ Error: File tidak ditemukan di {filepath}")
        return

    molecule_id = df['molecule_id'].iloc[0] if not df.empty else filepath.stem
    
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{molecule_id}_ir_spectrum.png"

    plt.figure(figsize=(12, 6))
    plt.plot(df['x_value'], df['y_value'], color='blue')
    plt.title(f"Spektrum IR untuk {molecule_id}", fontsize=16)
    plt.xlabel("Wavenumber (cm⁻¹)", fontsize=12)
    plt.ylabel("Absorbance (Normalized)", fontsize=12)
    plt.gca().invert_xaxis()  # Membalik sumbu x untuk spektrum IR
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"✅ Plot spektrum IR berhasil disimpan di: {output_path}")

if __name__ == "__main__":
    # Path ke file CSV yang ingin Anda visualisasikan
    ir_spectrum_csv_path = Path("/home/acer/ai_spectrum_analysis_v2/data/standarize/processed_spectrum/IR_0_504-20-1.csv")
    
    # Panggil fungsi visualisasi
    visualize_ir_spectrum_from_csv(ir_spectrum_csv_path)