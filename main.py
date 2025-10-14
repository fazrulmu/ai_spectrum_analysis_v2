import argparse
import yaml
import os
import sys
from pathlib import Path

# --- Menambahkan 'src' ke path agar semua impor absolut berfungsi ---
# Ini adalah langkah kunci untuk mengatasi semua ModuleNotFoundError
SRC_PATH = Path(__file__).parent / 'src'
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

# --- Impor semua fungsi utama dari skrip-skrip lain ---
from src.data_analyzer.smart_downloader import main as run_downloader
from src.run_training import execute_training
from src.run_prediction import execute_prediction
from src.data_analyzer.generate_rule_statistics import main as run_rule_analysis
from label_and_visualize import main as run_visualization
# --- TAMBAHKAN IMPOR UNTUK FUNGSI AUDIT ---
from data_analyzer.run_data_audit import main as run_data_audit

def main():
    """
    Fungsi utama yang bertindak sebagai pusat komando (CLI) untuk seluruh proyek.
    """
    parser = argparse.ArgumentParser(
        description="Pipeline AI Spektral Terpusat.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", required=True, help="Pilih perintah yang akan dijalankan")

    # --- Perintah 'download' ---
    _ = subparsers.add_parser("download", help="Menjalankan modul Smart Downloader.")
    
    # --- Perintah 'train' ---
    parser_train = subparsers.add_parser("train", help="Menjalankan alur kerja pelatihan model.")
    parser_train.add_argument("--stage", choices=["base", "meta", "all"], default="all", help="Tahap pelatihan yang akan dijalankan.")
    parser_train.add_argument("--type", choices=["ir", "uv"], help="Tipe spektrum (hanya untuk --stage base).")

    # --- Perintah 'predict' ---
    parser_predict = subparsers.add_parser("predict", help="Membuat prediksi pada file spektrum baru.")
    parser_predict.add_argument("--ir", type=str, help="Path ke file spektrum IR (.jdx).")
    parser_predict.add_argument("--uv", type=str, help="Path ke file spektrum UV-Vis (.jdx).")
    
    # --- Perintah 'analyze' ---
    _ = subparsers.add_parser("analyze", help="Menjalankan analisis statistik pada aturan spektral.")

    # --- Perintah 'visualize' ---
    parser_visualize = subparsers.add_parser("visualize", help="Melabeli & memvisualisasikan puncak pada satu file spektrum.")
    parser_visualize.add_argument("--file", type=str, required=True, help="Path ke file spektrum IR (.jdx) yang akan dianalisis.")

    # --- PERINTAH BARU 'audit' DENGAN OPSI ---
    parser_audit = subparsers.add_parser("audit", help="Menjalankan audit kualitas data canggih.")
    parser_audit.add_argument('--limit', type=int, default=None, help='(Opsional) Batasi jumlah file yang akan diproses.')
    parser_audit.add_argument('--source', default=None, help='(Opsional) Path ke direktori spesifik yang berisi file JDX.')

    args = parser.parse_args()
    
    config_path = 'main_config.yaml'
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"❌ Error: File '{config_path}' tidak ditemukan.")
        return

    # === Logika Pemanggilan Fungsi (Orchestrator) ===
    if args.command == "download":
        print("--- Memanggil Modul Downloader ---")
        output_dir = config['paths']['raw_data_dir']
        os.makedirs(output_dir, exist_ok=True)
        run_downloader(output_dir=output_dir, config_path=config_path)
    
    elif args.command == "train":
        print("--- Memanggil Modul Pelatihan ---")
        if args.stage == "base" and not args.type:
            parser_train.error("Argumen --type ('ir' atau 'uv') diperlukan untuk --stage base.")
        execute_training(args, config)
    
    elif args.command == "predict":
        print("--- Memanggil Modul Prediksi ---")
        if not args.ir and not args.uv:
            parser_predict.error("Harap sediakan setidaknya satu file spektrum (--ir atau --uv).")
        execute_prediction(args, config)
        
    elif args.command == "analyze":
        print("--- Memanggil Modul Analisis Statistik ---")
        output_dir = config['paths']['statistics_dir']
        os.makedirs(output_dir, exist_ok=True)
        run_rule_analysis(output_dir=output_dir, config_path=config_path)
        
    elif args.command == "visualize":
        print("--- Memanggil Modul Visualisasi Label ---")
        output_dir = os.path.join(config['paths']['figures_dir'], 'labeled_spectra')
        os.makedirs(output_dir, exist_ok=True)
        run_visualization(output_dir=output_dir, config=config, target_file=args.file)

    elif args.command == "audit":
        print("--- 🔬 Memanggil Modul Audit Kualitas Data ---")
        output_dir = os.path.join(config['paths']['reports_dir'], 'data_audit')
        os.makedirs(output_dir, exist_ok=True)
        # Teruskan argumen 'limit' dan 'source' ke fungsi audit
        run_data_audit(config_path=config_path, output_dir=output_dir, limit=args.limit, source=args.source)

if __name__ == "__main__":
    main()