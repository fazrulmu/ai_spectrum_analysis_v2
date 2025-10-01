# main.py (Pusat Kontrol CLI)

import argparse
import yaml

# Impor fungsi eksekusi dari folder scripts
from scripts.run_training import execute_training
from scripts.run_prediction import execute_prediction

def main():
    parser = argparse.ArgumentParser(
        description="Pipeline AI Spektral Terpusat.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", required=True, help="Pilih perintah yang akan dijalankan")

    # --- Perintah 'train' ---
    parser_train = subparsers.add_parser("train", help="Menjalankan alur kerja pelatihan model.")
    parser_train.add_argument(
        "--stage",
        choices=["base", "meta", "all"],
        default="all",
        help="Tahap pelatihan:\n"
            "  'base': Latih model dasar (perlu --type).\n"
            "  'meta': Latih meta-model ensemble.\n"
            "  'all': Jalankan semua tahap secara berurutan (default)."
    )
    parser_train.add_argument("--type", choices=["ir", "uv"], help="Tipe spektrum (hanya untuk --stage base).")

    # --- Perintah 'predict' ---
    parser_predict = subparsers.add_parser("predict", help="Melakukan prediksi pada spektrum baru.")
    parser_predict.add_argument("--ir", help="Path ke file JDX spektrum IR.")
    parser_predict.add_argument("--uv", help="Path ke file JDX spektrum UV-Vis.")

    # --- (Placeholder) Perintah 'evaluate' untuk masa depan ---
    # parser_evaluate = subparsers.add_parser("evaluate", help="Mengevaluasi performa model.")
    # parser_evaluate.add_argument("--model", required=True, help="Model yang akan dievaluasi.")

    args = parser.parse_args()
    with open('main_config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Memanggil fungsi yang sesuai berdasarkan perintah
    if args.command == "train":
        if args.stage == "base" and not args.type:
            parser_train.error("Argumen --type ('ir' atau 'uv') diperlukan untuk --stage base.")
        execute_training(args, config)
    
    elif args.command == "predict":
        if not args.ir and not args.uv:
            parser_predict.error("Harap sediakan setidaknya satu file spektrum (--ir atau --uv).")
        execute_prediction(args, config)

if __name__ == "__main__":
    main()