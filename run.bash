python main.py --help


# Melatih semua model (IR, UV, lalu Meta)
python main.py train --stage all

# Melatih hanya model dasar IR
python main.py train --stage base --type ir

# Melatih hanya meta-model (setelah base models selesai)
python main.py train --stage meta


# Prediksi ensemble (paling akurat)
python main.py predict --ir samples_to_predict/50-78-2_IR.jdx --uv samples_to_predict/50-78-2_UV-Vis_0.jdx

# Prediksi IR saja
python main.py predict --ir /path/ke/file.jdx