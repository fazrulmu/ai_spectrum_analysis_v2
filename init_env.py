"""
init_env.py — Setup environment untuk proyek ai_spectrum_analysis_v2
Menjamin semua modul dalam src/ dapat diimpor di Jupyter / script eksternal.
"""

import os
import sys
import importlib.util
from pathlib import Path

def setup_environment():
    # 1️⃣ Tentukan root proyek (1 tingkat di atas file ini)
    project_root = Path(__file__).resolve().parent
    src_path = project_root / "src"
    data_path = project_root / "data"

    # 2️⃣ Tambahkan src/ ke sys.path jika belum ada
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

    # 3️⃣ Pastikan file __init__.py ada di src/
    init_file = src_path / "__init__.py"
    if not init_file.exists():
        init_file.touch()
        print(f"🧩 Membuat __init__.py di {init_file}")

    # 4️⃣ Deteksi environment Jupyter / terminal
    is_jupyter = False
    try:
        get_ipython  # type: ignore
        is_jupyter = True
    except NameError:
        pass

    print("📁 Project root:", project_root)
    print("📦 SRC path:", src_path)
    print("📦 DATA path:", data_path)
    print(f"🧠 Environment: {'Jupyter Notebook' if is_jupyter else 'Terminal/Python Script'}")

    # 5️⃣ Tes modul utama
    try:
        spec = importlib.util.find_spec("data_processing")
        if spec is None:
            raise ImportError("data_processing belum bisa diimpor")
        else:
            print("✅ Modul 'data_processing' ditemukan.")
    except Exception as e:
        print("🔥 ERROR:", e)

    try:
        spec = importlib.util.find_spec("jcamp")
        if spec is None:
            raise ImportError("Modul 'jcamp' tidak ditemukan. Silakan instal dengan: pip install jcamp")
        else:
            print("✅ Modul 'jcamp' ditemukan.")
    except Exception as e:
        print("🔥 ERROR:", e)

    return {
        "root": project_root,
        "src": src_path,
        "data": data_path
    }

if __name__ == "__main__":
    setup_environment()
