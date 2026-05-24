import os
import yaml

# Path ke file YAML (relatif dari root project)
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "configs", "main_config.yaml")

def load_config(path: str = CONFIG_PATH):
    """
    Load konfigurasi dari file YAML.
    """
    path = os.path.abspath(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file tidak ditemukan: {path}")
    with open(path, "r") as f:
        return yaml.safe_load(f)
