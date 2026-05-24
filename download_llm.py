from huggingface_hub import hf_hub_download
import os

def download_model():
    repo_id = "Qwen/Qwen2.5-0.5B-Instruct-GGUF"
    filename = "qwen2.5-0.5b-instruct-q4_k_m.gguf"
    save_dir = "saved_models/llm"
    
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"⬇️  Downloading {filename} from {repo_id}...")
    path = hf_hub_download(repo_id=repo_id, filename=filename, local_dir=save_dir)
    print(f"✅ Model saved to: {path}")
    return path

if __name__ == "__main__":
    download_model()
