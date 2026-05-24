# src/data_farmer/smart_downloader.py (Versi Final dengan Perbaikan Error)

import os
import requests
import json
import time
import yaml
import re
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_URL = "https://webbook.nist.gov"
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

def download_jdx_with_retry(url, save_path, retries=10000, backoff_factor=1):
    """
    Fungsi unduh yang lebih tangguh dengan timeout yang cukup dan mekanisme retry.
    """
    for attempt in range(retries):
        try:
            response = requests.get(url, headers=HEADERS, timeout=130, allow_redirects=True)
            response.raise_for_status()
            
            if 'content-type' in response.headers and 'jcamp-dx' not in response.headers['content-type'].lower():
                return False, "Konten bukan JDX"
            
            if not response.content:
                return False, "File kosong"

            with open(save_path, 'wb') as f:
                f.write(response.content)
            return True, "Sukses"
            
        except requests.exceptions.Timeout:
            # Non-aktifkan pesan print untuk menjaga output tetap bersih, kecuali jika diperlukan untuk debug
            # print(f"Timeout saat mengunduh {url}. Mencoba lagi ({attempt + 1}/{retries})...")
            time.sleep(backoff_factor * (2 ** attempt))
            
        except requests.RequestException:
            return False, "Gagal terhubung"
            
    return False, f"Gagal setelah {retries} kali percobaan"


def get_links_from_html(html_content):
    """Mengekstrak semua link JDX dari konten HTML."""
    relative_links = re.findall(r'href="(/cgi/cbook\.cgi\?JCAMP=[^"]+)"', html_content)
    if not relative_links:
        return []
    full_links = [BASE_URL + link.replace('&amp;', '&') for link in relative_links]
    return list(set(full_links))

def download_all_for_cas(cas_no, save_root_dir):
    """Mengunduh semua spektrum untuk satu CAS number dengan logika proaktif dan retry."""
    cas_save_dir = os.path.join(save_root_dir, cas_no)
    os.makedirs(cas_save_dir, exist_ok=True)
    
    downloaded_files = set()
    success_count = 0
    
    # Strategi 1: Tembak langsung URL IR
    direct_ir_url = f"{BASE_URL}/cgi/cbook.cgi?JCAMP=C{cas_no.replace('-', '')}&Index=0&Type=IR"
    ir_filename = f"IR_direct_{cas_no}.jdx"
    ir_save_path = os.path.join(cas_save_dir, ir_filename)
    
    success, _ = download_jdx_with_retry(direct_ir_url, ir_save_path)
    if success:
        success_count += 1
        downloaded_files.add(ir_filename)

    # Strategi 2: Fallback ke parsing HTML
    search_url = f"{BASE_URL}/cgi/cbook.cgi?ID=C{cas_no.replace('-', '')}&Units=SI&Mask=80"
    try:
        response = requests.get(search_url, headers=HEADERS, timeout=15)
        if response.ok:
            other_links = get_links_from_html(response.text)
            for i, link in enumerate(other_links):
                spec_type_match = re.search(r'Type=([A-Z\-]+)', link)
                spec_type = spec_type_match.group(1) if spec_type_match else "UNKNOWN"
                index_match = re.search(r'Index=([0-9]+)', link)
                index = index_match.group(1) if index_match else i
                
                filename = f"{spec_type}_{index}_{cas_no}.jdx"
                
                if filename not in downloaded_files:
                    save_path = os.path.join(cas_save_dir, filename)
                    
                    # --- INI ADALAH BARIS YANG DIPERBAIKI ---
                    link_success, _ = download_jdx_with_retry(link, save_path)
                    
                    if link_success:
                        success_count += 1
                        downloaded_files.add(filename)
                    time.sleep(0.2)
    except requests.RequestException:
        pass

    return success_count

def main(output_dir=".", config_path="main_config.yaml"):
    """Fungsi utama untuk menjalankan proses pengunduhan data spektrum."""
    print("--- 🚚 Memulai Modul Smart Downloader (Versi Stabil & Diperbaiki) ---")
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"❌ Error: File konfigurasi '{config_path}' tidak ditemukan.")
        return

    cas_list_path = config['paths'].get('cas_list_json', 'cas_list.json')
    raw_data_dir = output_dir

    try:
        with open(cas_list_path, 'r') as f:
            cas_list = json.load(f)
    except FileNotFoundError:
        print(f"❌ Error: File daftar CAS '{cas_list_path}' tidak ditemukan.")
        return
        
    print(f"📚 Ditemukan {len(cas_list)} CAS numbers untuk diunduh.")
    
    total_downloaded = 0
    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_cas = {executor.submit(download_all_for_cas, cas, raw_data_dir): cas for cas in cas_list}
        
        for future in tqdm(as_completed(future_to_cas), total=len(cas_list), desc="Mengunduh Spektrum"):
            cas = future_to_cas[future]
            try:
                num_downloaded = future.result()
                if num_downloaded > 0:
                    total_downloaded += num_downloaded
            except Exception as exc:
                print(f"\nCAS {cas} menghasilkan error: {exc}")

    print(f"\n✅ Proses Selesai. Total file spektrum yang berhasil diunduh: {total_downloaded}")
    print(f"📁 Semua data disimpan di dalam folder: {raw_data_dir}")

if __name__ == '__main__':
    default_folder = "default_output/smart_downloader"
    if not os.path.exists(default_folder):
        os.makedirs(default_folder)
        
    print(f"Menjalankan {__file__} secara mandiri untuk pengujian...")
    main(output_dir=default_folder)