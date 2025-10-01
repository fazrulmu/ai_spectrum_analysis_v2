# smart_downloader.py
"""
Smart downloader untuk NIST WebBook (IR & UV-Vis JCAMP).
Fitur:
 - Scan folder per-CAS: hanya download jika IR/UV belum ada
 - Flexible scraping: coba link list dulu, bila gagal coba direct JCAMP URL (Index=0)
 - Retry + delay + random jitter + user-agent
 - Simpan metadata per-CAS di metadata.json
"""

import os
import time
import json
import random
import re
import requests
from bs4 import BeautifulSoup

# Konfigurasi
CAS_LIST_FILE = "cas_list.json"
BASE_OUTPUT_DIR = "data/raw/nist_database"
NIST_BASE_URL = "https://webbook.nist.gov"
REQUEST_HEADERS = {
    "User-Agent": "ai-spectrum-downloader/1.0 (+https://yourlab.example)"
}

# ------ Helpers ------
def has_ir_uv_files(output_dir):
    """Cek apakah dalam folder sudah terdapat file IR dan/atau UV"""
    found_ir = False
    found_uv = False
    if not os.path.exists(output_dir):
        return found_ir, found_uv
    for fn in os.listdir(output_dir):
        name = fn.lower()
        if name.endswith(".jdx") or name.endswith(".dx") or name.endswith(".txt"):
            if "ir" in name and ("uv" not in name):
                found_ir = True
            if "uv" in name or "uvvis" in name or "uv-vis" in name:
                found_uv = True
            # beberapa file penamaan lain: cek kata IR atau UV di file name
            if re.search(r"[\-_]ir[\-_\.]", name) or re.search(r"_ir_", name) or name.endswith("_ir.jdx"):
                found_ir = True
            if re.search(r"uv", name):
                found_uv = True
    return found_ir, found_uv

def get_soup(url, session=None, retries=3, delay=5, timeout=20):
    """Get BeautifulSoup with retry and jitter"""
    session = session or requests.Session()
    for attempt in range(retries):
        try:
            resp = session.get(url, headers=REQUEST_HEADERS, timeout=timeout)
            resp.raise_for_status()
            return BeautifulSoup(resp.text, "html.parser")
        except Exception as e:
            print(f"    -> Gagal akses {url} (attempt {attempt+1}/{retries}): {e}")
            if attempt < retries - 1:
                wait = delay + random.random()*2
                print(f"       Menunggu {wait:.1f}s sebelum retry...")
                time.sleep(wait)
            else:
                return None

def try_download_url(url, filepath, session=None, timeout=25):
    """Coba download file JCAMP dari url, simpan ke filepath. Return True jika ok."""
    session = session or requests.Session()
    try:
        resp = session.get(url, headers=REQUEST_HEADERS, timeout=timeout)
        if resp.status_code == 200 and resp.content and len(resp.content) > 100:
            with open(filepath, "wb") as f:
                f.write(resp.content)
            return True
        else:
            print(f"      -> Gagal download (status {resp.status_code}) dari {url}")
            return False
    except Exception as e:
        print(f"      -> Exception saat mendownload {url}: {e}")
        return False

# ------ Download IR ------
def download_ir_spectra(cas, output_dir, metadata_list, session=None):
    print("  -> Mencari spektrum IR...")
    cas_id = cas.replace("-", "")
    session = session or requests.Session()

    # 1. Coba akses halaman index IR-SPEC
    index_url = f"{NIST_BASE_URL}/cgi/cbook.cgi?ID=C{cas_id}&Units=SI&Type=IR-SPEC"
    soup = get_soup(index_url, session=session)

    downloaded_any = False

    if soup:
        # cari semua link spektrum IR
        links = [a['href'] for a in soup.select("a[href*='JCAMP=C']") if "Type=IR" in a['href']]
        if links:
            print(f"    -> Ditemukan {len(links)} variasi spektrum IR. Memulai unduhan...")
            for i, href in enumerate(links):
                jcamp_url = href if href.startswith("http") else (NIST_BASE_URL + href)
                filename = f"{cas}_IR_{i}.jdx"
                filepath = os.path.join(output_dir, filename)

                if os.path.exists(filepath):
                    print(f"      -> {filename} sudah ada, skip.")
                    continue

                ok = try_download_url(jcamp_url, filepath, session=session)
                if ok:
                    metadata_list.append({"cas": cas, "type": "IR", "url": jcamp_url, "local": filename})
                    print(f"      -> Berhasil unduh: {filename}")
                    downloaded_any = True
                    time.sleep(0.5 + random.random()*0.5)
        else:
            print("    -> Tidak ada link spektrum IR ditemukan di halaman daftar.")
    
    # 2. Kalau gagal, fallback ke direct link Index=0
    if not downloaded_any:
        fallback = f"{NIST_BASE_URL}/cgi/cbook.cgi?JCAMP=C{cas_id}&Index=0&Type=IR"
        filename = f"{cas}_IR_0.jdx"
        filepath = os.path.join(output_dir, filename)

        if not os.path.exists(filepath):
            ok = try_download_url(fallback, filepath, session=session)
            if ok:
                metadata_list.append({"cas": cas, "type": "IR", "url": fallback, "local": filename})
                print(f"      -> Berhasil unduh (fallback): {filename}")
                downloaded_any = True

            for idx in range(0, 3):
                jcamp_url = f"{NIST_BASE_URL}/cgi/cbook.cgi?JCAMP=C{cas_id}&Index={idx}&Type=IR"
                filename = f"{cas}_IR_{idx}.jdx"
                filepath = os.path.join(output_dir, filename)
                if os.path.exists(filepath):
                    print(f"      -> {filename} sudah ada, skip.")
                    continue
                ok = try_download_url(jcamp_url, filepath, session=session)
                if ok:
                    metadata_list.append({"cas": cas, "type": "IR", "url": jcamp_url, "local": filename})
                    print(f"      -> Berhasil unduh fallback: {filename}")
                    downloaded_any = True
                    break
                else:
                    time.sleep(1 + random.random()*0.8)

    else:
        print("    -> Halaman IR tidak dapat diakses, mencoba direct JCAMP fallback...")
        jcamp_url = f"{NIST_BASE_URL}/cgi/cbook.cgi?JCAMP=C{cas_id}&Index=0&Type=IR"
        filename = f"{cas}_IR_0.jdx"
        filepath = os.path.join(output_dir, filename)
        if try_download_url(jcamp_url, filepath, session=session):
            metadata_list.append({"cas": cas, "type": "IR", "url": jcamp_url, "local": filename})
            print(f"      -> Berhasil unduh fallback: {filename}")
            downloaded_any = True

    if not downloaded_any:
        print(f"    -> Gagal mengunduh IR untuk {cas} (tidak ada link atau semua gagal).")
    return downloaded_any

# ------ Download UV ------
def download_uv_spectra(cas, output_dir, metadata_list, session=None):
    print("  -> Mencari spektrum UV-Vis...")
    cas_id = cas.replace("-", "")
    uv_index_url = f"{NIST_BASE_URL}/cgi/cbook.cgi?ID=C{cas_id}&Units=SI&Mask=400"
    session = session or requests.Session()
    soup = get_soup(uv_index_url, session=session)
    downloaded_any = False

    if soup:
        links = []
        for a in soup.find_all("a", href=True):
            href = a["href"]
            href_low = href.lower()
            if "jcamp=c" in href_low and ("type=uv" in href_low or "type=uvvis" in href_low or "uvvis" in href_low):
                links.append(href)
            if "type=uv" in href_low or "uvvis" in href_low:
                links.append(href)
        links = list(dict.fromkeys(links))
        if links:
            print(f"    -> Menemukan {len(links)} link JCAMP UV-Vis. Mencoba unduh...")
            for href in links:
                jcamp_url = href if href.startswith("http") else (NIST_BASE_URL + href)
                index_match = re.search(r"index=(\d+)", jcamp_url, flags=re.IGNORECASE)
                idx = index_match.group(1) if index_match else "0"
                filename = f"{cas}_UVVis_{idx}.jdx"
                filepath = os.path.join(output_dir, filename)
                if os.path.exists(filepath):
                    print(f"      -> {filename} sudah ada, skip.")
                    continue
                ok = try_download_url(jcamp_url, filepath, session=session)
                if ok:
                    metadata_list.append({"cas": cas, "type": "UVVis", "url": jcamp_url, "local": filename})
                    print(f"      -> Berhasil unduh: {filename}")
                    downloaded_any = True
                    time.sleep(0.8 + random.random()*0.6)
                else:
                    fallback = f"{NIST_BASE_URL}/cgi/cbook.cgi?JCAMP=C{cas_id}&Index={idx}&Type=UVVis"
                    print(f"      -> Mencoba fallback: {fallback}")
                    ok2 = try_download_url(fallback, filepath, session=session)
                    if ok2:
                        metadata_list.append({"cas": cas, "type": "UVVis", "url": fallback, "local": filename})
                        print(f"      -> Berhasil unduh(w/ fallback): {filename}")
                        downloaded_any = True
                        time.sleep(0.8 + random.random()*0.6)
        else:
            print("    -> Tidak ditemukan link JCAMP khusus UV di halaman. Mencoba direct JCAMP fallback (Index=0)...")
            for idx in range(0, 3):
                jcamp_url = f"{NIST_BASE_URL}/cgi/cbook.cgi?JCAMP=C{cas_id}&Index={idx}&Type=UVVis"
                filename = f"{cas}_UVVis_{idx}.jdx"
                filepath = os.path.join(output_dir, filename)
                if os.path.exists(filepath):
                    continue
                ok = try_download_url(jcamp_url, filepath, session=session)
                if ok:
                    metadata_list.append({"cas": cas, "type": "UVVis", "url": jcamp_url, "local": filename})
                    print(f"      -> Berhasil unduh fallback: {filename}")
                    downloaded_any = True
                    break
                else:
                    time.sleep(1 + random.random()*0.8)
    else:
        print("    -> Halaman UV tidak dapat diakses, mencoba direct JCAMP fallback...")
        jcamp_url = f"{NIST_BASE_URL}/cgi/cbook.cgi?JCAMP=C{cas_id}&Index=0&Type=UVVis"
        filename = f"{cas}_UVVis_0.jdx"
        filepath = os.path.join(output_dir, filename)
        if try_download_url(jcamp_url, filepath, session=session):
            metadata_list.append({"cas": cas, "type": "UVVis", "url": jcamp_url, "local": filename})
            print(f"      -> Berhasil unduh fallback: {filename}")
            downloaded_any = True

    if not downloaded_any:
        print(f"    -> Gagal mengunduh UV-Vis untuk {cas} (tidak ada link atau semua gagal).")
    return downloaded_any

# ------ Runner ------
def run_download():
    """Wrapper untuk dipanggil dari main.py"""
    # load CAS list
    try:
        with open(CAS_LIST_FILE, "r") as f:
            cas_list = json.load(f)
    except Exception as e:
        print(f"[ERROR] Gagal buka {CAS_LIST_FILE}: {e}")
        return

    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
    session = requests.Session()

    for cas in cas_list:
        try:
            print(f"\n[PROSES] Memulai untuk CAS: {cas}")
            output_dir = os.path.join(BASE_OUTPUT_DIR, cas)
            os.makedirs(output_dir, exist_ok=True)

            # cek eksisting IR / UV
            has_ir, has_uv = has_ir_uv_files(output_dir)
            if has_ir and has_uv:
                print(f"  -> IR & UV sudah ada untuk {cas}. Melewati.")
                continue

            metadata = []
            # download IR jika belum ada
            if not has_ir:
                ok_ir = download_ir_spectra(cas, output_dir, metadata, session=session)
            else:
                ok_ir = False
                print("  -> IR sudah ada, skip download IR.")

            # download UV jika belum ada
            if not has_uv:
                ok_uv = download_uv_spectra(cas, output_dir, metadata, session=session)
            else:
                ok_uv = False
                print("  -> UV sudah ada, skip download UV.")

            if metadata:
                metadata_path = os.path.join(output_dir, "metadata.json")
                with open(metadata_path, "w", encoding="utf-8") as mf:
                    json.dump(metadata, mf, indent=2)
                print(f"  -> Metadata disimpan ke {metadata_path}")

            # jeda kecil sebelum lanjut CAS berikutnya
            time.sleep(1 + random.random()*0.8)

        except KeyboardInterrupt:
            print("\n[STOP] Download dihentikan oleh user.")
            break
        except Exception as e:
            print(f"[ERROR] Saat proses CAS {cas}: {e}")
            # lanjut ke CAS berikutnya
            continue

    print("\n[INFO] Selesai memproses daftar CAS.")
