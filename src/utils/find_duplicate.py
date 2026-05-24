import os
import glob
import hashlib # Library baru untuk membuat hash

# --- KONFIGURASI ---
SPECTRA_DIRECTORY = "data/raw/nist_jdx"
DRY_RUN = False # Ubah ke False untuk benar-benar menghapus file

# --- FUNGSI BARU UNTUK MENGHITUNG HASH ---
def calculate_sha256(filepath):
    """Membaca file dan menghitung sidik jari SHA-256 yang unik."""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f: # Buka file dalam mode binary ('rb')
        # Baca file dalam potongan kecil untuk efisiensi memori
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest() # Kembalikan sebagai string heksadesimal

# --- SCRIPT UTAMA YANG DIPERBARUI ---
def find_and_remove_duplicates(directory):
    """Mencari dan menghapus file yang 100% identik menggunakan hash SHA-256."""
    
    if not os.path.isdir(directory):
        print(f"Error: Folder '{directory}' tidak ditemukan.")
        return
        
    print(f"Memindai semua subfolder di dalam '{directory}'...")
    jdx_files = glob.glob(os.path.join(directory, '**', '*.jdx'), recursive=True)
    
    hashes = {}
    duplicates_to_remove = []
    
    print(f"Total ditemukan {len(jdx_files)} file .jdx untuk diperiksa.")
    
    for file_path in jdx_files:
        try:
            # Hitung sidik jari hash untuk file ini
            file_hash = calculate_sha256(file_path)
            
            # Cek apakah hash ini sudah pernah ditemukan
            if file_hash in hashes:
                original_file = hashes[file_hash]
                print(f"  -> Duplikat ditemukan: '{os.path.basename(file_path)}' identik dengan '{os.path.basename(original_file)}'")
                duplicates_to_remove.append(file_path)
            else:
                hashes[file_hash] = file_path
                
        except Exception as e:
            print(f"  -> Gagal memproses file {os.path.basename(file_path)}: {e}")
            
    print(f"\nTotal ditemukan {len(duplicates_to_remove)} file duplikat.")
    
    if not duplicates_to_remove:
        print("Tidak ada tindakan yang diperlukan.")
        return

    if DRY_RUN:
        print("\n[DRY RUN] Mode aman aktif. Tidak ada file yang akan dihapus.")
        print("File yang akan dihapus jika DRY_RUN = False:")
        for f in duplicates_to_remove:
            print(f"  - {f}")
    else:
        print("\n[LIVE RUN] Menghapus file duplikat...")
        for f in duplicates_to_remove:
            try:
                os.remove(f)
                print(f"  - Berhasil menghapus: {f}")
            except Exception as e:
                print(f"  - Gagal menghapus {f}: {e}")
        print("Proses penghapusan selesai.")

# --- Eksekusi ---
if __name__ == "__main__":
    find_and_remove_duplicates(SPECTRA_DIRECTORY)