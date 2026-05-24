# Panduan Instalasi ROCm & PyTorch untuk Fedora (Distrobox)

**PENTING:**
Anda menjalankan Fedora di dalam **Distrobox** (Container).

1.  **Driver GPU (Kernel):** TIDAK BOLEH diinstall di sini. Driver harus sudah terinstall di OS Utama (Fin OS).
2.  **User-Space (PyTorch):** Bisa diinstall di sini menggunakan `pip`.

Karena Anda menggunakan **Fedora**, perintah `apt` (Ubuntu) tidak akan jalan. Gunakan panduan di bawah ini.

## 1. Cek Driver (Dari dalam Container)

Pastikan container bisa melihat GPU dari Host.

```bash
rocminfo | grep "Marketing Name"
```

_Jika muncul nama GPU (misal Radeon 780M), berarti Host sudah aman._

## 2. Install Dependency Dasar (Fedora)

Install library yang dibutuhkan PyTorch.

```bash
sudo dnf install -y python3-pip python3-devel gcc gcc-c++ rocm-hip-runtime rocm-opencl-runtime
```

_(Catatan: Paket `rocm-_` di repo Fedora mungkin opsional jika kita pakai wheel PyTorch yang sudah bundle runtime, tapi bagus untuk jaga-jaga).\*

## 3. Install PyTorch ROCm (Via Pip)

Ini adalah langkah paling penting. Kita gunakan build resmi untuk Linux.

**Untuk Python 3.9 - 3.12:**

```bash
pip3 uninstall torch torchvision torchaudio -y

# Install PyTorch Nightly (Support ROCm 6.1 / 6.2)
# --pre artinya pre-release (Nightly/Beta) yang support hardware baru
pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/rocm6.1
```

_Catatan: URL `rocm6.1` dipilih karena stabil untuk RDNA3. Jika ingin mencoba versi 6.2, ganti `rocm6.1` dengan `rocm6.2`._

## 4. Verifikasi Instalasi

Buat file `cek_gpu.py`:

```python
import torch
print(f"PyTorch Version: {torch.__version__}")
print(f"ROCm Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device Name: {torch.cuda.get_device_name(0)}")
else:
    print("GPU tidak terdeteksi!")
```

Jalankan:

```bash
python3 cek_gpu.py
```

---

**JANGAN LAKUKAN INI DI DISTROBOX:**

- ❌ `amdgpu-install` (Ini akan merusak container)
- ❌ `dkms install` (Ini butuh akses kernel host)
- ❌ `apt install` (Ini perintah Ubuntu, Fedora pakai `dnf`)
