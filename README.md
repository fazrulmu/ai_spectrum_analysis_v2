Baik, kita akan bedah **Konsep Arsitektur & Alur Kerja (Workflow)** secara murni. Lupakan koding sejenak. Kita akan melihat ini sebagai sebuah sistem "Otak Hibrida".

Secara konseptual, sistem ini bekerja dalam 3 Fase Utama: **Standarisasi Persepsi (Input)**, **Ekstraksi Fitur (CNN)**, dan **Interpretasi Semantik (LLM)**.

Berikut adalah diagram alur logikanya:

---

### 1. Konsep Data Flow: "The Three-Stream Input"

Tantangan utama Anda adalah menyatukan data "kualitatif" (titik spesifik 3890 $cm^{-1}$) dengan data "kuantitatif" (spektrum penuh). Konsep solusinya adalah **Wide & Deep Architecture**.

Bayangkan model Anda memiliki **3 Mata** yang melihat objek yang sama (molekul) dari sudut pandang berbeda:

- **Mata 1 (IR Stream - High Detail):**

  - **Fokus:** Detail mikro. Vibrasi ikatan kimia.
  - **Cara Kerja:** Menerima input array panjang (3600 titik). Menggunakan _Convolution_ filter kecil (kernel size 3-5) untuk mendeteksi "jarum" atau peak tajam.
  - **Output:** "Saya melihat pola vibrasi C-H yang kuat."

- **Mata 2 (UV Stream - Low Detail):**

  - **Fokus:** Gambaran global. Transisi elektron (konjugasi/aromatik).
  - **Cara Kerja:** Menerima input array pendek (600 titik). Menggunakan _Convolution_ filter lebar (kernel size 11-15) untuk mendeteksi "bukit" atau kurva landai.
  - **Output:** "Saya melihat adanya sistem aromatik terkonjugasi."

- **Mata 3 (Explicit Feature - The Chemist's Hint):**
  - **Fokus:** Area kritis yang Anda tentukan (3890 & 2890 $cm^{-1}$).
  - **Cara Kerja:** _Direct Input_. Tidak perlu konvolusi. Nilai intensitas di titik ini langsung dimasukkan ke layer penggabungan (_Fusion Layer_). Ini seperti Anda berbisik ke model: _"Eh, perhatikan titik ini, ini penting!"_
  - **Output:** "Di titik 3890 nilainya 0.8 (tinggi), artinya ada gugus bebas."

---

### 2. Konsep Arsitektur: "Late Fusion & Embedding"

Bagaimana ketiga mata ini bersatu? Kita menggunakan konsep **Late Fusion** (Penggabungan Akhir).

1.  **Parallel Processing:** Ketiga input diproses secara terpisah terlebih dahulu. IR diproses oleh CNN-A, UV oleh CNN-B, dan Fitur Eksplisit oleh Neural Network biasa (MLP).
2.  **Bottleneck (Fusion Layer):** Hasil olahan ketiganya "digencet" menjadi satu vektor padat (misalnya vektor berisi 128 angka).
    - Vektor ini disebut **Latent Embedding**. Ini adalah "sari pati" atau "DNA digital" dari molekul tersebut berdasarkan spektrumnya.
3.  **Classification Head:** Dari vektor embedding, model menebak probabilitas gugus fungsi (misal: OH: 90%, CO: 10%).

---

### 3. Konsep Integrasi LLM: "From Vector to Voice"

Ini adalah bagian di mana **LLM Mini 1B** berperan. Karena Anda menggunakan CPU (AMD APU), kita harus efisien. Kita tidak akan men-training ulang LLM (terlalu berat), tapi kita menggunakan konsep **Context Injection** atau **RAG (Retrieval-Augmented Generation) style**.

Struktur kerjanya:

1.  **Analisis Numerik (Oleh CNN):**

    - CNN selesai bekerja dan memberikan output: _"Probabilitas Fenol: 0.98", "Probabilitas Alkohol: 0.95"_.
    - Sistem Feature Extraction memberikan data: _"Peak di 3890: Intensitas 0.85"_.

2.  **Penyusunan Prompt (The Bridge):**
    Kita mengubah angka-angka "dingin" di atas menjadi narasi terstruktur (Prompt) untuk LLM. LLM butuh konteks bahasa manusia.

    - _Input ke LLM (Internal Prompt):_
      > "Kamu adalah asisten lab. Saya punya data molekul.
      > Detektor AI mendeteksi probabilitas gugus Fenol 98%.
      > Data mentah menunjukkan adanya serapan kuat (0.85) pada area 3890 cm-1.
      > Apa artinya ini secara kimia?"

3.  **Inference (Oleh LLM 1B):**
    - LLM menggunakan _pre-trained knowledge_ (pengetahuan kimia yang sudah dia pelajari saat dibuat) untuk menghubungkan titik-titik tersebut.
    - _Output LLM:_ "Tingginya probabilitas fenol didukung oleh serapan kuat pada 3890 cm-1, yang mengindikasikan adanya gugus O-H bebas (tidak berikatan hidrogen intra-molekuler). Ini khas untuk struktur fenol dalam pelarut non-polar..."

---

### 4. Ringkasan Alur Kerja pada Hardware Anda (AMD APU)

Agar performa maksimal di CPU:

1.  **Serial, Bukan Paralel:** Jangan jalankan training CNN dan Inferensi LLM bersamaan.
2.  **Langkah 1 (Ringan):** Script Python memproses array spektrum -> Masuk ke CNN (bobot ringan <100MB) -> Keluar Angka Prediksi.
3.  **Langkah 2 (Berat):** Angka Prediksi disusun jadi teks -> Load LLM ke RAM (sekitar 1-2 GB untuk model 1B quantized) -> LLM membaca teks dan menjawab.
4.  **Selesai:** RAM dibersihkan.

Apakah konsep **"Tiga Mata"** (IR, UV, Eksplisit) yang bermuara ke **"Satu Otak Bahasa"** (LLM) ini sesuai dengan bayangan Anda?
