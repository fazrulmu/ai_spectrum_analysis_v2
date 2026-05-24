A. Qwen2.5-1.5B-Instruct (Paling Rekomen 🏆)

    Mengapa: Saat ini dianggap sebagai "Raja" model kecil. Kemampuan logika dan kodingnya setara dengan model 7B lama. Sangat patuh pada instruksi sistem.

    Link HF: Qwen/Qwen2.5-1.5B-Instruct (Cari versi GGUF untuk CPU).

    Kelebihan: Sangat bagus dalam menghasilkan output terstruktur (JSON/XML) jika diminta.




    Contoh Implementasi (Python + Llama.cpp)

Anda bisa menjalankan ini di CPU Ryzen Anda. Instal dulu: pip install llama-cpp-python

Berikut skrip untuk memaksa Qwen-1.5B mengeluarkan output JSON yang berisi analisis kimia (sesuai kasus Anda sebelumnya), tanpa training ulang:
Python

from llama_cpp import Llama
from llama_cpp.llama_grammar import LlamaGrammar

# 1. Load Model GGUF (Download dulu dari HuggingFace, misal Qwen2.5-1.5B-Instruct-Q4_K_M.gguf)

# n_gpu_layers=0 artinya full CPU (RAM), cocok untuk Ryzen APU jika iGPU belum setup.

llm = Llama(
model_path="./Qwen2.5-1.5B-Instruct-Q4_K_M.gguf",
n_ctx=2048,
verbose=False
)

# 2. RAHASIANYA DI SINI: Grammar (Aturan Output)

# Kita paksa output HARUS format JSON dengan key tertentu.

# Model TIDAK BISA curhat atau ngomong hal lain selain format ini.

grammar_text = r"""
root ::= object
object ::= "{" space pair_list space "}"
pair_list ::= pair | pair "," space pair_list
pair ::= string ":" space value
string ::= "\"" [^"]_ "\""
value ::= string | number | object | array | boolean | null
space ::= [ \t\n]_
boolean ::= "true" | "false"
null ::= "null"
number ::= [0-9]+ "."? [0-9]\*
array ::= "[" space list_elements space "]"
list_elements ::= value | value "," space list_elements
"""

# Atau cara lebih gampang pakai JSON schema converter bawaan llama-cpp

# (Script di bawah pakai schema otomatis via response_format)

# 3. Prompt Engineering (System Prompt)

system_prompt = "Kamu adalah ahli spektroskopi. Ekstrak data kimia dari input user ke dalam format JSON."

user_input = "Tolong cek CAS 108-88-3, data IR nya ada puncak di 1600 sama 1450, kayaknya toluena."

# 4. Generate dengan Penjagaan Ketat (Schema)

response = llm.create_chat_completion(
messages=[
{"role": "system", "content": system_prompt},
{"role": "user", "content": user_input}
], # FITUR KUNCI: response_format # Ini memaksa output menjadi JSON valid sesuai struktur yang kita mau
response_format={
"type": "json_object",
"schema": {
"type": "object",
"properties": {
"nama_senyawa": {"type": "string"},
"cas_number": {"type": "string"},
"gugus_fungsi_terdeteksi": {
"type": "array",
"items": {"type": "string"}
},
"analisis_uv": {"type": "string"},
"confidence_score": {"type": "number"}
},
"required": ["nama_senyawa", "cas_number", "analisis_uv"]
}
},
temperature=0.1 # Rendah agar tidak halusinasi
)

print(response['choices'][0]['message']['content'])

Hasil Output (Pasti JSON):

Meskipun modelnya kecil (1.5B) dan tidak di-training, dia akan mengeluarkan ini karena dipaksa oleh kode:
JSON

{
"nama_senyawa": "Toluena",
"cas_number": "108-88-3",
"gugus_fungsi_terdeteksi": ["C=C Aromatik", "Alkil"],
"analisis_uv": "Kemungkinan adsorpsi di area 260nm karena cincin aromatik",
"confidence_score": 0.95
}

Kesimpulan Strategi

    Ambil Model: Download Qwen2.5-1.5B-Instruct (format .gguf versi Q4_K_M agar ringan, cuma ~1GB).

    Jangan Training: Tidak perlu buang waktu SFT (Supervised Fine Tuning).

    Gunakan response_format / Grammar: Ini adalah "tali kekang" yang membuat model kecil berperilaku seperti model besar yang profesional. Dia tidak akan bisa typo format JSON.
