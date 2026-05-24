import pandas as pd
import numpy as np
import ast
import os
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, Dense, Reshape, Concatenate, LeakyReLU, Dropout, Embedding, Flatten, Conv1D, Conv1DTranspose
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

def build_generator(latent_dim, n_classes, spectrum_len):
    """Membangun model Generator untuk CGAN."""
    # Input untuk label kelas (kondisi)
    label_input = Input(shape=(1,), name="generator_label_input")
    # Embedding untuk label, menghasilkan vektor padat
    li = Embedding(n_classes, 50, name="generator_embedding")(label_input)
    # Perluas dimensi untuk digabungkan dengan noise
    li = Dense(latent_dim, name="generator_label_dense")(li)
    li = Reshape((latent_dim, 1), name="generator_label_reshape")(li)

    # Input untuk noise laten
    latent_input = Input(shape=(latent_dim,), name="generator_latent_input")
    # Perluas dimensi noise untuk lapisan konvolusi
    gen = Dense(latent_dim * 1, name="generator_noise_dense")(latent_input)
    gen = Reshape((latent_dim, 1), name="generator_noise_reshape")(gen)

    # Gabungkan label dan noise
    merge = Concatenate(axis=2)([gen, li])

    # PERBAIKAN: Menggunakan Conv1DTranspose (Deconvolution) untuk membangun spektrum
    # Ini lebih baik dalam menangkap struktur lokal (puncak) daripada Dense layers
    x = Conv1DTranspose(128, 5, strides=2, padding='same', activation='relu')(merge) # Upsample
    x = Conv1DTranspose(64, 5, strides=2, padding='same', activation='relu')(x) # Upsample
    x = Conv1DTranspose(32, 7, strides=2, padding='same', activation='relu')(x) # Upsample
    x = Conv1DTranspose(16, 7, strides=2, padding='same', activation='relu')(x) # Upsample
    x = Flatten()(x)
    out_layer = Dense(spectrum_len, activation='tanh', name="generator_output")(x)
    
    generator = Model([latent_input, label_input], out_layer)
    return generator

def build_discriminator(n_classes, spectrum_len):
    """Membangun model Discriminator untuk CGAN."""
    # Input untuk label kelas (kondisi)
    label_input = Input(shape=(1,), dtype='int32')
    # --- PERBAIKAN: Meratakan output dari Embedding layer ---
    # Ini mengubah output dari (None, 1, 50) menjadi (None, 50)
    li = Embedding(n_classes, 50)(label_input)
    li = Flatten()(li)
    li = Dense(spectrum_len)(li)

    # Input untuk spektrum
    spectrum_input = Input(shape=(spectrum_len,))

    # Gabungkan label dan spektrum
    merge = Concatenate()([spectrum_input, li])

    # Lapisan-lapisan untuk klasifikasi (real/fake)
    fe = Dense(512)(merge)
    fe = LeakyReLU(alpha=0.2)(fe)
    fe = Dropout(0.4)(fe)
    fe = Dense(256)(fe)
    fe = LeakyReLU(alpha=0.2)(fe)
    fe = Dropout(0.4)(fe)
    
    out_layer = Dense(1, activation='sigmoid')(fe)
    
    discriminator = Model([spectrum_input, label_input], out_layer)
    discriminator.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0002, beta_1=0.5), metrics=['accuracy'])
    return discriminator

def build_cgan(generator, discriminator):
    """Membangun model CGAN gabungan untuk melatih generator."""
    gen_noise, gen_label = generator.input
    gen_output = generator.output
    gan_output = discriminator([gen_output, gen_label])
    
    cgan = Model([gen_noise, gen_label], gan_output)
    cgan.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0002, beta_1=0.5))
    # PERBAIKAN: Atur discriminator.trainable menjadi False SETELAH cgan dikompilasi
    discriminator.trainable = False
    return cgan

def generate_synthetic_samples(generator, latent_dim, n_samples, class_label):
    """Menghasilkan sampel sintetis untuk kelas tertentu."""
    x_input = np.random.randn(latent_dim * n_samples).reshape(n_samples, latent_dim)
    labels = np.full(n_samples, class_label)
    X = generator.predict([x_input, labels])
    return X

def main():
    # --- 1. Konfigurasi dan Pemuatan Data ---
    print("📖 1. Memuat dan memproses data...")
    dataset_path = 'data/for_train/feature_extraction_dataset.csv'
    output_path = 'data/for_train/balanced_feature_extraction_dataset.csv'
    df = pd.read_csv(dataset_path)
    
    # Konversi dan padding spektrum
    # PERBAIKAN: Proses kedua kolom, absorbansi (y) dan bilangan gelombang (x)
    try:
        df['spectrum_values'] = df['spectrum_values'].apply(ast.literal_eval)
        df['wavenumber_values'] = df['wavenumber_values'].apply(ast.literal_eval)
    except KeyError as e:
        print(f"❌ Error: Kolom yang diperlukan ({e}) tidak ditemukan di '{dataset_path}'.")
        print("   Pastikan Anda telah menjalankan kembali skrip 'generate_feature_extraction_dataset.py' setelah pembaruan terakhir.")
        print("   Anda mungkin perlu menghapus file 'feature_extraction_dataset.csv' yang lama terlebih dahulu.")
        return
    except Exception as e:
        print(f"❌ Gagal memproses kolom spektrum. Error: {e}")
        return

    max_len = df['spectrum_values'].apply(len).max()
    X_padded = pad_sequences(df['spectrum_values'], maxlen=max_len, dtype='float32', padding='post', truncating='post')
    # Kita hanya melatih GAN pada nilai absorbansi (y), jadi nilai x tidak perlu diproses lebih lanjut di sini.

    # Normalisasi data spektrum ke rentang [-1, 1] untuk aktivasi tanh
    scaler = MinMaxScaler(feature_range=(-1, 1))
    X_scaled = scaler.fit_transform(X_padded)

    # Encoding label
    le = LabelEncoder()
    y_encoded = le.fit_transform(df['gugus_fungsi'])
    n_classes = len(le.classes_)

    print(f"Data diproses: {len(df)} sampel, {n_classes} kelas, panjang spektrum {max_len}.")

    # --- 2. Membangun Model CGAN ---
    print("\n🏗️ 2. Membangun model CGAN...")
    latent_dim = 100
    discriminator = build_discriminator(n_classes, max_len)
    generator = build_generator(latent_dim, n_classes, max_len)
    cgan = build_cgan(generator, discriminator)

    # --- 3. Melatih CGAN ---
    print("\n🚀 3. Melatih CGAN (ini mungkin memakan waktu)...")
    n_epochs = 15000  # PERBAIKAN: Tingkatkan epoch untuk pelatihan GAN yang lebih baik
    n_batch = 64
    half_batch = n_batch // 2
    
    for i in range(n_epochs):
        # --- Latih Discriminator ---
        # Ambil sampel asli secara acak
        ix_real = np.random.randint(0, X_scaled.shape[0], half_batch)
        X_real, labels_real = X_scaled[ix_real], y_encoded[ix_real]
        y_real = np.ones((half_batch, 1))
        d_loss1, _ = discriminator.train_on_batch([X_real, labels_real], y_real)

        # Hasilkan sampel palsu
        noise = np.random.randn(latent_dim * half_batch).reshape(half_batch, latent_dim)
        labels_fake_gen = np.random.randint(0, n_classes, half_batch)
        X_fake = generator.predict([noise, labels_fake_gen], verbose=0)
        y_fake = np.zeros((half_batch, 1))
        d_loss2, _ = discriminator.train_on_batch([X_fake, labels_fake_gen], y_fake)

        # --- Latih Generator ---
        noise_gan = np.random.randn(latent_dim * n_batch).reshape(n_batch, latent_dim)
        labels_gan = np.random.randint(0, n_classes, n_batch)
        y_gan = np.ones((n_batch, 1))
        g_loss = cgan.train_on_batch([noise_gan, labels_gan], y_gan)

        # Cetak progres dan simpan sampel gambar setiap 1000 epoch
        if (i + 1) % 1000 == 0:
            print(f"Epoch {i+1}/{n_epochs}, D Loss Real={d_loss1:.3f}, D Loss Fake={d_loss2:.3f}, G Loss={g_loss:.3f}")

    print("✅ Pelatihan CGAN selesai.")

    # --- 4. Menghasilkan Data Sintetis ---
    print("\n🧬 4. Menghasilkan data sintetis untuk menyeimbangkan dataset...")
    target_count = 50
    class_counts = df['gugus_fungsi'].value_counts()
    
    synthetic_data = []

    for class_name, count in class_counts.items():
        if count < target_count:
            n_to_generate = target_count - count
            class_label_encoded = le.transform([class_name])[0]
            
            print(f"  - Menghasilkan {n_to_generate} sampel untuk kelas '{class_name}'...")
            
            # Hasilkan spektrum sintetis
            generated_spectra_scaled = generate_synthetic_samples(generator, latent_dim, n_to_generate, class_label_encoded)
            
            # Denormalisasi spektrum kembali ke skala asli
            generated_spectra = scaler.inverse_transform(generated_spectra_scaled)
            
            for j in range(n_to_generate):
                synthetic_data.append({
                    "file_id": f"{class_name}_synthetic_{j+1}",
                    "gugus_fungsi": class_name,
                    "wavenumber_min": 0, # Placeholder
                    "wavenumber_max": 0, # Placeholder
                    # PERBAIKAN: Konversi setiap elemen ke float Python standar
                    "spectrum_values": [float(val) for val in generated_spectra[j]],
                    "wavenumber_values": [0.0] * max_len # PERBAIKAN: Tambahkan placeholder untuk wavenumber
                })

    if not synthetic_data:
        print("Dataset sudah seimbang, tidak ada data sintetis yang dihasilkan.")
    else:
        print(f"✅ {len(synthetic_data)} total sampel sintetis dihasilkan.")

    # --- 5. Menggabungkan dan Menyimpan Dataset Baru ---
    print("\n💾 5. Menggabungkan data asli dan sintetis...")
    
    # Buat DataFrame dari data sintetis
    df_synthetic = pd.DataFrame(synthetic_data)
    
    # Gabungkan dengan DataFrame asli
    # PERBAIKAN: Pastikan kolom 'wavenumber_values' dari data asli dipertahankan
    # dengan mengisi nilai yang hilang (pada data sintetis) dengan placeholder yang sesuai.
    df_balanced = pd.concat([df, df_synthetic], ignore_index=True).fillna(
        value={"wavenumber_values": str([0.0] * max_len)})
    
    # Simpan ke file CSV baru
    df_balanced.to_csv(output_path, index=False)
    
    print(f"🎉 Dataset seimbang berhasil disimpan di: {output_path}")
    print("\n--- Statistik Dataset Baru ---")
    print(df_balanced['gugus_fungsi'].value_counts())


if __name__ == '__main__':
    # Membuat direktori jika belum ada
    os.makedirs('data/for_train', exist_ok=True)
    
    main()