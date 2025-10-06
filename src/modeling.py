import os
import datetime
import pandas as pd
import joblib
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report,precision_score,recall_score,f1_score,precision_recall_fscore_support
from sklearn.model_selection import GroupShuffleSplit,GroupKFold
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import backend as K
from keras.saving import register_keras_serializable

from .model_architecture import build_cnn_model
from .data_processing import augment_spectrum
# =====================================================
# 🔹 Custom Metrics
# =====================================================

@register_keras_serializable(package="Custom")
def f1_m(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(tf.round(y_pred), tf.float32)

    tp = tf.reduce_sum(y_true * y_pred, axis=0)
    fp = tf.reduce_sum((1 - y_true) * y_pred, axis=0)
    fn = tf.reduce_sum(y_true * (1 - y_pred), axis=0)

    precision = tp / (tp + fp + tf.keras.backend.epsilon())
    recall = tp / (tp + fn + tf.keras.backend.epsilon())

    f1 = 2 * precision * recall / (precision + recall + tf.keras.backend.epsilon())
    return tf.reduce_mean(f1)


# =====================================================
# 🔹 Class Weights
# =====================================================
def get_multilabel_class_weights(y):
    n_samples, n_classes = y.shape
    class_weights = {}
    for i in range(n_classes):
        freq = np.sum(y[:, i])
        class_weights[i] = float(n_samples) / (n_classes * float(freq)) if freq > 0 else 1.0
    return class_weights


# =====================================================
# 🔹 Loss Functions
# =====================================================



@tf.keras.utils.register_keras_serializable(package="Custom", name="weighted_binary_crossentropy")
def weighted_binary_crossentropy(class_weights):
    weights_vector = tf.constant(
        [class_weights[i] for i in sorted(class_weights.keys())],
        dtype=tf.float32
    )

    def loss(y_true, y_pred):
        y_true_f = tf.cast(y_true, tf.float32)
        y_pred_f = tf.cast(y_pred, tf.float32)
        bce = tf.keras.backend.binary_crossentropy(y_true_f, y_pred_f)
        weighted_bce = bce * weights_vector
        return tf.reduce_mean(weighted_bce)

    return loss





@tf.keras.utils.register_keras_serializable(package="Custom", name="focal_loss")
def focal_loss(gamma=2.0, alpha=0.25):
    def loss(y_true, y_pred):
        y_true_f = tf.cast(y_true, tf.float32)
        y_pred_f = tf.cast(y_pred, tf.float32)
        eps = tf.keras.backend.epsilon()
        y_pred_f = tf.clip_by_value(y_pred_f, eps, 1.0 - eps)
        p_t = tf.where(tf.equal(y_true_f, 1.0), y_pred_f, 1.0 - y_pred_f)
        ce = - (y_true_f * tf.math.log(y_pred_f + eps) + (1.0 - y_true_f) * tf.math.log(1.0 - y_pred_f + eps))
        modulating_factor = tf.pow(1.0 - p_t, gamma)
        alpha_factor = tf.where(tf.equal(y_true_f, 1.0), alpha, 1.0 - alpha)
        fl = alpha_factor * modulating_factor * ce
        return tf.reduce_mean(fl)
    return loss



@tf.keras.utils.register_keras_serializable(package="Custom", name="hybrid_loss")
def hybrid_loss(class_weights, alpha=0.5, gamma=2.0):
    weights_vector = tf.constant(
        [class_weights[i] for i in sorted(class_weights.keys())],
        dtype=tf.float32
    )

    def loss(y_true, y_pred):
        y_true_f = tf.cast(y_true, tf.float32)
        y_pred_f = tf.cast(y_pred, tf.float32)
        bce = tf.keras.backend.binary_crossentropy(y_true_f, y_pred_f)
        weighted_bce = bce * weights_vector
        pt = tf.where(tf.equal(y_true_f, 1), y_pred_f, 1 - y_pred_f)
        focal = -tf.math.log(tf.clip_by_value(pt, 1e-7, 1.0)) * tf.pow(1 - pt, gamma)
        weighted_focal = focal * weights_vector
        return tf.reduce_mean(alpha * weighted_bce + (1.0 - alpha) * weighted_focal)

    return loss



# =====================================================
# 🔹 Training Function
# =====================================================
# src/modeling.py

# ... (kode lain di atas tetap sama) ...

# src/modeling.py

# ... (impor dan fungsi loss/metrik Anda berada di atas sini) ...
from sklearn.model_selection import GroupKFold
# Pastikan semua impor yang relevan ada di bagian atas file Anda

# =====================================================
# 🔹 Fungsi Pelatihan Model Dasar dengan Cross-Validation
# =====================================================
def train_base_model_cv(X_spec, X_meta, y, groups, config, spectrum_type):
    """
    Melatih model dasar (CNN) menggunakan GroupKFold Cross-Validation.
    Fungsi ini juga melakukan augmentasi data on-the-fly pada data training
    dan mengembalikan prediksi out-of-fold (OOF) untuk membangun dataset meta.
    """
    paths, model_config = config['paths'], config['modeling']
    train_config = model_config['training_params']
    aug_config = model_config['augmentation']
    n_splits = model_config.get('cv_splits', 5)

    # Inisialisasi array untuk menyimpan prediksi OOF
    oof_preds = np.zeros_like(y, dtype=float)
    
    # Inisialisasi GroupKFold (Perbaikan untuk NameError)
    gkf = GroupKFold(n_splits=n_splits)

    print(f"🔬 Memulai {n_splits}-Fold Cross-Validation untuk model {spectrum_type.upper()}...")

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X_spec, y, groups=groups)):
        print(f"\n===== FOLD {fold + 1}/{n_splits} =====")

        # 1. Membagi data untuk fold saat ini
        X_spec_train, X_spec_val = X_spec[train_idx], X_spec[val_idx]
        X_meta_train, X_meta_val = X_meta[train_idx], X_meta[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # 2. Augmentasi data on-the-fly (hanya pada data training)
        if aug_config['enabled']:
            print(f"Applying augmentation (factor: {aug_config['factor']})...")
            X_spec_train_aug, X_meta_train_aug, y_train_aug = [], [], []
            
            for i in range(len(X_spec_train)):
                # Tambahkan data asli
                X_spec_train_aug.append(X_spec_train[i])
                X_meta_train_aug.append(X_meta_train[i])
                y_train_aug.append(y_train[i])
                
                # Tambahkan data augmentasi
                augmented_spectra = augment_spectrum(X_spec_train[i].flatten(), factor=aug_config['factor'])
                for aug_spec in augmented_spectra:
                    X_spec_train_aug.append(aug_spec.reshape(-1, 1))
                    X_meta_train_aug.append(X_meta_train[i])
                    y_train_aug.append(y_train[i])

            X_spec_train = np.array(X_spec_train_aug)
            X_meta_train = np.array(X_meta_train_aug)
            y_train = np.array(y_train_aug)
            print(f"Ukuran training set setelah augmentasi: {len(X_spec_train)}")

        # 3. Membangun & Melatih Model
        # Model harus dibuat ulang di setiap fold untuk reset bobot
        model = build_cnn_model((X_spec.shape[1], 1), (X_meta.shape[1],), y.shape[1], config)
        
        class_weights = get_multilabel_class_weights(y_train)
        loss_type = config.get("loss_function", "hybrid")
        if loss_type == "hybrid": loss_fn = hybrid_loss(class_weights)
        elif loss_type == "focal": loss_fn = focal_loss()
        else: loss_fn = weighted_binary_crossentropy(class_weights)

        optimizer = tf.keras.optimizers.Adam(learning_rate=train_config['learning_rate'])
        
        # Kompilasi dengan run_eagerly=True (Perbaikan untuk NotImplementedError)
        model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy', f1_m], run_eagerly=True)

        early_stopping = EarlyStopping(
            monitor='val_loss', patience=train_config['early_stopping_patience'], mode='min', restore_best_weights=True
        )

        model.fit(
            [X_spec_train, X_meta_train], y_train,
            epochs=train_config['epochs'],
            batch_size=train_config['batch_size'],
            validation_data=([X_spec_val, X_meta_val], y_val),
            callbacks=[early_stopping],
            verbose=2
        )

        # 4. Simpan prediksi pada validation set (Out-of-Fold)
        oof_preds[val_idx] = model.predict([X_spec_val, X_meta_val])

    # 5. Latih Ulang Model Final pada Semua Data & Simpan
    print("\nTraining model final pada semua data untuk deployment...")
    final_model = build_cnn_model((X_spec.shape[1], 1), (X_meta.shape[1],), y.shape[1], config)
    
    # Dapatkan class_weights dari seluruh dataset 'y'
    final_class_weights = get_multilabel_class_weights(y)
    if loss_type == "hybrid": final_loss_fn = hybrid_loss(final_class_weights)
    elif loss_type == "focal": final_loss_fn = focal_loss()
    else: final_loss_fn = weighted_binary_crossentropy(final_class_weights)

    final_optimizer = tf.keras.optimizers.Adam(learning_rate=train_config['learning_rate'])
    final_model.compile(optimizer=final_optimizer, loss=final_loss_fn, metrics=['accuracy', f1_m], run_eagerly=True)
    
    # Melatih pada seluruh data tanpa augmentasi
    final_model.fit([X_spec, X_meta], y, epochs=train_config['epochs'], batch_size=train_config['batch_size'], verbose=0)
    
    model_save_path = os.path.join(paths['saved_models_dir'], f'base_model_{spectrum_type}_final.keras')
    final_model.save(model_save_path)
    print(f"✅ Model dasar final {spectrum_type.upper()} disimpan di {model_save_path}")

    return oof_preds