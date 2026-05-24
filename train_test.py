import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import keras_tuner as kt
import joblib

# -------------------- IR MODEL (CNN + Hyperparameter Tuning) --------------------

def build_ir_model(hp):
    model = models.Sequential()
    model.add(layers.Conv1D(
        filters=hp.Int('filters1', 32, 128, step=32),
        kernel_size=hp.Int('kernel1', 3, 7, step=2),
        activation='relu', padding='same',
        input_shape=(X_ir_train_cnn.shape[1], 1)))

    model.add(layers.MaxPooling1D(2))

    model.add(layers.Conv1D(
        filters=hp.Int('filters2', 64, 256, step=64),
        kernel_size=hp.Int('kernel2', 3, 7, step=2),
        activation='relu', padding='same'))

    model.add(layers.GlobalAveragePooling1D())

    model.add(layers.Dense(
        hp.Int('dense_units', 128, 512, step=128), activation='relu'))

    model.add(layers.Dense(y_ir_train.shape[1], activation='sigmoid'))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            hp.Float('lr', 1e-4, 1e-2, sampling='log')),
        loss='binary_crossentropy', metrics=['binary_accuracy'])

    return model

# Load IR data
ir_path = 'data/for_train/universal_training_dataset_IR.csv'
ir_df = pd.read_csv(ir_path)
ir_feature_cols = [col for col in ir_df.columns if col.startswith('num_peaks') or col.startswith('fwhm') or col.replace('.', '', 1).isdigit()]
X_ir = ir_df[ir_feature_cols].values
y_ir = ir_df[['cas_number', 'spectrum_type', 'state', 'matrix_sample', 'abs_matrix']]

encoder_ir = OneHotEncoder(sparse_output=False)
y_ir_encoded = encoder_ir.fit_transform(y_ir)
X_ir_train, X_ir_test, y_ir_train, y_ir_test = train_test_split(X_ir, y_ir_encoded, test_size=0.2, random_state=42)

scaler_ir = StandardScaler()
X_ir_train_scaled = scaler_ir.fit_transform(X_ir_train)
X_ir_test_scaled = scaler_ir.transform(X_ir_test)

X_ir_train_cnn = X_ir_train_scaled.reshape(-1, X_ir_train_scaled.shape[1], 1)
X_ir_test_cnn = X_ir_test_scaled.reshape(-1, X_ir_test_scaled.shape[1], 1)

# Tuner
ir_tuner = kt.RandomSearch(
    build_ir_model,
    objective='val_binary_accuracy',
    max_trials=10,
    executions_per_trial=1,
    overwrite=True,
    directory='tuner_ir',
    project_name='ir_cnn_opt')

ir_tuner.search(X_ir_train_cnn, y_ir_train, epochs=20, validation_split=0.1)
best_ir_model = ir_tuner.get_best_models(1)[0]
best_ir_model.save('ir_model_cnn_optimized.h5')
joblib.dump(scaler_ir, 'ir_scaler.pkl')
joblib.dump(encoder_ir, 'ir_encoder.pkl')

# -------------------- UV MODEL (CNN + Hyperparameter Tuning) --------------------

def build_uv_model(hp):
    model = models.Sequential()
    model.add(layers.Conv1D(
        filters=hp.Int('filters1', 32, 128, step=32),
        kernel_size=hp.Int('kernel1', 3, 7, step=2),
        activation='relu', padding='same',
        input_shape=(X_uv_train_cnn.shape[1], 1)))

    model.add(layers.MaxPooling1D(2))

    model.add(layers.Conv1D(
        filters=hp.Int('filters2', 64, 256, step=64),
        kernel_size=hp.Int('kernel2', 3, 7, step=2),
        activation='relu', padding='same'))

    model.add(layers.GlobalAveragePooling1D())

    model.add(layers.Dense(
        hp.Int('dense_units', 128, 512, step=128), activation='relu'))

    model.add(layers.Dense(y_uv_train.shape[1], activation='sigmoid'))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            hp.Float('lr', 1e-4, 1e-2, sampling='log')),
        loss='binary_crossentropy', metrics=['binary_accuracy'])

    return model

uv_path = 'data/for_train/universal_training_dataset_UV.csv'
uv_df = pd.read_csv(uv_path)
uv_feature_cols = [col for col in uv_df.columns if col in ['num_peaks', 'fwhm', 'lambda_max', 'log_epsilon'] or col.replace('.', '', 1).isdigit()]
X_uv = uv_df[uv_feature_cols].values
y_uv = uv_df[['cas_number', 'spectrum_type', 'state', 'matrix_sample', 'abs_matrix']]

encoder_uv = OneHotEncoder(sparse_output=False)
y_uv_encoded = encoder_uv.fit_transform(y_uv)
X_uv_train, X_uv_test, y_uv_train, y_uv_test = train_test_split(X_uv, y_uv_encoded, test_size=0.2, random_state=42)

scaler_uv = StandardScaler()
X_uv_train_scaled = scaler_uv.fit_transform(X_uv_train)
X_uv_test_scaled = scaler_uv.transform(X_uv_test)

X_uv_train_cnn = X_uv_train_scaled.reshape(-1, X_uv_train_scaled.shape[1], 1)
X_uv_test_cnn = X_uv_test_scaled.reshape(-1, X_uv_test_scaled.shape[1], 1)

uv_tuner = kt.RandomSearch(
    build_uv_model,
    objective='val_binary_accuracy',
    max_trials=10,
    executions_per_trial=1,
    overwrite=True,
    directory='tuner_uv',
    project_name='uv_cnn_opt')

uv_tuner.search(X_uv_train_cnn, y_uv_train, epochs=20, validation_split=0.1)
best_uv_model = uv_tuner.get_best_models(1)[0]
best_uv_model.save('uv_model_cnn_optimized.h5')
joblib.dump(scaler_uv, 'uv_scaler.pkl')
joblib.dump(encoder_uv, 'uv_encoder.pkl')
