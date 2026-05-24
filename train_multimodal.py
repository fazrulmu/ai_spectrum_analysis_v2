import pandas as pd
import numpy as np
import tensorflow as tf
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

def build_multimodal_model(ir_input_shape, uv_input_shape):
    """
    Builds a multimodal CNN model for IR and UV spectra.
    """
    # IR Branch
    ir_input = tf.keras.layers.Input(shape=ir_input_shape, name="ir_input")
    x1 = tf.keras.layers.Conv1D(32, 5, activation='relu', padding='same')(ir_input)
    x1 = tf.keras.layers.MaxPooling1D(2)(x1)
    x1 = tf.keras.layers.Conv1D(64, 5, activation='relu', padding='same')(x1)
    x1 = tf.keras.layers.MaxPooling1D(2)(x1)
    x1 = tf.keras.layers.Flatten()(x1)
    
    # UV Branch
    uv_input = tf.keras.layers.Input(shape=uv_input_shape, name="uv_input")
    x2 = tf.keras.layers.Conv1D(32, 5, activation='relu', padding='same')(uv_input)
    x2 = tf.keras.layers.MaxPooling1D(2)(x2)
    x2 = tf.keras.layers.Conv1D(64, 5, activation='relu', padding='same')(x2)
    x2 = tf.keras.layers.MaxPooling1D(2)(x2)
    x2 = tf.keras.layers.Flatten()(x2)
    
    # Fusion
    concatenated = tf.keras.layers.Concatenate()([x1, x2])
    x = tf.keras.layers.Dense(128, activation='relu')(concatenated)
    x = tf.keras.layers.Dropout(0.4)(x)
    output = tf.keras.layers.Dense(1, activation='sigmoid', name="output")(x)
    
    model = tf.keras.Model(inputs=[ir_input, uv_input], outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_multimodal(ir_path, uv_path, target_label, output_dir=Path("reports/multimodal_analysis"), model_dir=Path("models")):
    print(f"🚀 Starting multimodal training for target: '{target_label}'")
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Load Data
    try:
        ir_df = pd.read_csv(ir_path)
        uv_df = pd.read_csv(uv_path)
        print(f"Loaded IR data: {ir_df.shape}")
        print(f"Loaded UV data: {uv_df.shape}")
        # Debug: Print first few columns
        # print("IR Columns:", list(ir_df.columns)[:10])
    except Exception as e:
        print(f"❌ Error loading datasets: {e}")
        return

    # Merge on molecule_id (assuming it exists)
    if 'molecule_id' not in ir_df.columns or 'molecule_id' not in uv_df.columns:
        print("❌ Error: 'molecule_id' column missing in one of the datasets.")
        return

    # Prefix columns to avoid collision, except molecule_id and target_label
    # Actually, we assume features are bin_...
    # Let's separate features and labels
    
    # Check target label
    if target_label not in ir_df.columns: # Assuming labels are in IR dataset (or both)
        print(f"❌ Error: Target label '{target_label}' not found in IR dataset.")
        return
        
    # Merge
    # We need to be careful about column names. 
    # IR features: bin_...
    # UV features: bin_... (might overlap in name if ranges overlap, though unlikely for IR/UV)
    # But to be safe, let's rename features before merge
    
    ir_features = ir_df.filter(like='bin_')
    uv_features = uv_df.filter(like='bin_')
    
    # Rename columns to ensure uniqueness
    ir_df_renamed = ir_df.rename(columns={col: f"ir_{col}" for col in ir_features.columns})
    uv_df_renamed = uv_df.rename(columns={col: f"uv_{col}" for col in uv_features.columns})
    
    # Keep only molecule_id and target_label in one of them for merging
    # We assume target_label is in IR df
    cols_to_keep_ir = ['molecule_id', target_label] + [f"ir_{col}" for col in ir_features.columns]
    cols_to_keep_uv = ['molecule_id'] + [f"uv_{col}" for col in uv_features.columns]
    
    merged_df = pd.merge(ir_df_renamed[cols_to_keep_ir], uv_df_renamed[cols_to_keep_uv], on='molecule_id', how='inner')
    
    print(f"✅ Merged data: {len(merged_df)} samples common to both datasets.")
    
    if merged_df.empty:
        print("❌ Error: No common samples found.")
        return

    X_ir = merged_df.filter(like='ir_bin_').values
    X_uv = merged_df.filter(like='uv_bin_').values
    y = merged_df[target_label].values
    
    # Split
    X_ir_train, X_ir_test, X_uv_train, X_uv_test, y_train, y_test = train_test_split(
        X_ir, X_uv, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Reshape for CNN
    X_ir_train = np.expand_dims(X_ir_train, axis=-1)
    X_ir_test = np.expand_dims(X_ir_test, axis=-1)
    X_uv_train = np.expand_dims(X_uv_train, axis=-1)
    X_uv_test = np.expand_dims(X_uv_test, axis=-1)
    
    # Build Model
    model = build_multimodal_model(
        ir_input_shape=(X_ir_train.shape[1], 1),
        uv_input_shape=(X_uv_train.shape[1], 1)
    )
    model.summary()
    
    # Class weights
    neg, pos = np.bincount(y_train.astype(int))
    total = neg + pos
    class_weight = {0: (1 / neg) * (total / 2.0), 1: (1 / pos) * (total / 2.0)}
    
    # Train
    history = model.fit(
        {"ir_input": X_ir_train, "uv_input": X_uv_train},
        y_train,
        epochs=30,
        batch_size=32,
        validation_split=0.2,
        class_weight=class_weight,
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)],
        verbose=1
    )
    
    # Evaluate
    y_pred_proba = model.predict({"ir_input": X_ir_test, "uv_input": X_uv_test})
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    print(classification_report(y_test, y_pred, zero_division=0))
    
    # Save Model
    model_path = model_dir / f"multimodal_model_{target_label}.keras"
    model.save(model_path)
    print(f"✅ Model saved to {model_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train multimodal model")
    parser.add_argument("--target", type=str, help="Target functional group to train for")
    args = parser.parse_args()

    IR_FILE = Path("data/for_train/universal_training_dataset_IR.csv")
    UV_FILE = Path("data/for_train/universal_training_dataset_UV.csv")
    
    # Try to import target groups, otherwise use default
    try:
        from target_groups import TARGET_FUNCTIONAL_GROUPS
        TARGET_GROUPS = TARGET_FUNCTIONAL_GROUPS
    except ImportError:
        print("⚠️ Could not import target_groups. Using default list.")
        TARGET_GROUPS = ['alcohol', 'ketone', 'aldehyde']
        
    if args.target:
        if args.target not in TARGET_GROUPS:
             print(f"⚠️ Warning: '{args.target}' not in defined target groups. Proceeding anyway.")
        train_multimodal(IR_FILE, UV_FILE, args.target)
    else:
        if not TARGET_GROUPS:
            print("No target groups found.")
        else:
            print(f"🎯 Target groups: {TARGET_GROUPS}")
            for group in TARGET_GROUPS:
                train_multimodal(IR_FILE, UV_FILE, group)
