import joblib
import os
import numpy as np

try:
    uv_scaler = joblib.load('models/uv_scaler.pkl')
    print(f"UV Scaler type: {type(uv_scaler)}")
    if hasattr(uv_scaler, 'n_features_in_'):
        print(f"UV Scaler n_features_in_: {uv_scaler.n_features_in_}")
    if hasattr(uv_scaler, 'feature_names_in_'):
        print(f"UV Scaler feature_names_in_: {uv_scaler.feature_names_in_[:10]} ...")
        print(f"Total feature names: {len(uv_scaler.feature_names_in_)}")
    else:
        print("UV Scaler has no feature_names_in_")

    ir_scaler = joblib.load('models/ir_scaler.pkl')
    print(f"IR Scaler n_features_in_: {ir_scaler.n_features_in_}")

except Exception as e:
    print(f"Error: {e}")
