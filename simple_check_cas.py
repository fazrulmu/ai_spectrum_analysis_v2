import joblib
import os
import sys

try:
    encoder_path = 'models/encoders/ir_encoder.joblib'
    if not os.path.exists(encoder_path):
        # Try pkl
        encoder_path = 'models/encoders/ir_encoder.pkl'
        
    if not os.path.exists(encoder_path):
        print(f"Encoder not found in models/encoders/")
        # Try root models dir
        encoder_path = 'models/ir_encoder.pkl'
        
    if not os.path.exists(encoder_path):
        print("Could not find encoder file.")
        sys.exit(1)

    print(f"Loading encoder from {encoder_path}")
    encoder = joblib.load(encoder_path)
    
    classes = []
    if hasattr(encoder, 'categories_'):
        classes = encoder.categories_[0]
    elif hasattr(encoder, 'classes_'):
        classes = encoder.classes_
    
    target = '100-42-5'
    found = False
    for c in classes:
        if target in str(c):
            print(f"SUCCESS: Found {c}")
            found = True
            break
            
    if not found:
        print(f"FAILURE: CAS {target} NOT found in {len(classes)} classes.")
        print(f"First 5 classes: {classes[:5]}")

except Exception as e:
    print(f"Error: {e}")
