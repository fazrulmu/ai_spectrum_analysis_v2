import pandas as pd
from sklearn.preprocessing import OneHotEncoder

# Load IR data
ir_path = 'data/for_train/universal_training_dataset_IR.csv'
try:
    ir_df = pd.read_csv(ir_path)
    y_ir = ir_df[['cas_number', 'spectrum_type', 'state', 'matrix_sample', 'abs_matrix']]
    
    print("y_ir shape:", y_ir.shape)
    print("\nUnique values per column:")
    for col in y_ir.columns:
        print(f"{col}: {y_ir[col].nunique()}")
        
    encoder_ir = OneHotEncoder(sparse_output=False)
    y_ir_encoded = encoder_ir.fit_transform(y_ir)
    
    print("\ny_ir_encoded shape:", y_ir_encoded.shape)
    print("Sum of first row:", y_ir_encoded[0].sum())
    print("Max of first row:", y_ir_encoded[0].max())
    
except Exception as e:
    print(f"Error: {e}")
