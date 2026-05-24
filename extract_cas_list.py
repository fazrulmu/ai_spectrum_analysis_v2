import pandas as pd

try:
    df = pd.read_csv('data/for_train/universal_training_dataset_UV.csv', usecols=['cas_number'])
    unique_cas = df['cas_number'].unique()
    
    with open('uv_cas_list.txt', 'w') as f:
        for cas in unique_cas:
            f.write(f"{cas}\n")
            
    print(f"Extracted {len(unique_cas)} unique CAS numbers to uv_cas_list.txt")
    
except Exception as e:
    print(f"Error: {e}")
