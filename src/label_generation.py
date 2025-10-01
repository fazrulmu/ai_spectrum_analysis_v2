# src/label_generation.py

import pandas as pd
import os

# Database sederhana untuk memetakan Nomor CAS ke gugus fungsi
# Dalam proyek nyata, ini akan berasal dari database kimia atau anotasi manual
CAS_TO_FUNCTIONAL_GROUPS = {
    "67-64-1": ["ketone"],
    "67-63-0": ["alcohol"],
    "71-43-2": ["aromatic"],
    "108-88-3": ["aromatic", "alkane"],
    "78-93-3": ["ketone"],
    "64-17-5": ["alcohol"],
    "75-07-0": ["aldehyde"],
    "100-52-7": ["aldehyde", "aromatic"],
    "65-85-0": ["carboxylic_acid", "aromatic"],
    "64-19-7": ["carboxylic_acid"],
    "50-78-2": ["carboxylic_acid", "ester", "aromatic"], # Aspirin
    "62-53-3": ["amine", "aromatic"], # Aniline
}

def generate_labels(output_path="data/labels/auto_labels.csv"):
    """
    Membuat file CSV label berdasarkan database CAS_TO_FUNCTIONAL_GROUPS.
    """
    # Pastikan direktori ada
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    records = []
    all_groups = set()
    for cas, groups in CAS_TO_FUNCTIONAL_GROUPS.items():
        records.append({"cas_no": cas, "groups": ",".join(groups)})
        for group in groups:
            all_groups.add(group)
    
    df = pd.DataFrame(records)
    
    # One-hot encoding
    for group in sorted(list(all_groups)):
        df[group] = df['groups'].apply(lambda x: 1 if group in x.split(',') else 0)
        
    df = df.drop(columns=['groups'])
    
    df.to_csv(output_path, index=False)
    print(f"File label berhasil dibuat di: {output_path}")

if __name__ == '__main__':
    generate_labels()