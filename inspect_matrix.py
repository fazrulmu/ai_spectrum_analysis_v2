import json
from collections import Counter

file_path = 'data/for_train/universal_training_dataset_IR.jsonl'
matrix_counts = Counter()

try:
    with open(file_path, 'r') as f:
        for line in f:
            try:
                record = json.loads(line)
                # Check where matrix_sample is located.
                # Based on preprocess_universal.py, it's in 'metadata' -> 'matrix_sample'
                # But let's check top level too just in case
                meta = record.get('metadata', {})
                matrix = meta.get('matrix_sample', 'UNKNOWN')
                matrix_counts[matrix] += 1
            except:
                pass

    print("Matrix Sample Distribution:")
    for m, c in matrix_counts.most_common(20):
        print(f"{m}: {c}")

except FileNotFoundError:
    print(f"File not found: {file_path}")
