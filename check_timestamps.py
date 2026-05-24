import os
import datetime

files = [
    'models/ir_model.h5',
    'models/ir_scaler.pkl',
    'data/for_train/universal_training_dataset_IR.csv'
]

for f in files:
    if os.path.exists(f):
        ts = os.path.getmtime(f)
        dt = datetime.datetime.fromtimestamp(ts)
        print(f"{f}: {dt}")
    else:
        print(f"{f}: NOT FOUND")
