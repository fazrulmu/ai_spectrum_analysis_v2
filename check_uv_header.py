import pandas as pd
try:
    df = pd.read_csv('data/for_train/universal_training_dataset_UV.csv', nrows=0)
    print(df.columns.tolist())
except Exception as e:
    print(e)
