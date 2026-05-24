import pandas as pd

try:
    df = pd.read_csv('data/for_train/universal_training_dataset_IR.csv', nrows=0)
    count = len(df.columns)
    with open('col_count.txt', 'w') as f:
        f.write(str(count))
except Exception as e:
    with open('col_count.txt', 'w') as f:
        f.write(str(e))
