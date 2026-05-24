import pandas as pd
import glob

for file in glob.glob('extracted_*.csv'):
    print(f"--- {file} ---")
    try:
        df = pd.read_csv(file)
        print(df['state'].value_counts())
    except Exception as e:
        print(e)
