import pandas as pd
df = pd.read_parquet('india_stubble_clean.parquet')
with open('cols.txt', 'w') as f:
    f.write(','.join(df.columns.tolist()))
