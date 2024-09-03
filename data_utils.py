import os

import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd

def read_parquet_file(file_path):

    try:
        return pq.read_table(file_path).to_pandas()

    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return pd.DataFrame()

def write_parquet_file(df, file_path):
    try:
        table = pa.Table.from_pandas(df)
        pq.write_table(table, file_path)
    
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None
