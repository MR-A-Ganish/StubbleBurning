import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

csv_file = "india_stubble_clean.csv"
parquet_file = "india_stubble_clean.parquet"

chunk_size = 500000   # half million rows per chunk

writer = None

print("Starting conversion...")

for i, chunk in enumerate(pd.read_csv(
        csv_file,
        chunksize=chunk_size,
        usecols=["latitude","longitude","brightness","frp","acq_date"],
        dtype={
            "latitude":"float32",
            "longitude":"float32",
            "brightness":"float32",
            "frp":"float32"
        })):

    print(f"Processing chunk {i+1}...")

    chunk["acq_date"] = pd.to_datetime(chunk["acq_date"])

    table = pa.Table.from_pandas(chunk)

    if writer is None:
        writer = pq.ParquetWriter(parquet_file, table.schema)

    writer.write_table(table)

if writer:
    writer.close()

print("Conversion complete!")