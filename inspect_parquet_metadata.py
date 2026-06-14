from pathlib import Path
import pyarrow.parquet as pq

path = Path("fcn_debug/20260601060000-0h-oper-fc.parquet")
pf = pq.ParquetFile(path)

print("file:", path)
print("num_row_groups:", pf.num_row_groups)
print("schema:")
print(pf.schema)

meta = pf.metadata
print("\nmetadata:")
print("num_rows:", meta.num_rows)
print("num_columns:", meta.num_columns)
print("serialized_size:", meta.serialized_size)

total_uncompressed = 0
total_compressed = 0

print("\ncolumns:")
for i in range(meta.num_columns):
    col_name = meta.schema.column(i).name
    comp = 0
    uncomp = 0
    for rg in range(meta.num_row_groups):
        cc = meta.row_group(rg).column(i)
        comp += cc.total_compressed_size
        uncomp += cc.total_uncompressed_size
    total_compressed += comp
    total_uncompressed += uncomp
    print(f"{i:02d} {col_name:20s} compressed={comp/1024/1024:8.1f} MiB uncompressed={uncomp/1024/1024:8.1f} MiB")

print("\ntotal_compressed_columns_mib:", total_compressed/1024/1024)
print("total_uncompressed_columns_mib:", total_uncompressed/1024/1024)
