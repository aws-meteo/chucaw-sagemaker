from pathlib import Path
import pyarrow.parquet as pq

path = Path("fcn_debug/20260601060000-0h-oper-fc.parquet")
pf = pq.ParquetFile(path)
meta = pf.metadata

print("num_rows:", meta.num_rows)
print("num_row_groups:", meta.num_row_groups)
print("num_columns:", meta.num_columns)
print("\nschema:")
print(pf.schema)

# Locate useful columns
schema_names = [meta.schema.column(i).name for i in range(meta.num_columns)]
print("\ncolumns:", schema_names)

for col_name in ["variable", "isobaricInhPa", "latitude", "longitude"]:
    if col_name not in schema_names:
        print(f"\nCOLUMN MISSING: {col_name}")
        continue

    idx = schema_names.index(col_name)
    print(f"\nRow-group stats for {col_name}:")
    seen = set()

    for rg in range(min(meta.num_row_groups, 30)):
        col = meta.row_group(rg).column(idx)
        stats = col.statistics
        if stats is None:
            print(rg, "NO_STATS")
            continue

        item = (stats.min, stats.max, col.num_values)
        seen.add(item)
        print(f"rg={rg:03d} min={stats.min} max={stats.max} n={col.num_values}")

    print(f"unique first-30 stats for {col_name}:", len(seen))
