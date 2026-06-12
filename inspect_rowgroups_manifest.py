from pathlib import Path
import csv
import pyarrow.parquet as pq

path = Path("fcn_debug/20260601060000-0h-oper-fc.parquet")
pf = pq.ParquetFile(path)
meta = pf.metadata

schema_names = [meta.schema.column(i).name for i in range(meta.num_columns)]

def stats_for(row_group, col_name):
    idx = schema_names.index(col_name)
    col = meta.row_group(row_group).column(idx)
    stats = col.statistics
    if stats is None:
        return None, None
    return stats.min, stats.max

rows = []

for rg in range(meta.num_row_groups):
    var_min, var_max = stats_for(rg, "variable")
    lev_min, lev_max = stats_for(rg, "isobaricInhPa")
    lat_min, lat_max = stats_for(rg, "latitude")
    lon_min, lon_max = stats_for(rg, "longitude")
    n = meta.row_group(rg).num_rows

    rows.append({
        "row_group": rg,
        "n_rows": n,
        "variable_min": var_min,
        "variable_max": var_max,
        "level_min": lev_min,
        "level_max": lev_max,
        "lat_min": lat_min,
        "lat_max": lat_max,
        "lon_min": lon_min,
        "lon_max": lon_max,
    })

out = Path("fcn_debug/rowgroups_manifest.csv")
with out.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

print("written:", out)
print("row_groups:", len(rows))

print("\nFirst 80 row groups:")
for r in rows[:80]:
    print(
        f"rg={r['row_group']:03d}",
        f"n={r['n_rows']}",
        f"var={r['variable_min']}..{r['variable_max']}",
        f"level={r['level_min']}..{r['level_max']}",
        f"lat={r['lat_min']}..{r['lat_max']}",
        f"lon={r['lon_min']}..{r['lon_max']}",
    )
