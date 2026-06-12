from pathlib import Path
from collections import Counter, defaultdict
import pyarrow.parquet as pq

path = Path("fcn_debug/20260601060000-0h-oper-fc.parquet")

pf = pq.ParquetFile(path)

cols = ["variable", "isobaricInhPa"]

var_counts = Counter()
levels_by_var = defaultdict(set)
total_rows = 0

for batch in pf.iter_batches(columns=cols, batch_size=1_000_000):
    table = batch.to_pydict()
    variables = table["variable"]
    levels = table["isobaricInhPa"]

    total_rows += len(variables)

    for v, lev in zip(variables, levels):
        var_counts[v] += 1
        if lev is not None:
            levels_by_var[v].add(lev)

print("total_rows_seen:", total_rows)

print("\nvariables:")
for k, v in sorted(var_counts.items()):
    print(f"{k:12s} {v}")

print("\nlevels_by_variable:")
for var in sorted(levels_by_var):
    levels = sorted(levels_by_var[var])
    print(var, len(levels), levels[:20], "..." if len(levels) > 20 else "")
