from pathlib import Path
import pandas as pd

path = Path("fcn_debug/20260601060000-0h-oper-fc.parquet")

cols = ["variable", "isobaricinhpa"]
df = pd.read_parquet(path, columns=cols)

print(df.head())
print("shape:", df.shape)
print("memory_mib:", df.memory_usage(deep=True).sum()/1024/1024)

print("\nvariables:")
print(df["variable"].value_counts().sort_index())

print("\nlevels by variable:")
print(df.groupby("variable")["isobaricinhpa"].nunique().sort_index())
