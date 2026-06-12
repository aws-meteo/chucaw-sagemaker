import numpy as np
from pathlib import Path

out = Path("fcn_debug/synthetic_fcn_input.npy")
x = np.zeros((1, 20, 720, 1440), dtype=np.float32)
np.save(out, x)

print("written:", out)
print("shape:", x.shape)
print("dtype:", x.dtype)
print("size_mib:", out.stat().st_size / 1024 / 1024)
