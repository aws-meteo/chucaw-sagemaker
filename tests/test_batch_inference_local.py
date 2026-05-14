import subprocess
import sys
from pathlib import Path

import pandas as pd


def test_batch_inference_forecast_parquet_like_fixture(tmp_path: Path):
    output_path = tmp_path / "predictions.parquet"

    repo_root = Path(__file__).resolve().parents[1]
    input_path = repo_root / "tests" / "fixtures" / "forecast_parquet_like_t1000.csv"
    script_path = repo_root / "scripts" / "run_batch_inference_local.py"
    cmd = [
        sys.executable,
        str(script_path),
        "--input",
        str(input_path),
        "--output",
        str(output_path),
        "--source-year",
        "2026",
        "--source-month",
        "04",
        "--source-day",
        "09",
        "--source-hour",
        "18",
    ]
    subprocess.run(cmd, check=True)

    assert output_path.exists()
    out_df = pd.read_parquet(output_path)
    assert not out_df.empty
    assert out_df["predicted_value"].notna().all()
    assert {"latitude", "longitude", "predicted_value", "target_variable", "prediction_timestamp_utc"}.issubset(
        set(out_df.columns)
    )


def test_output_parquet_roundtrip_with_pyarrow_engine(tmp_path: Path):
    df = pd.DataFrame(
        [{"latitude": -33.5, "longitude": -70.6, "predicted_value": 285.2, "target_variable": "t"}]
    )
    path = tmp_path / "roundtrip.parquet"
    df.to_parquet(path, engine="pyarrow", index=False)
    loaded = pd.read_parquet(path, engine="pyarrow")
    assert loaded.loc[0, "predicted_value"] == 285.2
