import json
import os
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

import pytest

from src.quicksight_abrupt_classifier.inference import input_fn, model_fn, output_fn, predict_fn


def _run_inference(csv_text: str, content_type: str = "text/csv") -> str:
    model = model_fn(str(Path(__file__).resolve().parents[1] / "src" / "quicksight_abrupt_classifier"))
    parsed = input_fn(csv_text, content_type=content_type)
    pred = predict_fn(parsed, model)
    output, _ = output_fn(pred, accept="text/csv")
    return output


def test_single_row_produces_single_output_row():
    out = _run_inference("-33.4500,-70.6600,285.4\n")
    rows = [r for r in out.splitlines() if r.strip()]
    assert len(rows) == 1
    assert len(rows[0].split(",")) == 2


def test_multiple_rows_preserve_count_and_order():
    out = _run_inference("-33.45,-70.66,285.4\n-34.00,-71.00,305.0\n45.00,7.00,280.0\n")
    rows = [r for r in out.splitlines() if r.strip()]
    assert len(rows) == 3
    labels = [int(row.split(",")[0]) for row in rows]
    assert labels == [0, 1, 0]


def test_labels_are_binary_and_scores_are_decimal():
    out = _run_inference("-33.45,-70.66,285.4\n-34.00,-71.00,305.0\n")
    for row in out.splitlines():
        label_s, score_s = row.split(",")
        label = int(label_s)
        score = float(score_s)
        assert label in (0, 1)
        assert score >= 0.0


def test_output_has_no_header():
    out = _run_inference("-33.4500,-70.6600,285.4\n")
    first = out.splitlines()[0].lower()
    assert "abrupt_temp_change_label" not in first


def test_invalid_lat_lon_fail_clearly():
    with pytest.raises(ValueError, match="lat out of range"):
        _run_inference("120.0,-70.6600,285.4\n")

    with pytest.raises(ValueError, match="lon out of range"):
        _run_inference("-33.0,190.0,285.4\n")


def test_invalid_t2m_fails_clearly():
    with pytest.raises(ValueError, match="invalid t2m"):
        _run_inference("-33.4500,-70.6600,not-a-number\n")


def test_content_type_csv_and_text_csv_are_accepted():
    _run_inference("-33.4500,-70.6600,285.4\n", content_type="text/csv")
    _run_inference("-33.4500,-70.6600,285.4\n", content_type="CSV")


def test_schema_textcsv_variant_matches_io_contract():
    repo_root = Path(__file__).resolve().parents[1]
    schema_csv = json.loads((repo_root / "schemas" / "quicksight_abrupt_classifier_schema.json").read_text(encoding="utf-8"))
    schema_textcsv = json.loads(
        (repo_root / "schemas" / "quicksight_abrupt_classifier_schema_textcsv.json").read_text(encoding="utf-8")
    )

    assert schema_textcsv["inputContentType"] == "text/csv"
    assert schema_textcsv["outputContentType"] == "text/csv"
    assert schema_textcsv["input"] == schema_csv["input"]
    assert schema_textcsv["output"] == schema_csv["output"]


def test_packaging_contains_required_code_files():
    repo_root = Path(__file__).resolve().parents[1]
    os.makedirs(repo_root / ".tmp", exist_ok=True)
    with tempfile.TemporaryDirectory(dir=repo_root / ".tmp") as tmp_dir:
        out_tar = Path(tmp_dir) / "quicksight_abrupt_model.tar.gz"
        cmd = [
            sys.executable,
            str(repo_root / "scripts" / "package_quicksight_abrupt_model.py"),
            "--output",
            str(out_tar),
        ]
        subprocess.run(cmd, check=True, cwd=repo_root)

        with tarfile.open(out_tar, "r:gz") as tf:
            names = set(tf.getnames())
        assert "code/inference.py" in names
        assert "code/model_config.json" in names


def test_local_smoke_script_still_passes():
    repo_root = Path(__file__).resolve().parents[1]
    os.makedirs(repo_root / ".tmp", exist_ok=True)
    with tempfile.TemporaryDirectory(dir=repo_root / ".tmp") as tmp_dir:
        out_csv = Path(tmp_dir) / "out.csv"
        cmd = [
            sys.executable,
            str(repo_root / "scripts" / "test_quicksight_csv_local.py"),
            "--input",
            str(repo_root / "examples" / "quicksight_abrupt_input.csv"),
            "--output",
            str(out_csv),
        ]
        subprocess.run(cmd, check=True, cwd=repo_root)
        assert out_csv.exists()
        rows = [line for line in out_csv.read_text(encoding="utf-8").splitlines() if line.strip()]
        assert len(rows) == 3
