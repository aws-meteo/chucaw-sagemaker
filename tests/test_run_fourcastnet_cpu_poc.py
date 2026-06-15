import json
import subprocess
import sys
from pathlib import Path

SCRIPT = "scripts/run_fourcastnet_cpu_poc.py"
INPUT_S3_URI = (
    "s3://chucaw-data-platinum-processed-725644097028-us-east-1-an/ecmwf/fourcastnet/"
    "year=2026/month=06/day=06/hour=18z/20260606180000-0h-oper-fc_tensor.npy"
)


def _run(args, tmp_path):
    return subprocess.run(
        [sys.executable, SCRIPT, *args, "--artifacts-dir", str(tmp_path)],
        capture_output=True,
        text=True,
    )


def _manifest_record(tmp_path, mode):
    manifest_path = tmp_path / f"{mode}_canary_manifest.jsonl"
    line = manifest_path.read_text(encoding="utf-8").strip()
    return json.loads(line)


def test_manifest_generation_metadata_only(tmp_path):
    result = _run(["--input-s3-uri", INPUT_S3_URI, "--mode", "metadata_only"], tmp_path)
    assert result.returncode == 0

    record = _manifest_record(tmp_path, "metadata_only")
    assert record["mode"] == "metadata_only"
    assert record["input_s3_uri"] == INPUT_S3_URI


def test_manifest_generation_forward(tmp_path):
    result = _run(["--input-s3-uri", INPUT_S3_URI, "--mode", "forward"], tmp_path)
    assert result.returncode == 0

    record = _manifest_record(tmp_path, "forward")
    assert record["mode"] == "forward"
    assert record["input_s3_uri"] == INPUT_S3_URI


def test_metadata_only_and_forward_manifests_differ(tmp_path):
    result_metadata = _run(["--input-s3-uri", INPUT_S3_URI, "--mode", "metadata_only"], tmp_path)
    result_forward = _run(["--input-s3-uri", INPUT_S3_URI, "--mode", "forward"], tmp_path)
    assert result_metadata.returncode == 0
    assert result_forward.returncode == 0

    metadata_record = _manifest_record(tmp_path, "metadata_only")
    forward_record = _manifest_record(tmp_path, "forward")
    assert metadata_record["mode"] == "metadata_only"
    assert forward_record["mode"] == "forward"


def test_transform_job_dry_run_no_aws(tmp_path):
    result = _run(["--input-s3-uri", INPUT_S3_URI], tmp_path)
    assert result.returncode == 0
    assert "[DRY RUN]" in result.stdout
    assert "Creating Transform Job" not in result.stdout

    log = (tmp_path / "transform_job.log").read_text(encoding="utf-8")
    assert '"MaxConcurrentTransforms": 1' in log
    assert '"InstanceType": "ml.m5.large"' in log
    assert '"BatchStrategy": "SingleRecord"' in log


def test_no_endpoint_apis_in_script():
    content = Path(SCRIPT).read_text(encoding="utf-8")
    assert "create_endpoint(" not in content
    assert "create_endpoint_config(" not in content
    assert ".deploy(" not in content
    assert "run_fourcastnet_batch_transform.py" in content
