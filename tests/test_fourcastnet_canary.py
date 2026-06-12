import subprocess
import sys
import json
from pathlib import Path

def test_canary_generator_creates_artifacts(tmp_path):
    generator_script = Path("scripts/create_fourcastnet_canary_payload.py")
    outdir = tmp_path / "canary"
    cmd = [
        sys.executable, str(generator_script),
        "--output-dir", str(outdir),
        "--input-s3-uri", "s3://test/in",
        "--output-s3-uri", "s3://test/out"
    ]
    res = subprocess.run(cmd, check=True)
    assert res.returncode == 0
    
    assert (outdir / "canary_tensor.npy").exists()
    
    manifest = outdir / "canary_manifest.jsonl"
    assert manifest.exists()
    
    lines = manifest.read_text().strip().split("\n")
    assert len(lines) == 1
    
    data = json.loads(lines[0])
    assert data["max_runtime_guard"] is True
    assert data["mode"] == "metadata_only"

def test_run_script_rejects_multirecord_json():
    run_script = Path("scripts/run_fourcastnet_batch_transform.py")
    cmd = [
        sys.executable, str(run_script),
        "--model-name", "m",
        "--input-s3-uri", "s3://a",
        "--output-s3-uri", "s3://b",
        "--content-type", "application/jsonlines",
        "--strategy", "MultiRecord"
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 1
    assert "BatchStrategy=MultiRecord is not supported" in res.stderr

def test_run_script_rejects_placeholder_in_execute():
    run_script = Path("scripts/run_fourcastnet_batch_transform.py")
    cmd = [
        sys.executable, str(run_script),
        "--model-name", "m",
        "--input-s3-uri", "s3://PLACEHOLDER-BUCKET",
        "--output-s3-uri", "s3://PLACEHOLDER-BUCKET",
        "--execute"
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 1
    assert "Placeholder S3 URIs cannot be used" in res.stderr
