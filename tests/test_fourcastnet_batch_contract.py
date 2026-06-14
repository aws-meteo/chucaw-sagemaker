import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.fourcastnet.serving.inference import input_fn


def test_fourcastnet_input_fn_accepts_json_record():
    payload = json.dumps(
        {
            "input_s3_uri": "s3://bucket/prefix/input_tensor.npy",
            "output_s3_uri": "s3://bucket/prefix/outputs/run-id/",
            "mode": "metadata_only",
        }
    )
    for ct in ["application/json", "application/jsonlines", "application/x-jsonlines"]:
        parsed = input_fn(payload, ct)
        assert parsed["mode"] == "metadata_only"
        assert parsed["input_s3_uri"].endswith("input_tensor.npy")
        assert parsed["output_s3_uri"].endswith("/outputs/run-id/")



def test_run_fourcastnet_batch_transform_dry_run_has_no_endpoint_calls():
    script_path = REPO_ROOT / "scripts" / "run_fourcastnet_batch_transform.py"
    cmd = [
        sys.executable,
        str(script_path),
        "--config",
        str(REPO_ROOT / "configs" / "fourcastnet_batch_v0.json"),
        "--input-s3-uri",
        "s3://bucket/prefix/input/requests.jsonl",
        "--output-s3-uri",
        "s3://bucket/prefix/output/",
        "--region",
        "us-east-1",
        "--profile",
        "sbnai-725",
    ]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    combined = (result.stdout + result.stderr).lower()
    assert "create-transform-job" in combined
    assert "create_endpoint" not in combined
    assert "create-endpoint" not in combined


def test_fourcastnet_packaging_script_targets_serving_dir():
    packaging_script = REPO_ROOT / "src" / "fourcastnet" / "build_fcn_hosting_artifact.py"
    text = packaging_script.read_text(encoding="utf-8")
    assert 'default="src/fourcastnet/serving"' in text
    assert "inference/inference.py" not in text


def test_package_fourcastnet_model_script():
    packaging_script = REPO_ROOT / "scripts" / "package_fourcastnet_model.py"
    text = packaging_script.read_text(encoding="utf-8")
    assert 'DEFAULT_SERVING = "src/fourcastnet/serving"' in text
    assert "inference/inference.py" not in text


def test_describe_or_create_model_conflict():
    script_path = REPO_ROOT / "scripts" / "describe_or_create_fourcastnet_model.py"
    cmd = [
        sys.executable,
        str(script_path),
        "--model-data-url",
        "s3://chucaw-data-platinum-processed-725644097028-us-east-1-an/sagemaker/fourcastnet/fcn-v1/model/model.tar.gz",
        "--model-package-arn",
        "arn:aws:sagemaker:us-east-1:725644097028:model-package/sbnai-fourcastnet-fcn-v0/1",
        "--region",
        "us-east-1"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode != 0
    assert "Conflict detected" in result.stdout or "Conflict detected" in result.stderr


