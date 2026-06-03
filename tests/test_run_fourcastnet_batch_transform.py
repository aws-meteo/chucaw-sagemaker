import sys
from unittest.mock import patch, MagicMock
import pytest
from pathlib import Path

# Add scripts to path to import the run_fourcastnet_batch_transform directly if possible,
# or we can test it via subprocess.
import subprocess

def test_no_endpoint_api_calls_in_script():
    script_path = Path("scripts/run_fourcastnet_batch_transform.py")
    content = script_path.read_text(encoding="utf-8")
    assert ".deploy(" not in content
    assert "create_endpoint(" not in content
    assert "create_endpoint_config(" not in content
    assert "create_transform_job" in content

def test_dry_run_never_executes():
    result = subprocess.run([
        sys.executable, "scripts/run_fourcastnet_batch_transform.py",
        "--model-name", "test-model",
        "--input-s3-uri", "s3://test/input",
        "--output-s3-uri", "s3://test/output"
    ], capture_output=True, text=True)
    
    assert result.returncode == 0
    assert "Planned Transform Job Payload" in result.stdout
    assert "DRY RUN" in result.stdout
    assert "Creating Transform Job..." not in result.stdout

def test_execute_mode_payload():
    # We will just dry run and check the printed payload to ensure tags are correct
    result = subprocess.run([
        sys.executable, "scripts/run_fourcastnet_batch_transform.py",
        "--model-name", "test-model",
        "--input-s3-uri", "s3://test/input",
        "--output-s3-uri", "s3://test/output"
    ], capture_output=True, text=True)
    
    assert result.returncode == 0
    assert '"Key": "CostMode"' in result.stdout
    assert '"Value": "batch-only"' in result.stdout
    assert '"Key": "Project"' in result.stdout
    assert '"TransformJobName": "fcn-batch-' in result.stdout

def test_config_tags_include_finished_lifecycle():
    result = subprocess.run([
        sys.executable, "scripts/run_fourcastnet_batch_transform.py",
        "--config", "configs/fourcastnet_batch_v0.json",
        "--input-s3-uri", "s3://test/input",
        "--output-s3-uri", "s3://test/output"
    ], capture_output=True, text=True)

    assert result.returncode == 0
    assert '"Key": "Lifecycle"' in result.stdout
    assert '"Value": "finished"' in result.stdout
