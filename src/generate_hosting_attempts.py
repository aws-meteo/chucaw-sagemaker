#!/usr/bin/env python3
"""Diagnostic-only hosting artifact experiment generator.

Not part of the operational baseline. Baseline is the separate-source serving
bundle documented in docs/team_runbook.md.
"""
import json
import shutil
import subprocess
import sys
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import joblib


@dataclass
class AttemptSpec:
    name: str
    folder: str
    strategy: str
    expected_behavior: str


REPO_ROOT = Path(__file__).resolve().parents[1]
TMP_MODEL_PATH = REPO_ROOT / "tmp_model" / "model.joblib"
INFERENCE_SOURCE = REPO_ROOT / "inference" / "inference.py"
INFERENCE_REQUIREMENTS = REPO_ROOT / "inference" / "requirements.txt"
ATTEMPTS_ROOT = REPO_ROOT / "artifacts" / "hosting_attempts"
SMOKE_TEST_SCRIPT = REPO_ROOT / "src" / "smoke_test_local.py"

ATTEMPTS = [
    AttemptSpec(
        name="attempt_01_model_code_bundle",
        folder="attempt_01_model_code_bundle",
        strategy="Canonical model bundle with `model.joblib` + `code/inference.py` inside `model.tar.gz`.",
        expected_behavior=(
            "Framework container should find `/opt/ml/model/code/inference.py` when "
            "`SAGEMAKER_PROGRAM=inference.py`."
        ),
    ),
    AttemptSpec(
        name="attempt_02_submit_dir_tar_bundle",
        folder="attempt_02_submit_dir_tar_bundle",
        strategy=(
            "Split bundle: `model.tar.gz` has only `model.joblib`; serving code goes to separate "
            "`source.tar.gz` for `SAGEMAKER_SUBMIT_DIRECTORY`."
        ),
        expected_behavior=(
            "Container should pull serving code from `SAGEMAKER_SUBMIT_DIRECTORY=s3://.../source.tar.gz` "
            "instead of `/opt/ml/model/code`."
        ),
    ),
    AttemptSpec(
        name="attempt_03_root_inference_bundle",
        folder="attempt_03_root_inference_bundle",
        strategy=(
            "Diagnostic hybrid: `model.tar.gz` includes root `inference.py` and `code/inference.py` "
            "plus `model.joblib`."
        ),
        expected_behavior=(
            "If runtime module resolution differs, either root-level or `code/` entry may succeed."
        ),
    ),
]


def fail_if_missing_inputs() -> None:
    missing = []
    if not TMP_MODEL_PATH.exists():
        missing.append(str(TMP_MODEL_PATH))
    if not INFERENCE_SOURCE.exists():
        missing.append(str(INFERENCE_SOURCE))
    if missing:
        detail = "\n".join(f"- {item}" for item in missing)
        raise FileNotFoundError(
            "Missing required input files:\n"
            f"{detail}\n\n"
            "How to recover tmp_model/model.joblib from training artifact:\n"
            "1) Download training output model.tar.gz to local disk.\n"
            "2) Run: tar -xzf <training-output-model.tar.gz> -C tmp_model\n"
            "3) Verify file exists: tmp_model/model.joblib"
        )


def reset_attempts_root() -> None:
    if ATTEMPTS_ROOT.exists():
        shutil.rmtree(ATTEMPTS_ROOT)
    ATTEMPTS_ROOT.mkdir(parents=True, exist_ok=True)


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def create_tar_from_dir(source_dir: Path, tar_path: Path) -> List[str]:
    entries = []
    with tarfile.open(tar_path, "w:gz") as tar:
        for path in sorted(source_dir.rglob("*")):
            if path.is_dir():
                continue
            arcname = path.relative_to(source_dir).as_posix()
            tar.add(path, arcname=arcname)
            entries.append(arcname)
    return entries


def read_tar_entries(tar_path: Path) -> List[str]:
    with tarfile.open(tar_path, "r:gz") as tar:
        return sorted(name.rstrip("/") for name in tar.getnames())


def extract_tar(tar_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=output_dir)


def load_joblib(path: Path) -> Dict:
    obj = joblib.load(path)
    if not isinstance(obj, dict):
        raise TypeError(f"Expected dict model artifact at {path}, got {type(obj)}")
    return obj


def format_tree(root: Path) -> str:
    lines = [root.name + "/"]
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root).as_posix()
        suffix = "/" if path.is_dir() else ""
        lines.append(f"{rel}{suffix}")
    return "\n".join(lines)


def run_smoke_test(artifact_path: Path) -> Dict[str, str]:
    cmd = [
        sys.executable,
        str(SMOKE_TEST_SCRIPT),
        "--artifact-local",
        str(artifact_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO_ROOT))
    merged_lines = [line.strip() for line in (proc.stdout + "\n" + proc.stderr).splitlines() if line.strip()]
    key_line = "(no output)"
    for line in merged_lines:
        if "ERROR:" in line:
            key_line = line
            break
    if key_line == "(no output)":
        for line in reversed(merged_lines):
            if "Local smoke test passed." in line:
                key_line = line
                break
    if key_line == "(no output)" and merged_lines:
        key_line = merged_lines[-1]
    return {
        "command": " ".join(cmd),
        "returncode": str(proc.returncode),
        "key_line": key_line,
    }


def write_bundle_tree_file(
    attempt_dir: Path,
    model_entries: List[str],
    source_entries: Optional[List[str]] = None,
) -> None:
    lines = []
    lines.append("# Attempt folder tree")
    lines.append(format_tree(attempt_dir))
    lines.append("")
    lines.append("# model.tar.gz (tar -tzf)")
    lines.extend(model_entries if model_entries else ["(empty)"])
    if source_entries is not None:
        lines.append("")
        lines.append("# source.tar.gz (tar -tzf)")
        lines.extend(source_entries if source_entries else ["(empty)"])
    (attempt_dir / "bundle_tree.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_attempt_readme(
    attempt: AttemptSpec,
    attempt_dir: Path,
    summary: Dict[str, str],
    notes: List[str],
) -> None:
    readme_lines = [
        f"# {attempt.name}",
        "",
        "## Strategy",
        attempt.strategy,
        "",
        "## Expected SageMaker behavior",
        attempt.expected_behavior,
        "",
        "## Local checks",
        f"- model.tar.gz entries check: {summary['model_entries_status']}",
        f"- model.joblib load from extracted tar: {summary['joblib_status']}",
        f"- inference file location check: {summary['inference_path_status']}",
        f"- smoke test command: `{summary['smoke_command']}`",
        f"- smoke test return code: {summary['smoke_returncode']}",
        f"- smoke test key line: `{summary['smoke_key_line']}`",
    ]
    if notes:
        readme_lines.append("")
        readme_lines.append("## Notes")
        for note in notes:
            readme_lines.append(f"- {note}")
    (attempt_dir / "README.md").write_text("\n".join(readme_lines) + "\n", encoding="utf-8")


def attempt_01(attempt: AttemptSpec) -> Dict[str, str]:
    attempt_dir = ATTEMPTS_ROOT / attempt.folder
    staging_dir = attempt_dir / "staging"
    model_stage = staging_dir / "model_bundle"
    validation_dir = attempt_dir / "validation_extracted_model"

    model_stage.mkdir(parents=True, exist_ok=True)
    copy_file(TMP_MODEL_PATH, model_stage / "model.joblib")
    copy_file(INFERENCE_SOURCE, model_stage / "code" / "inference.py")

    model_tar = attempt_dir / "model.tar.gz"
    create_tar_from_dir(model_stage, model_tar)
    model_entries = read_tar_entries(model_tar)
    extract_tar(model_tar, validation_dir)

    extracted_model = validation_dir / "model.joblib"
    extracted_inference = validation_dir / "code" / "inference.py"
    load_joblib(extracted_model)

    smoke = run_smoke_test(model_tar)
    write_bundle_tree_file(attempt_dir, model_entries)

    summary = {
        "model_entries_status": "PASS",
        "joblib_status": "PASS",
        "inference_path_status": "PASS" if extracted_inference.exists() else "FAIL",
        "smoke_command": smoke["command"],
        "smoke_returncode": smoke["returncode"],
        "smoke_key_line": smoke["key_line"],
    }
    notes = [
        "No `code/requirements.txt` included on purpose. Diagnostic focuses on missing `/opt/ml/model/code` failure.",
    ]
    write_attempt_readme(attempt, attempt_dir, summary, notes)
    return {
        "name": attempt.name,
        "attempt_dir": str(attempt_dir),
        "model_tar": str(model_tar),
        "source_tar": "",
        "smoke_returncode": smoke["returncode"],
        "smoke_key_line": smoke["key_line"],
    }


def attempt_02(attempt: AttemptSpec) -> Dict[str, str]:
    attempt_dir = ATTEMPTS_ROOT / attempt.folder
    staging_dir = attempt_dir / "staging"
    model_stage = staging_dir / "model_only"
    source_stage = staging_dir / "source_bundle"
    validation_dir = attempt_dir / "validation_extracted_model"
    source_validation_dir = attempt_dir / "validation_extracted_source"

    model_stage.mkdir(parents=True, exist_ok=True)
    source_stage.mkdir(parents=True, exist_ok=True)

    copy_file(TMP_MODEL_PATH, model_stage / "model.joblib")
    copy_file(INFERENCE_SOURCE, source_stage / "inference.py")
    copy_file(INFERENCE_REQUIREMENTS, source_stage / "requirements.txt")

    model_tar = attempt_dir / "model.tar.gz"
    source_tar = attempt_dir / "source.tar.gz"
    create_tar_from_dir(model_stage, model_tar)
    create_tar_from_dir(source_stage, source_tar)

    model_entries = read_tar_entries(model_tar)
    source_entries = read_tar_entries(source_tar)
    extract_tar(model_tar, validation_dir)
    extract_tar(source_tar, source_validation_dir)

    extracted_model = validation_dir / "model.joblib"
    load_joblib(extracted_model)

    model_has_inference = (validation_dir / "code" / "inference.py").exists()
    source_has_inference = (source_validation_dir / "inference.py").exists()

    smoke = run_smoke_test(model_tar)
    write_bundle_tree_file(attempt_dir, model_entries, source_entries=source_entries)

    summary = {
        "model_entries_status": "PASS",
        "joblib_status": "PASS",
        "inference_path_status": (
            "PASS (model.tar.gz intentionally has no inference; source.tar.gz has inference.py)"
            if (not model_has_inference and source_has_inference)
            else "FAIL"
        ),
        "smoke_command": smoke["command"],
        "smoke_returncode": smoke["returncode"],
        "smoke_key_line": smoke["key_line"],
    }
    notes = [
        "This strategy requires deploy env `SAGEMAKER_SUBMIT_DIRECTORY` to point at `source.tar.gz` S3 URI.",
        "Local smoke test against `model.tar.gz` expected to fail because smoke helper expects `code/inference.py` inside model tarball.",
    ]
    write_attempt_readme(attempt, attempt_dir, summary, notes)
    return {
        "name": attempt.name,
        "attempt_dir": str(attempt_dir),
        "model_tar": str(model_tar),
        "source_tar": str(source_tar),
        "smoke_returncode": smoke["returncode"],
        "smoke_key_line": smoke["key_line"],
    }


def attempt_03(attempt: AttemptSpec) -> Dict[str, str]:
    attempt_dir = ATTEMPTS_ROOT / attempt.folder
    staging_dir = attempt_dir / "staging"
    model_stage = staging_dir / "root_inference_bundle"
    validation_dir = attempt_dir / "validation_extracted_model"

    model_stage.mkdir(parents=True, exist_ok=True)
    copy_file(TMP_MODEL_PATH, model_stage / "model.joblib")
    copy_file(INFERENCE_SOURCE, model_stage / "inference.py")
    copy_file(INFERENCE_SOURCE, model_stage / "code" / "inference.py")

    model_tar = attempt_dir / "model.tar.gz"
    create_tar_from_dir(model_stage, model_tar)
    model_entries = read_tar_entries(model_tar)
    extract_tar(model_tar, validation_dir)

    extracted_model = validation_dir / "model.joblib"
    extracted_root_inference = validation_dir / "inference.py"
    extracted_code_inference = validation_dir / "code" / "inference.py"
    load_joblib(extracted_model)

    smoke = run_smoke_test(model_tar)
    write_bundle_tree_file(attempt_dir, model_entries)

    summary = {
        "model_entries_status": "PASS",
        "joblib_status": "PASS",
        "inference_path_status": (
            "PASS"
            if extracted_root_inference.exists() and extracted_code_inference.exists()
            else "FAIL"
        ),
        "smoke_command": smoke["command"],
        "smoke_returncode": smoke["returncode"],
        "smoke_key_line": smoke["key_line"],
    }
    notes = [
        "Includes both root `inference.py` and `code/inference.py` to diagnose path resolution differences.",
        "Non-canonical layout; diagnostic only.",
    ]
    write_attempt_readme(attempt, attempt_dir, summary, notes)
    return {
        "name": attempt.name,
        "attempt_dir": str(attempt_dir),
        "model_tar": str(model_tar),
        "source_tar": "",
        "smoke_returncode": smoke["returncode"],
        "smoke_key_line": smoke["key_line"],
    }


def main() -> None:
    fail_if_missing_inputs()

    model_obj = load_joblib(TMP_MODEL_PATH)
    print(f"Input model: {TMP_MODEL_PATH}")
    print(f"Input model keys: {sorted(model_obj.keys())}")

    reset_attempts_root()

    results = []
    for attempt in ATTEMPTS:
        if attempt.name == "attempt_01_model_code_bundle":
            result = attempt_01(attempt)
        elif attempt.name == "attempt_02_submit_dir_tar_bundle":
            result = attempt_02(attempt)
        elif attempt.name == "attempt_03_root_inference_bundle":
            result = attempt_03(attempt)
        else:
            raise RuntimeError(f"Unknown attempt name: {attempt.name}")
        results.append(result)

    summary_path = ATTEMPTS_ROOT / "generation_summary.json"
    summary_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(f"Generated attempts root: {ATTEMPTS_ROOT}")
    for item in results:
        print(f"- {item['name']}: model={item['model_tar']}")
        if item["source_tar"]:
            print(f"  source={item['source_tar']}")
        print(f"  smoke rc={item['smoke_returncode']} | {item['smoke_key_line']}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
