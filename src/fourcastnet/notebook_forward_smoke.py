#!/usr/bin/env python3
"""Studio/Notebook GPU smoke test for FourCastNet using Model Registry + S3 artifacts.

Environments supported:
  A. Local PC: pass --profile sbnai-725 for boto3 credentials.
  B. SageMaker Studio/Notebook GPU: omit --profile; instance role is used automatically.

Usage examples:
  # metadata_only (local, with profile)
  python src/fourcastnet/notebook_forward_smoke.py \\
    --model-package-arn arn:aws:sagemaker:us-east-1:725644097028:model-package/sbnai-fourcastnet-fcn-v0/1 \\
    --input-tensor-s3-uri s3://chucaw-data-platinum-processed-725644097028-us-east-1-an/sagemaker/fourcastnet/fcn-v0/input/input_tensor.npy \\
    --mode metadata_only --region us-east-1 --profile sbnai-725 \\
    --output-report-s3-uri s3://bucket/prefix/report.json

  # forward (Studio/Notebook GPU, no profile)
  python src/fourcastnet/notebook_forward_smoke.py \\
    --model-package-arn arn:aws:sagemaker:us-east-1:725644097028:model-package/sbnai-fourcastnet-fcn-v0/1 \\
    --input-tensor-s3-uri s3://chucaw-data-platinum-processed-725644097028-us-east-1-an/sagemaker/fourcastnet/fcn-v0/input/input_tensor.npy \\
    --mode forward --region us-east-1
"""

from __future__ import annotations

import argparse
import io
import json
import platform
import tarfile
import tempfile
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover - defensive runtime fallback
    torch = None  # type: ignore[assignment]


DEFAULT_MODEL_PACKAGE_ARN = "arn:aws:sagemaker:us-east-1:725644097028:model-package/sbnai-fourcastnet-fcn-v0/1"
DEFAULT_INPUT_TENSOR_S3_URI = (
    "s3://chucaw-data-platinum-processed-725644097028-us-east-1-an/sagemaker/fourcastnet/fcn-v0/input/input_tensor.npy"
)
DEFAULT_MODE = "metadata_only"
DEFAULT_REGION = "us-east-1"
MAX_GUARD_ELEMENTS = 50_000_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run FourCastNet registry-based smoke test on notebook/Studio GPU",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model-package-arn",
        required=True,
        help="SageMaker Model Package ARN (required)",
    )
    parser.add_argument(
        "--input-tensor-s3-uri",
        required=True,
        help="S3 URI for input_tensor.npy (required)",
    )
    parser.add_argument(
        "--mode",
        choices=["metadata_only", "forward"],
        required=True,
        help="metadata_only: plumbing checks only; forward: real model import + checkpoint load + inference",
    )
    parser.add_argument(
        "--region",
        required=True,
        help="AWS region (required)",
    )
    parser.add_argument(
        "--profile",
        default="",
        help="AWS profile name. Leave empty inside Studio/Notebook (instance role used automatically).",
    )
    parser.add_argument(
        "--output-report-s3-uri",
        default="",
        help="Optional S3 URI to upload the JSON report after writing locally.",
    )
    parser.add_argument(
        "--work-dir",
        default="",
        help=(
            "Optional local directory for model artifact extraction and report output. "
            "Defaults to a system temp directory."
        ),
    )
    # Legacy: keep --output-report for backward compatibility with existing notebook cells
    parser.add_argument(
        "--output-report",
        default="",
        help="Optional local path for the JSON report (legacy; --work-dir preferred). "
             "If set, this path is used for the local report file.",
    )
    parser.add_argument(
        "--max-runtime-guard",
        action="store_true",
        help="Abort forward pass if input tensor exceeds MAX_GUARD_ELEMENTS.",
    )
    return parser.parse_args()


def split_s3_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("s3://"):
        raise ValueError(f"Expected s3:// URI, got {uri!r}")
    remainder = uri[len("s3://"):]
    bucket, sep, key = remainder.partition("/")
    if not bucket or not sep or not key:
        raise ValueError(f"Invalid S3 URI: {uri!r}")
    return bucket, key


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def make_session(profile: str, region: str):
    """Create a boto3 session.

    In Studio/Notebook (no profile) the instance role is picked up automatically.
    Locally, pass a named profile.
    """
    import boto3  # noqa: PLC0415

    if profile:
        return boto3.session.Session(profile_name=profile, region_name=region)
    return boto3.session.Session(region_name=region)


def read_s3_bytes(s3_uri: str, profile: str, region: str) -> bytes:
    bucket, key = split_s3_uri(s3_uri)
    s3 = make_session(profile, region).client("s3")
    response = s3.get_object(Bucket=bucket, Key=key)
    return response["Body"].read()


def upload_s3_bytes(data: bytes, s3_uri: str, profile: str, region: str, content_type: str = "application/json") -> None:
    bucket, key = split_s3_uri(s3_uri)
    s3 = make_session(profile, region).client("s3")
    s3.put_object(Bucket=bucket, Key=key, Body=data, ContentType=content_type)


def write_local_report(local_path: str, payload: dict[str, Any]) -> str:
    """Write JSON report to local filesystem. Returns resolved path string."""
    path = Path(local_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return str(path.resolve())


def resolve_local_report_path(args: argparse.Namespace, work_dir: Path) -> str:
    """Resolve where to write the local JSON report.

    Priority:
      1. --output-report (legacy explicit path)
      2. --work-dir / smoke_report.json
      3. Fallback: artifacts/fourcastnet/notebook_smoke/notebook_forward_smoke_report.json
    """
    if args.output_report:
        return args.output_report
    if args.work_dir:
        return str(work_dir / "smoke_report.json")
    return "artifacts/fourcastnet/notebook_smoke/notebook_forward_smoke_report.json"


def describe_model_package(model_package_arn: str, profile: str, region: str) -> dict[str, Any]:
    sm = make_session(profile, region).client("sagemaker")
    return sm.describe_model_package(ModelPackageName=model_package_arn)


def extract_model_data_url(model_package_desc: dict[str, Any]) -> str:
    spec = model_package_desc.get("InferenceSpecification") or {}
    containers = spec.get("Containers") or []
    for container in containers:
        model_data_url = str(container.get("ModelDataUrl", "")).strip()
        if model_data_url:
            return model_data_url

    additional = model_package_desc.get("AdditionalInferenceSpecifications") or []
    for item in additional:
        for container in (item.get("Containers") or []):
            model_data_url = str(container.get("ModelDataUrl", "")).strip()
            if model_data_url:
                return model_data_url

    raise ValueError("ModelDataUrl not found in model package inference specifications")


def unpack_model_tar(model_tar_bytes: bytes, extract_dir: Path) -> Path:
    """Extract model.tar.gz into extract_dir/model/ and return that path."""
    model_root = extract_dir / "model"
    model_root.mkdir(parents=True, exist_ok=True)
    tar_path = extract_dir / "model.tar.gz"
    tar_path.write_bytes(model_tar_bytes)
    with tarfile.open(tar_path, mode="r:gz") as tf:
        try:
            tf.extractall(model_root, filter="data")  # Python 3.12+ safe extraction
        except TypeError:
            tf.extractall(model_root)  # Python <3.12 fallback
    return model_root


def locate_asset(root: Path, filename: str) -> Path:
    candidates = [root / filename, root / "code" / filename]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    matches = list(root.rglob(filename))
    if matches:
        return matches[0]
    return root / filename


def load_npy_from_s3(s3_uri: str, profile: str, region: str) -> np.ndarray:
    raw = read_s3_bytes(s3_uri, profile=profile, region=region)
    return np.load(io.BytesIO(raw), allow_pickle=False)


def tensor_meta(array: np.ndarray) -> dict[str, Any]:
    return {
        "dtype": str(array.dtype),
        "shape": [int(dim) for dim in array.shape],
        "finite": bool(np.isfinite(array).all()),
        "nan_count": int(np.isnan(array).sum()),
        "min": float(np.nanmin(array)),
        "max": float(np.nanmax(array)),
        "mean": float(np.nanmean(array)),
        "std": float(np.nanstd(array)),
    }


def device_meta() -> dict[str, Any]:
    info: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch_available": bool(torch is not None),
        "cuda_available": False,
        "cuda_device_count": 0,
        "cuda_device_name": "",
    }
    if torch is None:
        return info

    info["torch_version"] = getattr(torch, "__version__", "unknown")
    cuda_available = bool(torch.cuda.is_available())
    info["cuda_available"] = cuda_available
    if cuda_available:
        count = int(torch.cuda.device_count())
        info["cuda_device_count"] = count
        if count > 0:
            info["cuda_device_name"] = torch.cuda.get_device_name(0)
            free, total = torch.cuda.mem_get_info(0)
            info["cuda_mem_free_bytes"] = int(free)
            info["cuda_mem_total_bytes"] = int(total)
    return info


def probe_backend() -> dict[str, Any]:
    """Try to import the FourCastNet model class from known packages."""
    probes = [
        ("modulus.models.fcn", "FourCastNet"),
        ("modulus.models.fourcastnet", "FourCastNet"),
        ("physicsnemo.models.fourcastnet", "FourCastNet"),
        ("fourcastnet", "FourCastNet"),
    ]
    attempts: list[dict[str, str]] = []
    for module_name, symbol_name in probes:
        try:
            module = __import__(module_name, fromlist=[symbol_name])
            symbol = getattr(module, symbol_name, None)
            if symbol is not None:
                return {"ok": True, "module": module_name, "symbol": symbol_name, "attempts": attempts}
            attempts.append({"module": module_name, "error": f"missing symbol {symbol_name}"})
        except Exception as exc:
            attempts.append({"module": module_name, "error": f"{type(exc).__name__}: {exc}"})
    return {"ok": False, "reason": "backend_not_found", "attempts": attempts}


def run_forward(
    input_tensor: np.ndarray,
    checkpoint_path: Path,
    backend: dict[str, Any],
    runtime_guard: bool,
) -> dict[str, Any]:
    """Attempt real model import, checkpoint load, and forward pass.

    Returns a dict with fourcastnet_proven=True ONLY if the full pass succeeds.
    fourcastnet_proven is NEVER set to True for partial/missing results.
    """
    if not backend.get("ok"):
        return {
            "ok": False,
            "reason": "backend_not_found",
            "fourcastnet_proven": False,
            "backend": backend,
            "note": (
                "FourCastNet backend code is missing. Install modulus, physicsnemo, or "
                "a custom fourcastnet package in the Studio/Notebook environment before running forward."
            ),
        }
    if torch is None:
        return {"ok": False, "reason": "torch_unavailable", "fourcastnet_proven": False}
    if runtime_guard and input_tensor.size > MAX_GUARD_ELEMENTS:
        return {
            "ok": False,
            "reason": "runtime_guard_blocked_large_tensor",
            "tensor_elements": int(input_tensor.size),
            "guard_elements_limit": MAX_GUARD_ELEMENTS,
            "fourcastnet_proven": False,
        }

    module_name = str(backend["module"])
    symbol_name = str(backend["symbol"])
    started = time.time()
    try:
        module = __import__(module_name, fromlist=[symbol_name])
        model_cls = getattr(module, symbol_name)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        load_attempts: list[str] = []
        instance = None
        if hasattr(model_cls, "load_from_checkpoint"):
            try:
                instance = model_cls.load_from_checkpoint(str(checkpoint_path), map_location=device)
                load_attempts.append("load_from_checkpoint: OK")
            except Exception as exc:
                load_attempts.append(f"load_from_checkpoint failed: {type(exc).__name__}: {exc}")

        if instance is None:
            try:
                instance = model_cls()
                load_attempts.append("model_cls(): OK")
            except Exception as exc:
                load_attempts.append(f"model_cls() failed: {type(exc).__name__}: {exc}")

        if instance is None:
            elapsed = time.time() - started
            return {
                "ok": False,
                "reason": "model_instantiation_failed",
                "load_attempts": load_attempts,
                "fourcastnet_proven": False,
                "runtime_seconds": elapsed,
            }

        instance.eval()
        instance.to(device)
        tensor_batch = np.asarray(input_tensor, dtype=np.float32)
        with torch.no_grad():
            output = instance(torch.from_numpy(tensor_batch).to(device))

        if hasattr(output, "detach"):
            output_np = output.detach().cpu().numpy()
        else:
            output_np = np.asarray(output)

        elapsed = time.time() - started
        return {
            "ok": True,
            "fourcastnet_proven": True,
            "input_shape": [int(dim) for dim in tensor_batch.shape],
            "output_shape": [int(dim) for dim in output_np.shape],
            "runtime_seconds": elapsed,
            "backend_module": module_name,
            "backend_symbol": symbol_name,
            "load_attempts": load_attempts,
        }
    except Exception as exc:
        elapsed = time.time() - started
        return {
            "ok": False,
            "reason": "forward_failed",
            "fourcastnet_proven": False,
            "runtime_seconds": elapsed,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "traceback": traceback.format_exc(limit=8),
            "backend_module": module_name,
            "backend_symbol": symbol_name,
        }


def main() -> int:
    args = parse_args()
    model_package_arn = args.model_package_arn.strip()
    input_tensor_s3_uri = args.input_tensor_s3_uri.strip()
    region = args.region.strip()
    profile = (args.profile or "").strip()
    output_report_s3_uri = (args.output_report_s3_uri or "").strip()

    # Validate S3 URIs early
    split_s3_uri(input_tensor_s3_uri)
    if output_report_s3_uri:
        split_s3_uri(output_report_s3_uri)

    # Resolve work directory
    if args.work_dir:
        work_dir = Path(args.work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        _tmp_cleanup = None
    else:
        import tempfile as _tempfile  # noqa: PLC0415
        _tmp_obj = _tempfile.TemporaryDirectory(prefix="fourcastnet_smoke_")
        work_dir = Path(_tmp_obj.name)
        _tmp_cleanup = _tmp_obj  # keep reference so it is not GC'd

    local_report_path = resolve_local_report_path(args, work_dir)

    report: dict[str, Any] = {
        "timestamp_utc": now_iso(),
        "mode": args.mode,
        "ok": False,
        "fourcastnet_proven": False,
        "model_package_arn": model_package_arn,
        "input_tensor_s3_uri": input_tensor_s3_uri,
        "environment": {
            "profile_used": bool(profile),
            "work_dir": str(work_dir),
        },
    }

    try:
        # --- Device / torch snapshot ---
        report["device"] = device_meta()

        # --- Step 1: Describe model package ---
        desc = describe_model_package(model_package_arn=model_package_arn, profile=profile, region=region)
        model_data_url = extract_model_data_url(desc)
        report["model_package_status"] = str(desc.get("ModelPackageStatus", ""))
        report["model_approval_status"] = str(desc.get("ModelApprovalStatus", ""))
        report["model_data_url"] = model_data_url

        # --- Step 2: Download model.tar.gz ---
        print(f"[smoke] Downloading model.tar.gz from {model_data_url} ...", flush=True)
        model_tar_bytes = read_s3_bytes(model_data_url, profile=profile, region=region)
        report["model_tar_size_bytes"] = len(model_tar_bytes)

        # --- Step 3: Extract model.tar.gz ---
        print(f"[smoke] Extracting model.tar.gz into {work_dir} ...", flush=True)
        model_root = unpack_model_tar(model_tar_bytes, work_dir)

        # --- Step 4: Locate assets ---
        checkpoint = locate_asset(model_root, "backbone.ckpt")
        global_means = locate_asset(model_root, "global_means.npy")
        global_stds = locate_asset(model_root, "global_stds.npy")

        report["assets"] = {
            "model_root": str(model_root),
            "checkpoint_path": str(checkpoint),
            "global_means_path": str(global_means),
            "global_stds_path": str(global_stds),
            "checkpoint_exists": checkpoint.exists(),
            "global_means_exists": global_means.exists(),
            "global_stds_exists": global_stds.exists(),
        }
        if not checkpoint.exists():
            report["result"] = "missing_backbone_ckpt"
            report["error_message"] = f"backbone.ckpt not found under {model_root}"
            _finalize_report(report, local_report_path, output_report_s3_uri, profile, region)
            return 1
        if not all([global_means.exists(), global_stds.exists()]):
            report["result"] = "missing_norm_stats"
            report["error_message"] = "global_means.npy or global_stds.npy not found"
            _finalize_report(report, local_report_path, output_report_s3_uri, profile, region)
            return 1

        # --- Step 5: Download and validate input tensor ---
        print(f"[smoke] Downloading input_tensor.npy from {input_tensor_s3_uri} ...", flush=True)
        input_tensor = np.asarray(load_npy_from_s3(input_tensor_s3_uri, profile=profile, region=region), dtype=np.float32)
        means = np.asarray(np.load(global_means, allow_pickle=False), dtype=np.float32)
        stds = np.asarray(np.load(global_stds, allow_pickle=False), dtype=np.float32)

        report["input_tensor"] = tensor_meta(input_tensor)
        report["global_means"] = tensor_meta(means)
        report["global_stds"] = tensor_meta(stds)

        # --- Step 6: Backend probe (always) ---
        report["backend_probe"] = probe_backend()

        # --- Step 7: Mode dispatch ---
        if args.mode == "metadata_only":
            report["ok"] = True
            report["fourcastnet_proven"] = False  # NEVER true for metadata_only
            report["result"] = "metadata_collected"
            report["note"] = (
                "metadata_only PASS proves artifact access and environment plumbing. "
                "fourcastnet_proven=false because no forward pass was executed."
            )
        else:
            print("[smoke] Running forward pass ...", flush=True)
            forward = run_forward(
                input_tensor=input_tensor,
                checkpoint_path=checkpoint,
                backend=report["backend_probe"],
                runtime_guard=bool(args.max_runtime_guard),
            )
            report["forward"] = forward
            report["ok"] = bool(forward.get("ok"))
            report["fourcastnet_proven"] = bool(forward.get("fourcastnet_proven", False))
            report["result"] = "forward_succeeded" if report["ok"] else "forward_failed"
            if not report["ok"] and not report["backend_probe"].get("ok"):
                report["action_required"] = (
                    "FourCastNet backend code is missing from this environment. "
                    "Install modulus, physicsnemo, or a custom fourcastnet package, then re-run --mode forward."
                )

        _finalize_report(report, local_report_path, output_report_s3_uri, profile, region)
        return 0 if report["ok"] else 1

    except Exception as exc:
        report["ok"] = False
        report["result"] = "script_failed"
        report["fourcastnet_proven"] = False
        report["error_type"] = type(exc).__name__
        report["error_message"] = str(exc)
        report["traceback"] = traceback.format_exc(limit=8)
        _finalize_report(report, local_report_path, output_report_s3_uri, profile, region)
        return 1
    finally:
        if _tmp_cleanup is not None:
            try:
                _tmp_cleanup.cleanup()
            except Exception:
                pass


def _finalize_report(
    report: dict[str, Any],
    local_path: str,
    s3_uri: str,
    profile: str,
    region: str,
) -> None:
    """Write report locally and optionally upload to S3."""
    text = json.dumps(report, indent=2)
    encoded = text.encode("utf-8")

    # Always write locally
    try:
        resolved = write_local_report(local_path, report)
        report["report_written_to_local"] = resolved
    except Exception as exc:
        report["report_local_write_error"] = f"{type(exc).__name__}: {exc}"

    # Upload to S3 if requested
    if s3_uri:
        try:
            upload_s3_bytes(encoded, s3_uri, profile=profile, region=region)
            report["report_uploaded_to_s3"] = s3_uri
            print(f"[smoke] Report uploaded to {s3_uri}", flush=True)
        except Exception as exc:
            report["report_s3_upload_error"] = f"{type(exc).__name__}: {exc}"
            print(f"[smoke] WARNING: S3 upload failed: {exc}", flush=True)

    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
