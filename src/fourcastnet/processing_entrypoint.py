#!/usr/bin/env python3
"""Processing-container entrypoint for FCN smoke checks."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FCN smoke checks inside SageMaker Processing")
    parser.add_argument("--input-tensor", required=True, help="Path to input tensor (.npy)")
    parser.add_argument("--global-means", required=True, help="Path to global_means.npy")
    parser.add_argument("--global-stds", required=True, help="Path to global_stds.npy")
    parser.add_argument("--checkpoint", required=True, help="Path to backbone checkpoint")
    parser.add_argument("--output-report", required=True, help="Output JSON report path")
    parser.add_argument(
        "--local-repo",
        default="",
        help="Optional path to local NVlabs FourCastNet repo for real forward-pass attempt",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Requested device for forward pass",
    )
    parser.add_argument(
        "--require-forward",
        action="store_true",
        help="Fail when forward pass cannot be attempted",
    )
    return parser.parse_args()


def resolve_device(choice: str) -> str:
    if choice == "cpu":
        return "cpu"
    if choice == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        return "cuda"
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_stats(path: Path) -> np.ndarray:
    arr = np.asarray(np.load(path), dtype=np.float32)
    if arr.ndim == 1:
        if arr.shape[0] < 20:
            raise ValueError(f"Stats has fewer than 20 channels: {arr.shape}")
        arr = arr[:20].reshape(1, 20, 1, 1)
    elif arr.ndim == 4:
        if arr.shape[1] < 20:
            raise ValueError(f"Stats has fewer than 20 channels: {arr.shape}")
        arr = arr[:, :20, :, :]
    else:
        raise ValueError(f"Unsupported stats shape: {arr.shape}")
    return arr


def extract_state_dict(ckpt_payload: Any) -> dict[str, Any]:
    if isinstance(ckpt_payload, dict):
        for key in ("model_state", "state_dict", "module"):
            value = ckpt_payload.get(key)
            if isinstance(value, dict):
                return value
        if all(isinstance(k, str) for k in ckpt_payload.keys()):
            return ckpt_payload
    raise RuntimeError("Could not extract state_dict from checkpoint payload")


def try_forward_nv(repo_path: Path, x_norm: torch.Tensor, checkpoint_path: Path) -> tuple[bool, dict[str, Any]]:
    sys.path.insert(0, str(repo_path))
    sys.path.insert(0, str(repo_path / "networks"))

    module = None
    import_errors: list[str] = []
    for module_name in ("networks.afnonet", "afnonet"):
        try:
            module = importlib.import_module(module_name)
            break
        except Exception as exc:
            import_errors.append(f"{module_name}: {exc}")
    if module is None or not hasattr(module, "AFNONet"):
        return False, {"reason": "AFNONet import failed", "errors": import_errors}

    model_class = getattr(module, "AFNONet")
    model = None
    init_errors: list[str] = []
    candidate_kwargs = [
        {"img_size": (x_norm.shape[2], x_norm.shape[3]), "in_chans": 20, "out_chans": 20},
        {"in_chans": 20, "out_chans": 20},
        {},
    ]
    for kwargs in candidate_kwargs:
        try:
            model = model_class(**kwargs)
            break
        except Exception as exc:
            init_errors.append(f"{kwargs}: {exc}")
    if model is None:
        return False, {"reason": "AFNONet init failed", "errors": init_errors}

    checkpoint_payload = torch.load(checkpoint_path, map_location="cpu")
    state_dict = extract_state_dict(checkpoint_payload)
    load_result = model.load_state_dict(state_dict, strict=False)
    model = model.to(x_norm.device).eval()
    with torch.no_grad():
        y_norm = model(x_norm)

    y_np = y_norm.detach().cpu().numpy()
    return True, {
        "output_shape": list(y_np.shape),
        "output_finite": bool(np.isfinite(y_np).all()),
        "missing_keys_count": len(getattr(load_result, "missing_keys", [])),
        "unexpected_keys_count": len(getattr(load_result, "unexpected_keys", [])),
        "output_min": float(np.min(y_np)),
        "output_max": float(np.max(y_np)),
        "output_mean": float(np.mean(y_np)),
    }


def main() -> int:
    args = parse_args()
    started = time.time()

    tensor_path = Path(args.input_tensor)
    means_path = Path(args.global_means)
    stds_path = Path(args.global_stds)
    checkpoint_path = Path(args.checkpoint)
    output_report_path = Path(args.output_report)
    local_repo = Path(args.local_repo) if args.local_repo else None

    report: dict[str, Any] = {
        "ok": False,
        "checkpoint_load_ok": False,
        "forward_attempted": False,
        "forward_success": False,
        "output_finite": None,
        "duration_seconds": None,
        "device": None,
        "failure_reason": None,
    }

    try:
        for required_file in (tensor_path, means_path, stds_path, checkpoint_path):
            if not required_file.exists():
                raise FileNotFoundError(str(required_file))

        tensor = np.asarray(np.load(tensor_path), dtype=np.float32)
        if tuple(tensor.shape) != (1, 20, 720, 1440):
            raise ValueError(f"Unexpected tensor shape: {tensor.shape}")
        if not np.isfinite(tensor).all():
            raise ValueError("Input tensor contains non-finite values")

        means = load_stats(means_path)
        stds = load_stats(stds_path)
        x_norm_np = (tensor - means) / stds
        if not np.isfinite(x_norm_np).all():
            raise ValueError("Normalization produced non-finite values")

        checkpoint_payload = torch.load(checkpoint_path, map_location="cpu")
        _ = extract_state_dict(checkpoint_payload)
        report["checkpoint_load_ok"] = True

        device = resolve_device(args.device)
        report["device"] = device

        if local_repo and local_repo.exists():
            report["forward_attempted"] = True
            x_norm = torch.from_numpy(x_norm_np).to(device)
            success, detail = try_forward_nv(local_repo, x_norm, checkpoint_path)
            report["forward_success"] = success
            report["forward_detail"] = detail
            if success:
                report["output_finite"] = detail.get("output_finite")
        else:
            report["forward_detail"] = {
                "reason": "No local FourCastNet repo provided; checkpoint-only validation performed."
            }

        if args.require_forward and not report["forward_success"]:
            raise RuntimeError("Forward pass required but not successful")

        report["ok"] = bool(report["checkpoint_load_ok"] and (not args.require_forward or report["forward_success"]))
    except Exception as exc:
        report["ok"] = False
        report["failure_reason"] = str(exc)

    report["duration_seconds"] = time.time() - started
    output_report_path.parent.mkdir(parents=True, exist_ok=True)
    output_report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Report: {output_report_path}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

