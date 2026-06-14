"""FourCastNet SageMaker inference entrypoint focused on async S3 IO smoke tests."""

from __future__ import annotations

import io
import json
import os
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


MAX_GUARD_ELEMENTS = 50_000_000


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ok(flag: bool, reason: str | None = None, **extra: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {"ok": bool(flag)}
    if reason:
        payload["reason"] = reason
    payload.update(extra)
    return payload


def _split_s3_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("s3://"):
        raise ValueError(f"Expected s3:// URI, got: {uri}")
    remainder = uri[len("s3://") :]
    bucket, sep, key = remainder.partition("/")
    if not bucket or not sep or not key:
        raise ValueError(f"Invalid S3 URI: {uri}")
    return bucket, key


def _ensure_boto3():
    import boto3  # local import so container without boto3 fails clearly

    return boto3


def _load_npy_from_s3(uri: str) -> np.ndarray:
    boto3 = _ensure_boto3()
    bucket, key = _split_s3_uri(uri)
    s3 = boto3.client("s3")
    response = s3.get_object(Bucket=bucket, Key=key)
    blob = response["Body"].read()
    return np.load(io.BytesIO(blob), allow_pickle=False)


def _write_json_to_s3(uri: str, payload: dict[str, Any]) -> str:
    boto3 = _ensure_boto3()
    bucket, key = _split_s3_uri(uri)
    if uri.endswith("/"):
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        key = f"{key.rstrip('/')}/inference_report_{timestamp}.json"
    body = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
    s3 = boto3.client("s3")
    s3.put_object(Bucket=bucket, Key=key, Body=body, ContentType="application/json")
    return f"s3://{bucket}/{key}"


def _resolve_asset(model_dir: Path, env_key: str, filename: str) -> Path:
    env_value = os.getenv(env_key, "").strip()
    if env_value:
        path = Path(env_value)
        if path.exists():
            return path

    direct = model_dir / filename
    if direct.exists():
        return direct

    nested = model_dir / "code" / filename
    if nested.exists():
        return nested

    return direct


def _probe_backend() -> dict[str, Any]:
    """Readiness signal for native forward inference: is the vendored NVlabs AFNONet
    backend importable (i.e. afnonet.py present and timm/einops installed)?

    This is the Phase-2 stack. The minimal Phase-1 (metadata_only) container ships
    without timm/einops, so this reports ok=false there — which is expected and does
    not affect the metadata_only path.
    """
    attempts: list[dict[str, str]] = []
    try:
        _import_afnonet()
        return _ok(True, backend="afnonet", module="afnonet", symbol="AFNONet", attempts=attempts)
    except Exception as exc:
        attempts.append({"module": "afnonet", "error": f"{type(exc).__name__}: {exc}"})
        return _ok(False, reason="afnonet_backend_unavailable", attempts=attempts)


def _device_report() -> dict[str, Any]:
    if torch is None:
        return {
            "torch_available": False,
            "cuda_available": False,
            "cuda_device_count": 0,
            "selected_device": "cpu",
            "cuda_device_name": "",
        }

    cuda_available = bool(torch.cuda.is_available())
    count = int(torch.cuda.device_count()) if cuda_available else 0
    name = torch.cuda.get_device_name(0) if cuda_available and count > 0 else ""
    return {
        "torch_available": True,
        "cuda_available": cuda_available,
        "cuda_device_count": count,
        "selected_device": "cuda" if cuda_available else "cpu",
        "cuda_device_name": name,
    }


def _tensor_metadata(array: np.ndarray) -> dict[str, Any]:
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


def _load_input_tensor(input_data: dict[str, Any]) -> np.ndarray:
    if input_data.get("tensor") is not None:
        return np.asarray(input_data["tensor"], dtype=np.float32)

    input_s3_uri = str(input_data.get("input_s3_uri", "")).strip()
    if input_s3_uri:
        tensor = _load_npy_from_s3(input_s3_uri)
        return np.asarray(tensor, dtype=np.float32)

    raise ValueError("No tensor provided. Set input_s3_uri or send application/x-npy payload.")


def _read_stats(path: Path) -> dict[str, Any]:
    if not path.exists():
        return _ok(False, reason="missing_file", path=str(path))

    array = np.load(path, allow_pickle=False)
    return _ok(True, path=str(path), **_tensor_metadata(np.asarray(array)))


class _AFNOParams:
    """Minimal params object consumed by the vendored NVlabs AFNONet constructor.

    AFNONet reads patch_size / N_in_channels / N_out_channels / num_blocks off this
    object (overriding the matching constructor kwargs), so they must be present.
    """

    def __init__(self, patch_size: int, n_in: int, n_out: int, num_blocks: int) -> None:
        self.patch_size = patch_size
        self.N_in_channels = n_in
        self.N_out_channels = n_out
        self.num_blocks = num_blocks


def _forward_params() -> dict[str, int]:
    """Architecture params for the NVlabs FourCastNet AFNO backbone (backbone.ckpt).

    Defaults match the published backbone (img 720x1440, patch 8, 20 channels,
    embed_dim 768, depth 12, num_blocks 8). Each is env-overridable for variants.
    """

    def _int_env(key: str, default: int) -> int:
        raw = os.getenv(key, "").strip()
        try:
            return int(raw) if raw else default
        except ValueError:
            return default

    return {
        "img_h": _int_env("FOURCASTNET_IMG_H", 720),
        "img_w": _int_env("FOURCASTNET_IMG_W", 1440),
        "patch_size": _int_env("FOURCASTNET_PATCH_SIZE", 8),
        "n_channels": _int_env("FOURCASTNET_N_CHANNELS", 20),
        "embed_dim": _int_env("FOURCASTNET_EMBED_DIM", 768),
        "depth": _int_env("FOURCASTNET_DEPTH", 12),
        "num_blocks": _int_env("FOURCASTNET_NUM_BLOCKS", 8),
    }


def _import_afnonet():
    """Lazy import so the Phase-1 metadata_only container (no timm/einops) is unaffected."""
    try:
        from .afnonet import AFNONet  # type: ignore
    except ImportError:  # flat layout: SageMaker puts code/ on sys.path
        from afnonet import AFNONet  # type: ignore
    return AFNONet


def _strip_module_prefix(state: dict[str, Any]) -> dict[str, Any]:
    return {(k[7:] if k.startswith("module.") else k): v for k, v in state.items()}


def _load_afnonet(checkpoint_path: str, device: Any, params: dict[str, int]) -> tuple[Any, dict[str, Any]]:
    AFNONet = _import_afnonet()
    afno_params = _AFNOParams(
        patch_size=params["patch_size"],
        n_in=params["n_channels"],
        n_out=params["n_channels"],
        num_blocks=params["num_blocks"],
    )
    net = AFNONet(
        afno_params,
        img_size=(params["img_h"], params["img_w"]),
        patch_size=(params["patch_size"], params["patch_size"]),
        in_chans=params["n_channels"],
        out_chans=params["n_channels"],
        embed_dim=params["embed_dim"],
        depth=params["depth"],
        num_blocks=params["num_blocks"],
    )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        state, checkpoint_key = checkpoint["model_state"], "model_state"
    elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state, checkpoint_key = checkpoint["model_state_dict"], "model_state_dict"
    else:
        state, checkpoint_key = checkpoint, "raw"

    result = net.load_state_dict(_strip_module_prefix(state), strict=False)
    load_info = {
        "checkpoint_key": checkpoint_key,
        "missing_keys": list(getattr(result, "missing_keys", [])),
        "unexpected_keys": list(getattr(result, "unexpected_keys", [])),
    }
    return net, load_info


def _load_norm_stats(path: str, n_channels: int) -> np.ndarray:
    """Load global_means/global_stds as a (1, C, 1, 1) broadcastable array."""
    arr = np.load(path, allow_pickle=False).astype(np.float32)
    if arr.ndim == 4 and arr.shape[1] >= n_channels:
        return arr[:, :n_channels, :1, :1].reshape(1, n_channels, 1, 1)
    flat = arr.reshape(-1)
    if flat.size < n_channels:
        raise ValueError(f"stats file {path} has {flat.size} values, need >= {n_channels}")
    return flat[:n_channels].reshape(1, n_channels, 1, 1)


def _attempt_forward(model: dict[str, Any], tensor: np.ndarray, runtime_guard: bool) -> dict[str, Any]:
    if torch is None:
        return _ok(False, reason="torch_not_available", fourcastnet_proven=False)

    if runtime_guard and tensor.size > MAX_GUARD_ELEMENTS:
        return _ok(
            False,
            reason="runtime_guard_blocked_large_tensor",
            guard_elements_limit=MAX_GUARD_ELEMENTS,
            tensor_elements=int(tensor.size),
            fourcastnet_proven=False,
        )

    params = _forward_params()
    started = time.time()
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        arr = np.asarray(tensor, dtype=np.float32)
        if arr.ndim == 3:
            arr = arr[np.newaxis, ...]
        if arr.ndim != 4:
            return _ok(
                False,
                reason="unexpected_input_rank",
                fourcastnet_proven=False,
                input_shape=[int(d) for d in np.asarray(tensor).shape],
            )
        _, c, h, w = arr.shape
        if c != params["n_channels"] or h != params["img_h"] or w != params["img_w"]:
            return _ok(
                False,
                reason="input_shape_mismatch",
                fourcastnet_proven=False,
                expected=[params["n_channels"], params["img_h"], params["img_w"]],
                got=[int(c), int(h), int(w)],
                hint="Set FOURCASTNET_N_CHANNELS/IMG_H/IMG_W to match the checkpoint, or fix the input tensor.",
            )

        net, load_info = _load_afnonet(model["checkpoint_path"], device, params)
        net.eval()
        net.to(device)

        means = _load_norm_stats(model["global_means_path"], c)
        stds = _load_norm_stats(model["global_stds_path"], c)
        normalized = (arr - means) / stds

        input_tensor = torch.from_numpy(normalized.astype(np.float32)).to(device)
        with torch.no_grad():
            output = net(input_tensor)
        out_np = output.detach().cpu().numpy()
        denormalized = out_np * stds + means

        elapsed = time.time() - started
        # A clean load (no missing keys) is what makes the run trustworthy; a partial
        # load means some weights stayed at init, so we flag it rather than claim proof.
        proven = len(load_info["missing_keys"]) == 0
        return _ok(
            proven,
            reason=None if proven else "checkpoint_partial_load",
            fourcastnet_proven=proven,
            input_shape=[int(d) for d in arr.shape],
            output_shape=[int(d) for d in out_np.shape],
            output_stats=_tensor_metadata(denormalized),
            runtime_seconds=elapsed,
            architecture="AFNONet",
            forward_params=params,
            load_info=load_info,
        )
    except Exception as exc:
        elapsed = time.time() - started
        return _ok(
            False,
            reason="forward_execution_failed",
            fourcastnet_proven=False,
            runtime_seconds=elapsed,
            error_type=type(exc).__name__,
            error_message=str(exc),
            traceback=traceback.format_exc(limit=8),
            forward_params=params,
        )


def model_fn(model_dir: str) -> dict[str, Any]:
    root = Path(model_dir)
    checkpoint = _resolve_asset(root, "FOURCASTNET_CHECKPOINT_PATH", "backbone.ckpt")
    means = _resolve_asset(root, "FOURCASTNET_GLOBAL_MEANS_PATH", "global_means.npy")
    stds = _resolve_asset(root, "FOURCASTNET_GLOBAL_STDS_PATH", "global_stds.npy")

    missing = [str(path) for path in (checkpoint, means, stds) if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing model artifact files: {missing}")

    backend_probe = _probe_backend()
    return {
        "model_dir": str(root),
        "checkpoint_path": str(checkpoint),
        "global_means_path": str(means),
        "global_stds_path": str(stds),
        "backend_probe": backend_probe,
    }


def input_fn(request_body: Any, request_content_type: str = "application/json") -> dict[str, Any]:
    content_type = (request_content_type or "application/json").strip().lower()

    if content_type == "application/x-npy":
        raw = request_body.encode("latin1") if isinstance(request_body, str) else request_body
        tensor = np.load(io.BytesIO(raw), allow_pickle=False)
        return {
            "mode": "forward",
            "max_runtime_guard": True,
            "input_s3_uri": "",
            "output_s3_uri": "",
            "tensor": np.asarray(tensor, dtype=np.float32),
        }

    if content_type not in {"application/json", "application/jsonlines", "application/x-jsonlines"}:
        raise ValueError(f"Unsupported content type: {request_content_type}")

    if isinstance(request_body, (bytes, bytearray)):
        payload = json.loads(request_body.decode("utf-8"))
    elif isinstance(request_body, str):
        payload = json.loads(request_body)
    else:
        payload = request_body

    if not isinstance(payload, dict):
        raise ValueError("JSON payload must be an object.")

    mode = str(payload.get("mode", "metadata_only")).strip().lower()
    if mode not in {"metadata_only", "forward"}:
        raise ValueError("mode must be metadata_only or forward")

    return {
        "mode": mode,
        "max_runtime_guard": bool(payload.get("max_runtime_guard", True)),
        "input_s3_uri": str(payload.get("input_s3_uri", "")).strip(),
        "output_s3_uri": str(payload.get("output_s3_uri", "")).strip(),
    }


def predict_fn(input_data: dict[str, Any], model: dict[str, Any]) -> dict[str, Any]:
    started = time.time()
    report: dict[str, Any] = {
        "ok": False,
        "mode": input_data.get("mode", "metadata_only"),
        "timestamp_utc": _utc_now_iso(),
        "fourcastnet_proven": False,
        "model_assets": {
            "checkpoint_path": model["checkpoint_path"],
            "global_means_path": model["global_means_path"],
            "global_stds_path": model["global_stds_path"],
            "checkpoint_exists": Path(model["checkpoint_path"]).exists(),
            "global_means_exists": Path(model["global_means_path"]).exists(),
            "global_stds_exists": Path(model["global_stds_path"]).exists(),
        },
        "backend_probe": model.get("backend_probe", {}),
        "device": _device_report(),
        "output_s3_uri": str(input_data.get("output_s3_uri", "")).strip(),
        "input_s3_uri": str(input_data.get("input_s3_uri", "")).strip(),
    }

    try:
        means_meta = _read_stats(Path(model["global_means_path"]))
        stds_meta = _read_stats(Path(model["global_stds_path"]))
        report["stats"] = {"global_means": means_meta, "global_stds": stds_meta}

        tensor = _load_input_tensor(input_data)
        report["input_tensor"] = _tensor_metadata(tensor)

        if report["mode"] == "metadata_only":
            report["ok"] = True
            report["fourcastnet_proven"] = False
            report["result"] = "metadata_collected"
        else:
            forward = _attempt_forward(model, tensor, runtime_guard=bool(input_data.get("max_runtime_guard", True)))
            report["forward"] = forward
            report["ok"] = bool(forward.get("ok"))
            report["fourcastnet_proven"] = bool(forward.get("fourcastnet_proven", False))
            report["result"] = "forward_succeeded" if report["ok"] else "forward_failed"
    except Exception as exc:
        report["ok"] = False
        report["result"] = "prediction_failed"
        report["error_type"] = type(exc).__name__
        report["error_message"] = str(exc)
        report["traceback"] = traceback.format_exc(limit=8)

    report["runtime_seconds"] = time.time() - started
    return report


def output_fn(prediction: dict[str, Any], accept: str = "application/json") -> str:
    accept_type = (accept or "application/json").strip().lower()
    if accept_type != "application/json":
        raise ValueError(f"Unsupported accept type: {accept}")

    output_s3_uri = str(prediction.get("output_s3_uri", "")).strip()
    if output_s3_uri:
        try:
            written_uri = _write_json_to_s3(output_s3_uri, prediction)
            prediction["report_s3_uri"] = written_uri
        except Exception as exc:
            prediction["s3_write_error"] = f"{type(exc).__name__}: {exc}"

    return json.dumps(prediction)
