"""Tests for channel-wise stats helper, normalization warning, and optional tensor S3 write.

All tests are local-only and use synthetic numpy tensors.  No real checkpoint,
no real AWS credentials.
"""

from __future__ import annotations

import json
import io
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.fourcastnet.serving.inference import (
    CHANNEL_NAMES,
    _channel_wise_stats,
    _load_norm_stats,
    _tensor_metadata,
    _write_npy_to_s3,
    input_fn,
    predict_fn,
)


# ---------------------------------------------------------------------------
# 1. channel-wise stats on a synthetic [1, 2, 2, 2] tensor
# ---------------------------------------------------------------------------

class TestChannelWiseStats:
    def test_basic_shape_and_values(self):
        """Stats on [1,2,2,2] synthetic tensor produce correct per-channel records."""
        arr = np.array(
            [[[[1.0, 2.0], [3.0, 4.0]], [[10.0, 20.0], [30.0, 40.0]]]],
            dtype=np.float32,
        )
        assert arr.shape == (1, 2, 2, 2)

        names = ["alpha", "beta"]
        stats = _channel_wise_stats(arr, names)

        assert len(stats) == 2

        # Channel 0: values 1,2,3,4
        ch0 = stats[0]
        assert ch0["index"] == 0
        assert ch0["name"] == "alpha"
        assert ch0["dtype"] == "float32"
        assert ch0["shape"] == [2, 2]
        assert ch0["finite"] is True
        assert ch0["nan_count"] == 0
        assert ch0["min"] == pytest.approx(1.0)
        assert ch0["max"] == pytest.approx(4.0)
        assert ch0["mean"] == pytest.approx(2.5)

        # Channel 1: values 10,20,30,40
        ch1 = stats[1]
        assert ch1["index"] == 1
        assert ch1["name"] == "beta"
        assert ch1["min"] == pytest.approx(10.0)
        assert ch1["max"] == pytest.approx(40.0)
        assert ch1["mean"] == pytest.approx(25.0)

    def test_nan_handling(self):
        """NaN values are counted and finite is False."""
        arr = np.array(
            [[[[np.nan, 1.0], [2.0, 3.0]], [[4.0, 5.0], [6.0, 7.0]]]],
            dtype=np.float32,
        )
        stats = _channel_wise_stats(arr, ["a", "b"])
        assert stats[0]["nan_count"] == 1
        assert stats[0]["finite"] is False
        assert stats[1]["nan_count"] == 0
        assert stats[1]["finite"] is True

    def test_3d_input_accepted(self):
        """[C, H, W] (no batch dim) is also accepted."""
        arr = np.ones((3, 4, 4), dtype=np.float32)
        stats = _channel_wise_stats(arr, ["x", "y", "z"])
        assert len(stats) == 3
        assert all(s["shape"] == [4, 4] for s in stats)

    def test_missing_names_fallback(self):
        """When fewer names than channels, extras get 'unknown_<i>'."""
        arr = np.ones((1, 3, 2, 2), dtype=np.float32)
        stats = _channel_wise_stats(arr, ["only_one"])
        assert stats[0]["name"] == "only_one"
        assert stats[1]["name"] == "unknown_1"
        assert stats[2]["name"] == "unknown_2"


# ---------------------------------------------------------------------------
# 2. channel names preserved in order
# ---------------------------------------------------------------------------

class TestChannelNames:
    def test_canonical_order(self):
        expected = [
            "u10", "v10", "t2m", "sp", "msl",
            "t850", "u1000", "v1000", "z1000",
            "u850", "v850", "z850",
            "u500", "v500", "z500", "t500",
            "z50", "r500", "r850", "tcwv",
        ]
        assert CHANNEL_NAMES == expected

    def test_channel_count(self):
        assert len(CHANNEL_NAMES) == 20

    def test_stats_preserve_order(self):
        """When channel_names == CHANNEL_NAMES, output records preserve the exact order."""
        arr = np.random.randn(1, 20, 2, 2).astype(np.float32)
        stats = _channel_wise_stats(arr, CHANNEL_NAMES)
        for i, rec in enumerate(stats):
            assert rec["index"] == i
            assert rec["name"] == CHANNEL_NAMES[i]


# ---------------------------------------------------------------------------
# 3. normalization extra-channel warning
# ---------------------------------------------------------------------------

class TestNormalizationExtraChannelWarning:
    def test_21_channel_stats_with_20_model_channels(self, tmp_path):
        """When stats file has 21 channels and model uses 20, info warns."""
        stats_21 = np.random.randn(21).astype(np.float32)
        path = tmp_path / "global_means.npy"
        np.save(path, stats_21, allow_pickle=False)

        arr, info = _load_norm_stats(str(path), 20)
        assert arr.shape == (1, 20, 1, 1)
        assert info["normalization_stats_channels"] == 21
        assert info["model_channels"] == 20
        assert info["normalization_stats_extra_channels_ignored"] is True

    def test_20_channel_stats_with_20_model_no_warning(self, tmp_path):
        """When stats have exactly 20 channels, no extra-channel warning."""
        stats_20 = np.random.randn(20).astype(np.float32)
        path = tmp_path / "global_means.npy"
        np.save(path, stats_20, allow_pickle=False)

        arr, info = _load_norm_stats(str(path), 20)
        assert arr.shape == (1, 20, 1, 1)
        assert info["normalization_stats_channels"] == 20
        assert info["model_channels"] == 20
        assert "normalization_stats_extra_channels_ignored" not in info

    def test_4d_stats_21_channels_ignored(self, tmp_path):
        """4-D stats array [1,21,1,1] also triggers the warning."""
        stats_4d = np.random.randn(1, 21, 1, 1).astype(np.float32)
        path = tmp_path / "global_means_4d.npy"
        np.save(path, stats_4d, allow_pickle=False)

        arr, info = _load_norm_stats(str(path), 20)
        assert arr.shape == (1, 20, 1, 1)
        assert info["normalization_stats_extra_channels_ignored"] is True


# ---------------------------------------------------------------------------
# 4. output tensor writing is opt-in
# ---------------------------------------------------------------------------

class TestOutputTensorWriteOptIn:
    def test_default_false_no_write(self):
        """Default write_output_tensor=false means no S3 write is attempted."""
        payload = json.dumps({
            "mode": "forward",
            "input_s3_uri": "s3://bucket/input.npy",
            "output_s3_uri": "s3://bucket/output/",
        })
        parsed = input_fn(payload, "application/json")
        assert parsed["write_output_tensor"] is False
        assert parsed["output_tensor_s3_uri"] == ""

    def test_true_parsed(self):
        """write_output_tensor=true is correctly parsed from JSON manifest."""
        payload = json.dumps({
            "mode": "forward",
            "input_s3_uri": "s3://bucket/input.npy",
            "output_s3_uri": "s3://bucket/output/",
            "write_output_tensor": True,
            "output_tensor_s3_uri": "s3://bucket/output/forecast_tensor.npy",
        })
        parsed = input_fn(payload, "application/json")
        assert parsed["write_output_tensor"] is True
        assert parsed["output_tensor_s3_uri"] == "s3://bucket/output/forecast_tensor.npy"

    def test_predict_fn_no_tensor_write_when_default(self, monkeypatch):
        """predict_fn does NOT call _write_npy_to_s3 when write_output_tensor is false."""
        import src.fourcastnet.serving.inference as inf

        mock_write = MagicMock()
        monkeypatch.setattr(inf, "_write_npy_to_s3", mock_write)

        # Mock _load_npy_from_s3 so predict_fn can load the "input tensor"
        monkeypatch.setattr(inf, "_load_npy_from_s3", lambda uri: np.ones((1, 20, 720, 1440), dtype=np.float32))

        input_data = {
            "mode": "forward",
            "input_s3_uri": "s3://bucket/input.npy",
            "output_s3_uri": "s3://bucket/output/",
            "max_runtime_guard": True,
            "write_output_tensor": False,
            "output_tensor_s3_uri": "",
        }
        mock_model = {
            "checkpoint_path": "dummy.ckpt",
            "global_means_path": "dummy_means.npy",
            "global_stds_path": "dummy_stds.npy",
            "backend_probe": {"ok": False},
        }

        report = predict_fn(input_data, mock_model)
        # Forward will fail (no real checkpoint), but _write_npy_to_s3 must NOT be called
        mock_write.assert_not_called()

    def test_write_npy_to_s3_called_with_uri(self, monkeypatch):
        """When write_output_tensor=true AND forward succeeds, _write_npy_to_s3 is called."""
        import src.fourcastnet.serving.inference as inf

        # Create a fake successful forward that returns the denormalized tensor
        fake_denormalized = np.ones((1, 20, 720, 1440), dtype=np.float32)

        def fake_attempt_forward(model, tensor, runtime_guard):
            return {
                "ok": True,
                "fourcastnet_proven": True,
                "input_shape": [1, 20, 720, 1440],
                "output_shape": [1, 20, 720, 1440],
                "output_stats": {},
                "_denormalized": fake_denormalized,
            }

        monkeypatch.setattr(inf, "_attempt_forward", fake_attempt_forward)
        monkeypatch.setattr(inf, "_load_npy_from_s3", lambda uri: np.ones((1, 20, 720, 1440), dtype=np.float32))

        mock_write = MagicMock(return_value="s3://bucket/output/forecast_tensor.npy")
        monkeypatch.setattr(inf, "_write_npy_to_s3", mock_write)

        # Fake _read_stats to avoid needing real files
        monkeypatch.setattr(inf, "_read_stats", lambda path: {"ok": False, "reason": "mock"})

        input_data = {
            "mode": "forward",
            "input_s3_uri": "s3://bucket/input.npy",
            "output_s3_uri": "s3://bucket/output/",
            "max_runtime_guard": True,
            "write_output_tensor": True,
            "output_tensor_s3_uri": "s3://bucket/output/forecast_tensor.npy",
        }
        mock_model = {
            "checkpoint_path": "dummy.ckpt",
            "global_means_path": "dummy_means.npy",
            "global_stds_path": "dummy_stds.npy",
            "backend_probe": {"ok": True},
        }

        report = predict_fn(input_data, mock_model)
        mock_write.assert_called_once_with(
            "s3://bucket/output/forecast_tensor.npy", fake_denormalized
        )
        assert report["output_tensor_written"] is True
        assert report["output_tensor_s3_uri"] == "s3://bucket/output/forecast_tensor.npy"
        assert report["output_tensor_shape"] == [1, 20, 720, 1440]

    def test_fallback_uri_from_output_s3_uri(self, monkeypatch):
        """When output_tensor_s3_uri is empty, falls back to output_s3_uri/forecast_tensor.npy."""
        import src.fourcastnet.serving.inference as inf

        fake_denormalized = np.ones((1, 2, 3, 3), dtype=np.float32)

        def fake_attempt_forward(model, tensor, runtime_guard):
            return {
                "ok": True,
                "fourcastnet_proven": True,
                "_denormalized": fake_denormalized,
            }

        monkeypatch.setattr(inf, "_attempt_forward", fake_attempt_forward)
        monkeypatch.setattr(inf, "_load_npy_from_s3", lambda uri: np.ones((1, 2, 3, 3), dtype=np.float32))
        monkeypatch.setattr(inf, "_read_stats", lambda path: {"ok": False, "reason": "mock"})

        captured_uri = {}

        def mock_write(uri, arr):
            captured_uri["uri"] = uri
            return uri

        monkeypatch.setattr(inf, "_write_npy_to_s3", mock_write)

        input_data = {
            "mode": "forward",
            "input_s3_uri": "s3://bucket/input.npy",
            "output_s3_uri": "s3://bucket/output/prefix/",
            "max_runtime_guard": True,
            "write_output_tensor": True,
            "output_tensor_s3_uri": "",  # empty -> fallback
        }
        mock_model = {
            "checkpoint_path": "dummy.ckpt",
            "global_means_path": "d.npy",
            "global_stds_path": "d.npy",
            "backend_probe": {"ok": True},
        }

        predict_fn(input_data, mock_model)
        assert captured_uri["uri"] == "s3://bucket/output/prefix/forecast_tensor.npy"


# ---------------------------------------------------------------------------
# 4b. tensor-write failure semantics (write_output_tensor=true)
# ---------------------------------------------------------------------------

def _forward_ok_input(write_tensor=True, output_tensor_s3_uri="s3://bucket/out/t.npy", output_s3_uri="s3://bucket/out/"):
    return {
        "mode": "forward",
        "input_s3_uri": "s3://bucket/input.npy",
        "output_s3_uri": output_s3_uri,
        "max_runtime_guard": True,
        "write_output_tensor": write_tensor,
        "output_tensor_s3_uri": output_tensor_s3_uri,
    }


_MOCK_MODEL = {
    "checkpoint_path": "dummy.ckpt",
    "global_means_path": "d.npy",
    "global_stds_path": "d.npy",
    "backend_probe": {"ok": True},
}


def _patch_successful_forward(monkeypatch, inf):
    fake = np.ones((1, 20, 8, 8), dtype=np.float32)

    def fake_attempt_forward(model, tensor, runtime_guard):
        return {"ok": True, "fourcastnet_proven": True, "_denormalized": fake}

    monkeypatch.setattr(inf, "_attempt_forward", fake_attempt_forward)
    monkeypatch.setattr(inf, "_load_npy_from_s3", lambda uri: fake)
    monkeypatch.setattr(inf, "_read_stats", lambda path: {"ok": False, "reason": "mock"})
    return fake


class TestTensorWriteFailureSemantics:
    def test_write_disabled_keeps_forward_ok(self, monkeypatch):
        """write_output_tensor=false: forward success yields top-level ok=true."""
        import src.fourcastnet.serving.inference as inf

        _patch_successful_forward(monkeypatch, inf)
        monkeypatch.setattr(inf, "_write_npy_to_s3", MagicMock())

        report = predict_fn(_forward_ok_input(write_tensor=False), _MOCK_MODEL)
        assert report["ok"] is True
        assert report["result"] == "forward_succeeded"
        assert report["output_tensor_written"] is False

    def test_write_success_full_contract(self, monkeypatch):
        """Forward + tensor write both succeed: full success contract."""
        import src.fourcastnet.serving.inference as inf

        _patch_successful_forward(monkeypatch, inf)
        monkeypatch.setattr(inf, "_write_npy_to_s3", MagicMock(return_value="s3://bucket/out/t.npy"))

        report = predict_fn(_forward_ok_input(), _MOCK_MODEL)
        assert report["ok"] is True
        assert report["result"] == "forward_succeeded"
        assert report["output_tensor_written"] is True
        assert report["output_tensor_s3_uri"] == "s3://bucket/out/t.npy"
        assert report["output_tensor_shape"] == [1, 20, 8, 8]
        assert report["output_tensor_dtype"] == "float32"

    def test_write_failure_demotes_top_level_ok(self, monkeypatch):
        """Forward succeeds but tensor write raises: top-level ok=false, forward stays ok."""
        import src.fourcastnet.serving.inference as inf

        _patch_successful_forward(monkeypatch, inf)

        def boom(uri, arr):
            raise RuntimeError("s3 down")

        monkeypatch.setattr(inf, "_write_npy_to_s3", boom)

        report = predict_fn(_forward_ok_input(), _MOCK_MODEL)
        assert report["forward"]["ok"] is True
        assert report["forward"]["fourcastnet_proven"] is True
        assert report["output_tensor_written"] is False
        assert "output_tensor_write_error" in report
        assert "RuntimeError" in report["output_tensor_write_error"]
        assert report["ok"] is False
        assert report["result"] == "forward_succeeded_tensor_write_failed"

    def test_no_uri_and_no_fallback_is_write_failure(self, monkeypatch):
        """write_output_tensor=true but no tensor URI and no output_s3_uri fallback -> write failure."""
        import src.fourcastnet.serving.inference as inf

        _patch_successful_forward(monkeypatch, inf)
        mock_write = MagicMock()
        monkeypatch.setattr(inf, "_write_npy_to_s3", mock_write)

        report = predict_fn(
            _forward_ok_input(output_tensor_s3_uri="", output_s3_uri=""),
            _MOCK_MODEL,
        )
        mock_write.assert_not_called()
        assert report["output_tensor_written"] is False
        assert "output_tensor_write_error" in report
        assert report["ok"] is False
        assert report["result"] == "forward_succeeded_tensor_write_failed"


# ---------------------------------------------------------------------------
# 5. _write_npy_to_s3 unit test (mock boto3)
# ---------------------------------------------------------------------------

class TestWriteNpyToS3:
    def test_writes_correct_npy(self, monkeypatch):
        """_write_npy_to_s3 serializes the numpy array and uploads via put_object."""
        import src.fourcastnet.serving.inference as inf

        captured = {}
        mock_s3 = MagicMock()

        def fake_put(**kwargs):
            captured.update(kwargs)

        mock_s3.put_object = fake_put

        mock_boto3 = MagicMock()
        mock_boto3.client.return_value = mock_s3
        monkeypatch.setattr(inf, "_ensure_boto3", lambda: mock_boto3)

        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = _write_npy_to_s3("s3://mybucket/path/tensor.npy", arr)

        assert result == "s3://mybucket/path/tensor.npy"
        assert captured["Bucket"] == "mybucket"
        assert captured["Key"] == "path/tensor.npy"
        # Verify the body is a valid .npy
        recovered = np.load(io.BytesIO(captured["Body"]), allow_pickle=False)
        np.testing.assert_array_equal(recovered, arr)
