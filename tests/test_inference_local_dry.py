import json
import io
import numpy as np
from pathlib import Path
from src.fourcastnet.serving.inference import input_fn, predict_fn, output_fn

def test_inference_pipeline_dry_run():
    # 1. Synthesize input payload
    tensor = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    buf = io.BytesIO()
    np.save(buf, tensor, allow_pickle=False)
    raw_payload = buf.getvalue()

    # 2. Test input_fn with x-npy
    parsed = input_fn(raw_payload, request_content_type="application/x-npy")
    assert parsed["mode"] == "forward"
    assert "tensor" in parsed
    assert np.array_equal(parsed["tensor"], tensor)

    # 3. Test predict_fn
    mock_model = {
        "checkpoint_path": "dummy.ckpt",
        "global_means_path": "dummy_means.npy",
        "global_stds_path": "dummy_stds.npy",
        "backend_probe": {"ok": False}  # Force fail the forward but succeed parsing
    }
    
    report = predict_fn(parsed, mock_model)
    assert report["ok"] is False
    assert report["mode"] == "forward"
    assert "backend_unavailable" in str(report.get("forward", {}).get("reason", ""))
    
    # 4. Test output_fn
    output_str = output_fn(report, accept="application/json")
    output_dict = json.loads(output_str)
    assert output_dict["ok"] is False
    assert output_dict["mode"] == "forward"

def test_inference_pipeline_json_dry_run(monkeypatch):
    # Mock _load_npy_from_s3 for json payload
    def mock_load(uri):
        return np.array([1.0, 2.0], dtype=np.float32)
    
    import src.fourcastnet.serving.inference as inf
    monkeypatch.setattr(inf, "_load_npy_from_s3", mock_load)

    payload_dict = {
        "mode": "metadata_only",
        "input_s3_uri": "s3://mock/uri",
        "output_s3_uri": ""
    }
    raw_payload = json.dumps(payload_dict).encode("utf-8")

    parsed = input_fn(raw_payload, request_content_type="application/json")
    
    mock_model = {
        "checkpoint_path": "dummy.ckpt",
        "global_means_path": "dummy_means.npy",
        "global_stds_path": "dummy_stds.npy",
        "backend_probe": {"ok": True}
    }
    
    report = predict_fn(parsed, mock_model)
    # metadata_only forces ok=True without running torch
    assert report["ok"] is True
    assert report["result"] == "metadata_collected"
    
    output_str = output_fn(report, accept="application/json")
    output_dict = json.loads(output_str)
    assert output_dict["ok"] is True
