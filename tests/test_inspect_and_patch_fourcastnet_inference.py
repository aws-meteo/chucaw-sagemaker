import importlib.util
import sys
import tarfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "scripts" / "inspect_and_patch_fourcastnet_inference.py"

spec = importlib.util.spec_from_file_location("inspect_and_patch_fourcastnet_inference", MODULE_PATH)
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)


CUDA_HANDLER = '''
import torch

def model_fn(model_dir):
    device = torch.device("cuda")
    ckpt = torch.load(model_dir + "/backbone.ckpt")
    model = Net()
    model.load_state_dict(ckpt)
    model = model.cuda()
    return model.to("cuda")

def predict_fn(x, model):
    return model(x.cuda())
'''

CPU_SAFE_HANDLER = '''
import torch

def model_fn(model_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(model_dir + "/backbone.ckpt", map_location="cpu")
    model = Net()
    model.load_state_dict(ckpt)
    return model.to(device)
'''


def test_scan_flags_hardcoded_cuda():
    scan = mod.scan_source(CUDA_HANDLER)
    assert not scan.cpu_safe
    assert len(scan.blockers) >= 3  # device('cuda'), .cuda(), .to('cuda'), x.cuda()
    # torch.load without map_location should be a warning.
    assert any("map_location" in f.label for f in scan.warnings)


def test_scan_marks_cpu_safe_handler():
    scan = mod.scan_source(CPU_SAFE_HANDLER)
    assert scan.cpu_safe
    assert len(scan.blockers) == 0
    assert any(f.severity == "SAFE" for f in scan.findings)


def test_patch_removes_all_blockers():
    patched, changes = mod.patch_source(CUDA_HANDLER)
    assert changes, "expected at least one rewrite"
    rescan = mod.scan_source(patched)
    assert rescan.cpu_safe, f"blockers remain: {[f.label for f in rescan.blockers]}"
    assert 'torch.device("cpu")' in patched
    assert ".cpu()" in patched
    assert "map_location" in patched


def test_patch_is_noop_for_cpu_safe_handler():
    patched, changes = mod.patch_source(CPU_SAFE_HANDLER)
    assert changes == []
    assert mod.scan_source(patched).cpu_safe


def test_comment_lines_do_not_trip_blockers():
    src = "# we used to call model.cuda() here\nx = 1\n"
    scan = mod.scan_source(src)
    assert scan.cpu_safe
    assert scan.blockers == []


def _write_model_tar(path: Path, inference_src: str, with_code_dir: bool = True):
    with tarfile.open(path, "w:gz") as tar:
        # a non-code file to ensure copy-through works
        joblib_info = tarfile.TarInfo("backbone.ckpt")
        joblib_bytes = b"\x00\x01\x02"
        joblib_info.size = len(joblib_bytes)
        import io

        tar.addfile(joblib_info, io.BytesIO(joblib_bytes))

        name = "code/inference.py" if with_code_dir else "inference.py"
        inf_info = tarfile.TarInfo(name)
        data = inference_src.encode("utf-8")
        inf_info.size = len(data)
        tar.addfile(inf_info, io.BytesIO(data))


def test_emit_patched_tar_roundtrip(tmp_path):
    src = tmp_path / "model.tar.gz"
    dst = tmp_path / "model-cpu.tar.gz"
    _write_model_tar(src, CUDA_HANDLER)

    with tarfile.open(src, "r:gz") as tar:
        member = mod._find_inference_member(tar)
        assert member.name == "code/inference.py"
        patched, _ = mod.patch_source(tar.extractfile(member).read().decode("utf-8"))

    mod.emit_patched_tar(str(src), str(dst), "code/inference.py", patched.encode("utf-8"))

    with tarfile.open(dst, "r:gz") as tar:
        names = set(tar.getnames())
        assert "backbone.ckpt" in names  # copied through untouched
        assert "code/inference.py" in names
        out = tar.extractfile("code/inference.py").read().decode("utf-8")
    assert mod.scan_source(out).cpu_safe
    # original archive must be unchanged
    with tarfile.open(src, "r:gz") as tar:
        orig = tar.extractfile("code/inference.py").read().decode("utf-8")
    assert not mod.scan_source(orig).cpu_safe
