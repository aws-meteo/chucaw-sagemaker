import argparse
import shutil
import tarfile
from pathlib import Path


def _validate_tarball(tar_path: Path) -> None:
    required = {
        "code/inference.py",
        "code/model_config.json",
    }
    with tarfile.open(tar_path, "r:gz") as tf:
        names = set(tf.getnames())
    missing = sorted(required - names)
    if missing:
        raise RuntimeError(f"Malformed tarball. Missing entries: {missing}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Package QuickSight abrupt classifier model.tar.gz")
    parser.add_argument(
        "--output",
        default="artifacts/quicksight_abrupt_model.tar.gz",
        help="Output tar.gz path",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    src_dir = repo_root / "src" / "quicksight_abrupt_classifier"
    inference_file = src_dir / "inference.py"
    config_file = src_dir / "model_config.json"
    out_path = repo_root / args.output
    build_dir = repo_root / ".build_quicksight_abrupt"
    code_dir = build_dir / "code"

    for required in (inference_file, config_file):
        if not required.exists():
            raise FileNotFoundError(f"Missing required file: {required}")

    if build_dir.exists():
        shutil.rmtree(build_dir)
    code_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(inference_file, code_dir / "inference.py")
    shutil.copy2(config_file, code_dir / "model_config.json")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()

    with tarfile.open(out_path, "w:gz") as tf:
        tf.add(code_dir, arcname="code")

    _validate_tarball(out_path)
    shutil.rmtree(build_dir, ignore_errors=True)
    print(f"Packaged tarball: {out_path}")


if __name__ == "__main__":
    main()
