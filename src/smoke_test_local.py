#!/usr/bin/env python3
import argparse
import importlib.util
import json
import sys
import tarfile
import tempfile
from pathlib import Path


REQUIRED_KEYS = {"t2m", "lat_grid", "lon_grid", "units", "source"}


def parse_args():
    parser = argparse.ArgumentParser(description="Local smoke test for inference contract")
    parser.add_argument(
        "--artifact-local",
        default="",
        help=(
            "Optional model artifact tar.gz path. If set, smoke test loads model from "
            "extracted tarball."
        ),
    )
    parser.add_argument(
        "--source-artifact-local",
        default="",
        help=(
            "Optional source artifact tar.gz path. Required for separate-source serving bundle "
            "if model artifact does not contain code/inference.py."
        ),
    )
    return parser.parse_args()


def load_inference_module_from_file(module_path: Path):
    spec = importlib.util.spec_from_file_location("hosting_inference_module", str(module_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to import inference module from: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_smoke(model_dir: Path, inference_py: Path):
    module = load_inference_module_from_file(inference_py)
    input_fn = module.input_fn
    model_fn = module.model_fn
    output_fn = module.output_fn
    predict_fn = module.predict_fn

    request_json = json.dumps({"lat": -33.5, "lon": -70.6})
    model = model_fn(str(model_dir))
    parsed_input = input_fn(request_json, content_type="application/json")
    prediction = predict_fn(parsed_input, model)
    response_body, content_type = output_fn(prediction, accept="application/json")

    if content_type != "application/json":
        raise ValueError(f"Unexpected content_type: {content_type}")

    response = json.loads(response_body)
    missing = sorted(REQUIRED_KEYS - set(response.keys()))
    if missing:
        raise ValueError(f"Local smoke response missing keys: {', '.join(missing)}")

    if not isinstance(response["t2m"], (float, int)):
        raise TypeError("response['t2m'] must be numeric")
    if not isinstance(response["lat_grid"], (float, int)):
        raise TypeError("response['lat_grid'] must be numeric")
    if not isinstance(response["lon_grid"], (float, int)):
        raise TypeError("response['lon_grid'] must be numeric")
    if response["units"] != "K":
        raise ValueError("response['units'] must be 'K'")
    if response["source"] != "ECMWF SCDA":
        raise ValueError("response['source'] must be 'ECMWF SCDA'")

    return response


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]

    if not args.artifact_local:
        model_path = repo_root / "model.joblib"
        if not model_path.exists():
            raise FileNotFoundError(f"model.joblib not found: {model_path}")
        response = run_smoke(
            model_dir=repo_root,
            inference_py=repo_root / "inference" / "inference.py",
        )
        print(f"Local smoke source: {model_path}")
    else:
        artifact_path = Path(args.artifact_local)
        if not artifact_path.is_absolute():
            artifact_path = (Path.cwd() / artifact_path).resolve()
        if not artifact_path.exists():
            raise FileNotFoundError(f"Hosting artifact not found: {artifact_path}")

        with tempfile.TemporaryDirectory(prefix="smoke-hosting-artifact-") as temp_dir_str:
            temp_dir = Path(temp_dir_str)
            with tarfile.open(artifact_path, "r:gz") as tar:
                tar.extractall(path=temp_dir)

            extracted_model = temp_dir / "model.joblib"
            extracted_inference = temp_dir / "code" / "inference.py"
            if not extracted_model.exists():
                raise FileNotFoundError(f"Missing model.joblib in artifact: {artifact_path}")

            if not extracted_inference.exists():
                source_artifact_path = args.source_artifact_local.strip()
                if not source_artifact_path:
                    sibling_source = artifact_path.parent / "source.tar.gz"
                    if sibling_source.exists():
                        source_artifact_path = str(sibling_source)

                if not source_artifact_path:
                    raise FileNotFoundError(
                        "Missing code/inference.py in model artifact and no source artifact provided. "
                        "Pass --source-artifact-local <path to source.tar.gz>."
                    )

                source_path = Path(source_artifact_path)
                if not source_path.is_absolute():
                    source_path = (Path.cwd() / source_path).resolve()
                if not source_path.exists():
                    raise FileNotFoundError(f"Source artifact not found: {source_path}")

                source_dir = temp_dir / "source"
                source_dir.mkdir(parents=True, exist_ok=True)
                with tarfile.open(source_path, "r:gz") as source_tar:
                    source_tar.extractall(path=source_dir)

                extracted_inference = source_dir / "inference.py"
                if not extracted_inference.exists():
                    raise FileNotFoundError(
                        f"Missing inference.py in source artifact: {source_path}"
                    )

            response = run_smoke(model_dir=temp_dir, inference_py=extracted_inference)
            print(f"Local smoke source: {artifact_path}")

    print("Local smoke test passed.")
    print(json.dumps(response, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
