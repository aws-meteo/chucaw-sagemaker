import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.quicksight_abrupt_classifier.inference import input_fn, model_fn, output_fn, predict_fn


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local CSV smoke test for QuickSight abrupt classifier.")
    parser.add_argument("--input", required=True, help="Path to input CSV with rows: lat,lon,t2m")
    parser.add_argument("--output", required=True, help="Path to output CSV")
    parser.add_argument(
        "--model-dir",
        default=str(Path(__file__).resolve().parents[1] / "src" / "quicksight_abrupt_classifier"),
        help="Directory containing model_config.json",
    )
    parser.add_argument("--content-type", default="text/csv", help="Input content type")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    model_dir = Path(args.model_dir)

    raw = input_path.read_text(encoding="utf-8")
    model = model_fn(str(model_dir))
    parsed = input_fn(raw, content_type=args.content_type)
    predicted = predict_fn(parsed, model)
    out, _ = output_fn(predicted, accept="text/csv")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(out, encoding="utf-8")
    print(f"Wrote {len(predicted)} rows to {output_path}")


if __name__ == "__main__":
    main()
