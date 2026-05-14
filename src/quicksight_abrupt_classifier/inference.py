import csv
import io
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

ACCEPTED_CONTENT_TYPES = {"text/csv", "application/csv", "csv"}
CSV_ACCEPT_TYPES = {"text/csv", "application/csv", "csv"}
DEFAULT_CONFIG = {"mean_t2m": 285.0, "std_t2m": 8.0, "threshold": 2.0}


def model_fn(model_dir: str) -> Dict[str, float]:
    config_path = Path(model_dir) / "model_config.json"
    if not config_path.exists():
        return DEFAULT_CONFIG.copy()
    with config_path.open("r", encoding="utf-8") as fp:
        data = json.load(fp)
    cfg = DEFAULT_CONFIG.copy()
    cfg.update(data)
    _validate_config(cfg)
    return cfg


def input_fn(request_body, content_type: str = "text/csv") -> List[Tuple[float, float, float]]:
    if not isinstance(content_type, str) or content_type.lower() not in ACCEPTED_CONTENT_TYPES:
        raise ValueError(f"Unsupported content type: {content_type}")

    if isinstance(request_body, (bytes, bytearray)):
        text = request_body.decode("utf-8")
    elif isinstance(request_body, str):
        text = request_body
    else:
        raise ValueError("CSV payload must be string or bytes")

    rows: List[Tuple[float, float, float]] = []
    reader = csv.reader(io.StringIO(text))
    for idx, row in enumerate(reader, start=1):
        if len(row) != 3:
            raise ValueError(f"Row {idx}: expected exactly 3 columns (lat, lon, t2m), got {len(row)}")
        lat = _parse_float(row[0], "lat", idx)
        lon = _parse_float(row[1], "lon", idx)
        t2m = _parse_float(row[2], "t2m", idx)
        _validate_lat_lon(lat, lon, idx)
        rows.append((lat, lon, t2m))

    if not rows:
        raise ValueError("Input CSV has no rows")
    return rows


def predict_fn(input_data: Sequence[Tuple[float, float, float]], model: Dict[str, float]) -> List[Tuple[int, float]]:
    mean_t2m = float(model["mean_t2m"])
    std_t2m = float(model["std_t2m"])
    threshold = float(model["threshold"])
    if std_t2m <= 0.0:
        raise ValueError("std_t2m must be > 0")

    output: List[Tuple[int, float]] = []
    for _, _, t2m in input_data:
        score = abs(float(t2m) - mean_t2m) / std_t2m
        label = 1 if score >= threshold else 0
        output.append((label, score))
    return output


def output_fn(prediction: Sequence[Tuple[int, float]], accept: str = "text/csv"):
    accept_value = (accept or "text/csv").lower()
    if accept_value not in CSV_ACCEPT_TYPES:
        raise ValueError(f"Unsupported accept type: {accept}")

    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    for label, score in prediction:
        writer.writerow([int(label), f"{float(score):.6f}"])
    return buffer.getvalue(), "text/csv"


def _validate_config(config: Dict[str, float]) -> None:
    for key in ("mean_t2m", "std_t2m", "threshold"):
        if key not in config:
            raise ValueError(f"Missing config key: {key}")
        try:
            float(config[key])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Config key '{key}' must be numeric") from exc
    if float(config["std_t2m"]) <= 0:
        raise ValueError("Config key 'std_t2m' must be > 0")


def _parse_float(raw: str, name: str, row_num: int) -> float:
    try:
        return float(str(raw).strip())
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Row {row_num}: invalid {name} value '{raw}'") from exc


def _validate_lat_lon(lat: float, lon: float, row_num: int) -> None:
    if lat < -90.0 or lat > 90.0:
        raise ValueError(f"Row {row_num}: lat out of range [-90, 90]: {lat}")
    if lon < -180.0 or lon > 180.0:
        raise ValueError(f"Row {row_num}: lon out of range [-180, 180]: {lon}")
