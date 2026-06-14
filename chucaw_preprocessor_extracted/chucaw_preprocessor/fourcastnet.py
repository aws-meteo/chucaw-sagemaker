"""FourCastNet snapshot preprocessing helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

EXPECTED_LATS = np.arange(90.0, -90.0, -0.25, dtype=np.float32)
EXPECTED_LONS = np.arange(0.0, 360.0, 0.25, dtype=np.float32)
ALLOWED_LATITUDE_POLICIES = {"fail", "drop_south_pole", "drop_north_pole"}
FCN_DATASET_NAME = "fields"
FOURCASTNET_REQUIRED_CHANNELS = [
    ("u10", None),
    ("v10", None),
    ("t2m", None),
    ("sp", None),
    ("msl", None),
    ("t", 850.0),
    ("u", 1000.0),
    ("v", 1000.0),
    ("z", 1000.0),
    ("u", 850.0),
    ("v", 850.0),
    ("z", 850.0),
    ("u", 500.0),
    ("v", 500.0),
    ("z", 500.0),
    ("t", 500.0),
    ("z", 50.0),
    ("r", 500.0),
    ("r", 850.0),
    ("tcwv", None),
]


def normalize_level_column(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize pressure-level column naming into ``isobaricInhPa``."""
    if "isobaricInhPa" in df.columns:
        return df
    if "isobaricinhpa" in df.columns:
        normalized = df.copy(deep=False)
        normalized["isobaricInhPa"] = normalized["isobaricinhpa"]
        return normalized
    raise ValueError("Missing pressure-level column: expected isobaricInhPa or isobaricinhpa")


def normalize_longitudes(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize longitudes into [0, 360)."""
    if "longitude" not in df.columns:
        raise ValueError("Missing required column: longitude")
    normalized = df.copy(deep=False)
    normalized["longitude"] = np.mod(normalized["longitude"].astype(float), 360.0)
    return normalized


def validate_latitude_policy(latitude_policy: str) -> str:
    policy = str(latitude_policy or "fail").strip().lower()
    if policy not in ALLOWED_LATITUDE_POLICIES:
        raise ValueError(
            f"Invalid latitude_policy={latitude_policy!r}. "
            f"Expected one of {sorted(ALLOWED_LATITUDE_POLICIES)}"
        )
    return policy


def resolve_expected_lats(
    observed_lats: np.ndarray,
    latitude_policy: str = "fail",
) -> tuple[np.ndarray | None, dict[str, Any]]:
    policy = validate_latitude_policy(latitude_policy)
    lats = np.array(sorted(np.asarray(observed_lats, dtype=float).tolist(), reverse=True), dtype=np.float32)

    details: dict[str, Any] = {
        "original_lat_count": int(len(lats)),
        "final_lat_count": int(len(lats)),
        "original_lat_min": float(np.min(lats)) if len(lats) else None,
        "original_lat_max": float(np.max(lats)) if len(lats) else None,
        "final_lat_min": float(np.min(lats)) if len(lats) else None,
        "final_lat_max": float(np.max(lats)) if len(lats) else None,
        "latitude_policy": policy,
        "latitude_policy_applied": "none",
    }

    if len(lats) == 720:
        return lats, details

    if len(lats) == 721:
        if policy == "fail":
            details["latitude_policy_applied"] = "latitude_policy_required"
            return None, details
        if policy == "drop_south_pole":
            dropped_lat = -90.0
            out = lats[lats > -90.0]
        else:
            dropped_lat = 90.0
            out = lats[lats < 90.0]
        details["latitude_policy_applied"] = policy
        details["dropped_latitude"] = dropped_lat
        details["final_lat_count"] = int(len(out))
        details["final_lat_min"] = float(np.min(out)) if len(out) else None
        details["final_lat_max"] = float(np.max(out)) if len(out) else None
        return out.astype(np.float32), details

    details["latitude_policy_applied"] = "unsupported_latitude_count"
    return None, details


def build_fourcastnet_channel_map() -> list[tuple[str, float | None]]:
    """Return FourCastNet V0 channel mapping as (variable, level)."""
    return list(FOURCASTNET_REQUIRED_CHANNELS)


def build_fourcastnet_channel_descriptors() -> list[dict[str, Any]]:
    """Return channel descriptors with stable order and index."""
    out: list[dict[str, Any]] = []
    for idx, (variable, level) in enumerate(build_fourcastnet_channel_map()):
        out.append({"channel_index": idx, "variable": variable, "level": level})
    return out


def specific_humidity_to_relative_humidity(
    q: np.ndarray,
    t_k: np.ndarray,
    pressure_hpa: float,
    *,
    clip_percent: bool = True,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Derive FCN-compatible relative humidity from specific humidity.

    This helper is intended for deriving the FourCastNet ``r`` channel (RH %) from
    specific humidity ``q`` (kg/kg), temperature ``t_k`` (K), and pressure (hPa).
    It must be validated against trusted ERA5 relative_humidity before production use.
    """
    q_arr = np.asarray(q, dtype=np.float64)
    t_arr = np.asarray(t_k, dtype=np.float64)
    if q_arr.shape != t_arr.shape:
        raise ValueError(f"q and t_k shape mismatch: {q_arr.shape} != {t_arr.shape}")
    if not np.isfinite(pressure_hpa) or pressure_hpa <= 0.0:
        raise ValueError(f"Invalid pressure_hpa={pressure_hpa!r}")
    if not np.all(np.isfinite(q_arr)) or not np.all(np.isfinite(t_arr)):
        raise ValueError("q and t_k must be finite")

    # Bolton-style saturation vapor pressure over water (hPa).
    e_s = 6.112 * np.exp((17.67 * (t_arr - 273.15)) / (t_arr - 29.65))
    # Convert q to mixing ratio.
    w = q_arr / np.maximum(1.0 - q_arr, 1e-12)
    epsilon = 0.622
    p = float(pressure_hpa)
    ws = (epsilon * e_s) / np.maximum(p - e_s, 1e-6)
    rh_ratio = w / np.maximum(ws, 1e-12)
    rh_percent = 100.0 * rh_ratio

    clipped_count = 0
    if clip_percent:
        before = rh_percent.copy()
        rh_percent = np.clip(rh_percent, 0.0, 100.0)
        clipped_count = int(np.count_nonzero(before != rh_percent))

    if not np.all(np.isfinite(rh_percent)):
        raise ValueError("Derived RH contains non-finite values")

    summary = {
        "pressure_hpa": float(pressure_hpa),
        "clip_percent": bool(clip_percent),
        "clipped_value_count": clipped_count,
        "rh_percent_min": float(np.min(rh_percent)),
        "rh_percent_max": float(np.max(rh_percent)),
        "rh_percent_mean": float(np.mean(rh_percent)),
    }
    return rh_percent.astype(np.float32), summary


def select_channel_frame(df: pd.DataFrame, variable: str, level: float | None) -> pd.DataFrame:
    """Select one channel dataframe from normalized tidy rows."""
    base = normalize_level_column(df)
    required = {"variable", "latitude", "longitude", "value", "isobaricInhPa"}
    missing_cols = sorted(required - set(base.columns))
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    vdf = base[base["variable"].astype(str).str.lower() == variable.lower()].copy()
    level_values = pd.to_numeric(vdf["isobaricInhPa"], errors="coerce")
    if level is None:
        surface_mask = level_values.isna() | np.isclose(level_values, 0.0, atol=1e-6)
        return vdf[surface_mask].copy()

    return vdf[np.isclose(level_values, float(level), atol=1e-6)].copy()


def field_to_grid(
    df_channel: pd.DataFrame,
    expected_lats: np.ndarray = EXPECTED_LATS,
    expected_lons: np.ndarray = EXPECTED_LONS,
) -> np.ndarray:
    """Convert a tidy channel dataframe into a 2D ordered FCN grid."""
    if df_channel.empty:
        raise ValueError("Channel frame is empty")

    work = normalize_longitudes(df_channel)
    work = work[["latitude", "longitude", "value"]].copy()
    work["latitude"] = work["latitude"].astype(float)
    work["longitude"] = work["longitude"].astype(float)

    rounded_lats = np.round(work["latitude"].to_numpy(), 5)
    rounded_lons = np.round(work["longitude"].to_numpy(), 5)
    if len(np.unique(rounded_lats)) != len(expected_lats):
        raise ValueError("Latitude cardinality mismatch for channel")
    if len(np.unique(rounded_lons)) != len(expected_lons):
        raise ValueError("Longitude cardinality mismatch for channel")

    duplicates = work.duplicated(subset=["latitude", "longitude"]).any()
    if duplicates:
        raise ValueError("Duplicate (latitude, longitude) points found")

    pivot = work.pivot(index="latitude", columns="longitude", values="value")
    pivot = pivot.reindex(index=expected_lats, columns=expected_lons)

    if pivot.isnull().any().any():
        raise ValueError("Channel grid has missing points after reindex")

    return pivot.to_numpy(dtype=np.float32)


def build_validation_report(df: pd.DataFrame, latitude_policy: str = "fail") -> dict[str, Any]:
    """Build a validation report for schema/channel/grid coverage."""
    report: dict[str, Any] = {
        "ok": False,
        "missing_columns": [],
        "missing_channels": [],
        "available_variables": [],
        "available_variable_levels": [],
        "duplicate_points_by_channel": [],
        "grid": {},
        "row_count": int(len(df)),
    }

    required_cols = ["latitude", "longitude", "variable", "value", "date", "run"]
    missing_cols = [col for col in required_cols if col not in df.columns]

    has_level = "isobaricInhPa" in df.columns or "isobaricinhpa" in df.columns
    if not has_level:
        missing_cols.append("isobaricInhPa|isobaricinhpa")

    report["missing_columns"] = missing_cols
    if missing_cols:
        return report

    normalized = normalize_level_column(df)
    levels = pd.to_numeric(normalized["isobaricInhPa"], errors="coerce")
    lat_values = pd.to_numeric(normalized["latitude"], errors="coerce").dropna().unique()
    lon_values = np.mod(pd.to_numeric(normalized["longitude"], errors="coerce").dropna().unique(), 360.0)
    expected_lats, lat_meta = resolve_expected_lats(lat_values, latitude_policy=latitude_policy)
    variable_raw = normalized["variable"]
    variable_cat = variable_raw.astype("category")
    unique_variables = variable_cat.cat.categories.tolist()
    lower_to_variants: dict[str, list[Any]] = {}
    for raw in unique_variables:
        lower_to_variants.setdefault(str(raw).lower(), []).append(raw)
    report["available_variables"] = sorted(lower_to_variants.keys())

    lowered_categories = variable_cat.cat.categories.astype("string").str.lower()
    lowered_variable = variable_cat.cat.rename_categories(lowered_categories)
    grouped = pd.DataFrame({"variable": lowered_variable, "isobaricInhPa": levels}).drop_duplicates()
    grouped = grouped.sort_values(["variable", "isobaricInhPa"], na_position="first")
    grouped["count"] = None
    report["available_variable_levels"] = grouped.to_dict(orient="records")

    missing_channels: list[dict[str, Any]] = []
    duplicate_channels: list[dict[str, Any]] = []
    for variable, level in build_fourcastnet_channel_map():
        variants = lower_to_variants.get(variable, [])
        if not variants:
            missing_channels.append({"variable": variable, "level": level})
            continue
        var_mask = variable_raw.isin(variants)
        if level is None:
            lvl_mask = levels.isna() | np.isclose(levels, 0.0, atol=1e-6)
        else:
            lvl_mask = np.isclose(levels, float(level), atol=1e-6)
        selected = normalized.loc[var_mask & lvl_mask, ["latitude", "longitude"]]
        if selected.empty:
            missing_channels.append({"variable": variable, "level": level})
            continue
        dup_count = int(selected.duplicated(subset=["latitude", "longitude"]).sum())
        if dup_count > 0:
            duplicate_channels.append(
                {"variable": variable, "level": level, "duplicate_points": dup_count}
            )

    report["missing_channels"] = missing_channels
    report["duplicate_points_by_channel"] = duplicate_channels
    report["grid"] = {
        "expected_lat_count": int(len(EXPECTED_LATS)),
        "expected_lon_count": int(len(EXPECTED_LONS)),
        "observed_lat_count": int(len(lat_values)),
        "observed_lon_count": int(len(lon_values)),
        "lat_descending_expected": True,
        "lon_ascending_expected": True,
        "longitude_normalization_status": "normalized_to_0_360",
    }
    report["latitude"] = lat_meta
    latitude_ready = expected_lats is not None and len(expected_lats) == 720
    report["ok"] = (
        len(missing_channels) == 0
        and len(report["duplicate_points_by_channel"]) == 0
        and latitude_ready
    )
    return report


def build_available_variable_levels(df: pd.DataFrame) -> pd.DataFrame:
    """Return availability matrix grouped by variable and pressure level."""
    normalized = normalize_level_column(df)
    variable_cat = normalized["variable"].astype("category")
    lowered_categories = variable_cat.cat.categories.astype("string").str.lower()
    variables = variable_cat.cat.rename_categories(lowered_categories)
    levels = pd.to_numeric(normalized["isobaricInhPa"], errors="coerce")
    unique_levels = (
        pd.DataFrame({"variable": variables.str.lower(), "isobaricInhPa": levels})
        .drop_duplicates()
        .sort_values(["variable", "isobaricInhPa"], na_position="first")
    )
    unique_levels["count"] = None
    return unique_levels


def build_fourcastnet_tensor(
    df: pd.DataFrame,
    expected_lats: np.ndarray = EXPECTED_LATS,
    expected_lons: np.ndarray = EXPECTED_LONS,
    validate_shape: bool = True,
    latitude_policy: str = "fail",
) -> np.ndarray:
    """Build one FCN tensor with shape ``(1, 20, 720, 1440)``."""
    normalized = normalize_longitudes(normalize_level_column(df))
    normalized["variable"] = normalized["variable"].astype(str).str.lower()
    if expected_lats is EXPECTED_LATS:
        observed_lats = pd.to_numeric(normalized["latitude"], errors="coerce").dropna().unique()
        resolved_lats, lat_meta = resolve_expected_lats(observed_lats, latitude_policy=latitude_policy)
        if resolved_lats is None:
            raise ValueError(
                f"Latitude policy prevented tensor build: {lat_meta['latitude_policy_applied']}"
            )
        expected_lats = resolved_lats

    channels = []
    for variable, level in build_fourcastnet_channel_map():
        selected = select_channel_frame(normalized, variable, level)
        if selected.empty:
            level_text = "surface" if level is None else f"{level:g}"
            raise ValueError(f"Missing required channel: variable={variable}, level={level_text}")
        channels.append(field_to_grid(selected, expected_lats, expected_lons))

    stacked = np.stack(channels, axis=0).astype(np.float32)
    tensor = np.expand_dims(stacked, axis=0)

    if validate_shape and tensor.shape != (1, 20, 720, 1440):
        raise ValueError(f"Unexpected tensor shape: {tensor.shape}")
    return tensor


def write_manifest(path: str | Path, payload: dict[str, Any]) -> str:
    """Write JSON manifest/validation report to disk."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return str(out)
