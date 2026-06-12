"""Core ECMWF preprocessing utilities for GRIB ingestion and serialization."""

import os
from pathlib import Path

import boto3

try:
    import eccodeslib

    lib_dir_candidate = Path(eccodeslib.__file__).resolve().parent / "lib"
    if lib_dir_candidate.exists():
        lib_dir = lib_dir_candidate
    else:
        eccodes_root = Path(eccodeslib.__file__).resolve().parent
        matches = list(eccodes_root.rglob("libeccodes*.so*"))
        if matches:
            lib_dir = matches[0].parent
        else:
            lib_dir = None

    if lib_dir and lib_dir.exists():
        os.environ["ECCODES_DIR"] = str(lib_dir)
        current_ld_path = os.environ.get("LD_LIBRARY_PATH", "")
        os.environ["LD_LIBRARY_PATH"] = f"{lib_dir}:{current_ld_path}" if current_ld_path else str(lib_dir)
except ImportError:
    pass

try:
    import cfgrib
except ModuleNotFoundError:  # pragma: no cover - exercised in minimal test envs
    cfgrib = None
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import xarray as xr

EXPECTED_PRESSURE_LEVELS = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]
_DROP_COORDS = ["heightAboveGround", "meanSea", "entireAtmosphere", "soilLayer", "surface"]
PANGU_SURFACE_VARS = ["msl", "u10", "v10", "t2m"]
PANGU_UPPER_VARS = ["q", "t", "u", "v"]
_GRAVITY = 9.80665

# Expressive mapping for cryptic GRIB codes to easy-to-write names.
# Core FourCastNet variables (u10, v10, t2m, sp, msl, tcwv, z, q, t, u, v, r) 
# are preserved with their standard short-names for model compatibility.
_EXPRESSIVE_MAPPING = {
    "p49": "wind_gust_max_10m",
    "tp": "total_precipitation",
    "skt": "skin_temperature",
    "u100": "wind_u_100m",
    "v100": "wind_v_100m",
    "asn": "snow_albedo",
    "rsn": "snow_density",
    "sd": "snow_depth",
    "sf": "snowfall",
    "ssrd": "solar_radiation_downwards",
    "strd": "thermal_radiation_downwards",
    "ssr": "net_solar_radiation",
    "str": "net_thermal_radiation",
    "ro": "runoff",
    "tprate": "total_precipitation_rate",
    "sithick": "sea_ice_thickness",
    "zos": "sea_surface_height",
}


def download_grib_from_s3(bucket: str, key: str, download_dir: str = "/tmp") -> str:
    """Download GRIB object from S3.

    Parameters
    ----------
    bucket : str
        Source S3 bucket.
    key : str
        Source S3 key.
    download_dir : str, default "/tmp"
        Local directory used for temporary download.

    Returns
    -------
    str
        Local path to downloaded GRIB file.
    """
    local_path = str(Path(download_dir) / Path(key).name)
    s3 = boto3.client("s3")
    s3.download_file(bucket, key, local_path)
    return local_path


def upload_file_to_s3(local_path: str, bucket: str, key: str) -> None:
    """Upload local file to S3."""
    s3 = boto3.client("s3")
    s3.upload_file(local_path, bucket, key)


def load_merged_dataset(grib_path: str) -> xr.Dataset:
    """Load and merge GRIB message groups into a single dataset.

    Parameters
    ----------
    grib_path : str
        Local path to a GRIB file.

    Returns
    -------
    xarray.Dataset
        Merged dataset sorted by latitude (descending), when available.
    """
    if cfgrib is None:
        raise ModuleNotFoundError(
            "cfgrib is required to load GRIB files. Install cfgrib/eccodes for runtime ingestion."
        )

    # Use decode_timedelta=False to avoid pandas 3.x compatibility issues
    backend_kwargs = {"decode_timedelta": False}
    datasets = cfgrib.open_datasets(grib_path, backend_kwargs=backend_kwargs)
    cleaned = []
    for dataset in datasets:
        cleaned.append(dataset.drop_vars([c for c in dataset.coords if c in _DROP_COORDS], errors="ignore"))
    merged = xr.merge(cleaned, compat="override")
    if "latitude" in merged.coords:
        merged = merged.sortby("latitude", ascending=False)
    return merged


def _squeeze(data_array: xr.DataArray) -> np.ndarray:
    return np.asarray(data_array.values).squeeze()


def build_pangu_arrays(ds: xr.Dataset) -> tuple[np.ndarray, np.ndarray]:
    """Build Pangu tensors from merged ECMWF dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        Merged ECMWF dataset.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        ``(surface_array, upper_array)`` with ``float32`` dtype.
    """
    # Pangu strictly expects exactly 4 surface vars and 5 upper vars (including z)
    surface_values = [_squeeze(ds[var]) for var in PANGU_SURFACE_VARS]
    surface_array = np.stack(surface_values, axis=0).astype(np.float32)

    ds_pl = ds.sel(isobaricInhPa=EXPECTED_PRESSURE_LEVELS)
    z = _squeeze(ds_pl["gh"]) * _GRAVITY
    upper_values = [z] + [_squeeze(ds_pl[var]) for var in PANGU_UPPER_VARS]
    upper_array = np.stack(upper_values, axis=0).astype(np.float32)
    return surface_array, upper_array


def _split_platinum_vars(ds: xr.Dataset) -> tuple[list[str], list[str]]:
    """Split dataset vars into surface-like and pressure-level vars.

    Platinum ingestion policy is exhaustive: include all variables available in the
    source dataset, and for upper-air fields include all pressure levels present.
    """
    surface_vars: list[str] = []
    upper_vars: list[str] = []
    for var in ds.data_vars:
        if "isobaricInhPa" in ds[var].coords:
            upper_vars.append(var)
        else:
            surface_vars.append(var)
    return surface_vars, upper_vars


def build_parquet_frames(ds: xr.Dataset) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build exhaustive Platinum tidy dataframes for surface/upper fields.

    Parameters
    ----------
    ds : xarray.Dataset
        Merged ECMWF dataset.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.DataFrame]
        ``(surface_df, upper_df)``.
    """
    surface_vars, upper_vars = _split_platinum_vars(ds)

    surface_frames: list[pd.DataFrame] = []
    for var in surface_vars:
        ds_var = ds[[var]].to_dataframe().reset_index().rename(columns={var: "value"})
        ds_var["variable"] = _EXPRESSIVE_MAPPING.get(var, var)
        ds_var["value"] = ds_var["value"].astype("float32")
        ds_var["isobaricInhPa"] = np.nan
        surface_frames.append(ds_var[["isobaricInhPa", "latitude", "longitude", "variable", "value"]])

    if surface_frames:
        surface = pd.concat(surface_frames, ignore_index=True)
    else:
        surface = pd.DataFrame(columns=["isobaricInhPa", "latitude", "longitude", "variable", "value"])

    upper_frames: list[pd.DataFrame] = []
    for var in upper_vars:
        ds_pl = ds[[var]]
        for p_level in ds_pl.isobaricInhPa.values:
            ds_sub = ds_pl.sel(isobaricInhPa=p_level)
            if var == "gh":
                z_array = ds_sub["gh"] * _GRAVITY
                z_array.name = "z"
                ds_df = z_array.to_dataframe().reset_index()
                v_name = "z"
                rename_source = "z"
            else:
                ds_df = ds_sub.to_dataframe().reset_index()
                v_name = _EXPRESSIVE_MAPPING.get(var, var)
                rename_source = var
            ds_df = ds_df.rename(columns={rename_source: "value"})
            ds_df["variable"] = v_name
            ds_df["value"] = ds_df["value"].astype("float32")
            ds_df["isobaricInhPa"] = float(p_level)
            upper_frames.append(ds_df[["isobaricInhPa", "latitude", "longitude", "variable", "value"]])

    if upper_frames:
        upper = pd.concat(upper_frames, ignore_index=True)
    else:
        upper = pd.DataFrame(columns=["isobaricInhPa", "latitude", "longitude", "variable", "value"])
    return surface, upper


def write_pangu_arrays(surface_array: np.ndarray, upper_array: np.ndarray, output_dir: str) -> tuple[str, str]:
    """Write Pangu arrays to ``.npy`` files."""
    os.makedirs(output_dir, exist_ok=True)
    surface_path = str(Path(output_dir) / "input_surface.npy")
    upper_path = str(Path(output_dir) / "input_upper.npy")
    np.save(surface_path, surface_array)
    np.save(upper_path, upper_array)
    return surface_path, upper_path


def write_parquet_frames(surface_df: pd.DataFrame, upper_df: pd.DataFrame, output_dir: str) -> tuple[str, str]:
    """Write surface/upper dataframes to parquet files."""
    os.makedirs(output_dir, exist_ok=True)
    surface_path = str(Path(output_dir) / "surface.parquet")
    upper_path = str(Path(output_dir) / "upper.parquet")
    surface_df.to_parquet(surface_path, index=False, compression="snappy")
    upper_df.to_parquet(upper_path, index=False, compression="snappy")
    return surface_path, upper_path

import pyarrow as pa
import pyarrow.parquet as pq


def serialize_parquet_chunked(ds: xr.Dataset, output_path: str, date_str: str, run_str: str) -> str:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    writer = None
    
    surface_vars, upper_vars = _split_platinum_vars(ds)

    # A. Process Surface Variables (and all others without pressure levels)
    for var in surface_vars:
        # Handle soilLayer variables by encoding level in name
        if "soilLayer" in ds[var].coords:
            for sl in ds[var].soilLayer.values:
                ds_sub = ds[[var]].sel(soilLayer=sl).to_dataframe().reset_index()
                ds_sub = ds_sub.rename(columns={var: "value"})
                v_name = f"{var}_soil{int(sl)}"
                ds_sub["variable"] = _EXPRESSIVE_MAPPING.get(v_name, v_name)
                ds_sub["value"] = ds_sub["value"].astype("float32")
                ds_sub["date"] = date_str
                ds_sub["run"] = run_str
                ds_sub["isobaricInhPa"] = np.nan
                cols = ["isobaricInhPa", "latitude", "longitude", "variable", "value", "date", "run"]
                table = pa.Table.from_pandas(ds_sub[cols])
                if writer is None:
                    writer = pq.ParquetWriter(output_path, table.schema, compression="snappy")
                writer.write_table(table)
            continue

        ds_var = ds[[var]].to_dataframe().reset_index()
        ds_var = ds_var.rename(columns={var: "value"})
        
        # Apply expressive name mapping (keep both for FCN compatibility if needed, 
        # but here we follow user request for expressive names)
        # Decision: If a mapping exists, use it. FCN logic will be updated to look for these.
        # Actually, let's keep FCN core names if they are already easy.
        v_name = _EXPRESSIVE_MAPPING.get(var, var)
        
        ds_var["variable"] = v_name
        ds_var["value"] = ds_var["value"].astype("float32")
        ds_var["date"] = date_str
        ds_var["run"] = run_str
        ds_var["isobaricInhPa"] = np.nan
        cols = ["isobaricInhPa", "latitude", "longitude", "variable", "value", "date", "run"]
        table = pa.Table.from_pandas(ds_var[cols])
        if writer is None:
            writer = pq.ParquetWriter(output_path, table.schema, compression="snappy")
        writer.write_table(table)

    # B. Process Upper-Level Variables
    for var in upper_vars:
        ds_pl = ds[[var]]
        for p_level in ds_pl.isobaricInhPa.values:
            ds_sub = ds_pl.sel(isobaricInhPa=p_level)
            
            # Special case: Geopotential height to Geopotential
            if var == "gh":
                z_array = ds_sub["gh"] * _GRAVITY
                z_array.name = "z"
                ds_sub_df = z_array.to_dataframe().reset_index()
                v_name = "z"
            else:
                ds_sub_df = ds_sub.to_dataframe().reset_index()
                v_name = _EXPRESSIVE_MAPPING.get(var, var)

            ds_sub_df = ds_sub_df.rename(columns={var if var != "gh" else "z": "value"})
            ds_sub_df["variable"] = v_name
            ds_sub_df["value"] = ds_sub_df["value"].astype("float32")
            ds_sub_df["date"] = date_str
            ds_sub_df["run"] = run_str
            ds_sub_df["isobaricInhPa"] = float(p_level)
            
            cols = ["isobaricInhPa", "latitude", "longitude", "variable", "value", "date", "run"]
            table = pa.Table.from_pandas(ds_sub_df[cols])
            if writer is None:
                writer = pq.ParquetWriter(output_path, table.schema, compression="snappy")
            writer.write_table(table)

    if writer: writer.close()
    return output_path
