"""Core library for ECMWF preprocessing jobs in AWS Glue."""

__all__ = []

try:
    from .ecmwf import (
        EXPECTED_PRESSURE_LEVELS,
        build_pangu_arrays,
        build_parquet_frames,
        load_merged_dataset,
    )

    __all__.extend(
        [
            "EXPECTED_PRESSURE_LEVELS",
            "build_pangu_arrays",
            "build_parquet_frames",
            "load_merged_dataset",
        ]
    )
except ModuleNotFoundError:
    # Allow importing submodules that do not require cfgrib/eccodes.
    pass
