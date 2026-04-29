from __future__ import annotations

"""
Current PARM API surface.

All implementation lives in the `parm/` package. This file simply re-exports the public objects
for convenience within this private repo.
"""

from parm import (  # type: ignore
    Config,
    ParmDataset,
    ParmONNXWrapper,
    ParmPINN,
    build_rolling_samples_from_timeseries,
    export_onnx,
    load_aggregate_csv,
    load_any_csv_as_trajectories,
    load_filtered_csv,
    load_openrocket_csv,
    physics_residual_loss,
    prepare_training_data_from_openrocket_exports,
    stft_features_from_window,
    total_loss,
    train_model,
)

__all__ = [
    "Config",
    "ParmDataset",
    "ParmONNXWrapper",
    "ParmPINN",
    "build_rolling_samples_from_timeseries",
    "export_onnx",
    "load_aggregate_csv",
    "load_any_csv_as_trajectories",
    "load_filtered_csv",
    "load_openrocket_csv",
    "physics_residual_loss",
    "prepare_training_data_from_openrocket_exports",
    "stft_features_from_window",
    "total_loss",
    "train_model",
]