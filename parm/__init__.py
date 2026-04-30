"""
PARM (Predictive Adaptive Resonance Mitigation) package.

This package contains the split-out implementation that used to live in `PARM/model.py`.
`PARM/model.py` remains as a compatibility shim for older imports.
"""

from .config import Config
from .data import (
    ParmDataset,
    build_rolling_samples_from_timeseries,
    load_aggregate_csv,
    load_any_csv_as_trajectories,
    load_filtered_csv,
    load_openrocket_csv,
    prepare_training_data_from_openrocket_exports,
)
from .export import ParmONNXWrapper, export_onnx
from .features import spectral_features_from_windows
from .losses import physics_residual_loss, total_loss
from .network import ParmPINN
from .training import train_model

__all__ = [
    "Config",
    "ParmDataset",
    "ParmPINN",
    "ParmONNXWrapper",
    "build_rolling_samples_from_timeseries",
    "export_onnx",
    "load_aggregate_csv",
    "load_any_csv_as_trajectories",
    "load_filtered_csv",
    "load_openrocket_csv",
    "physics_residual_loss",
    "prepare_training_data_from_openrocket_exports",
    "spectral_features_from_windows",
    "total_loss",
    "train_model",
]
