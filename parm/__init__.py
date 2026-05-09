"""
PARM (Predictive Adaptive Resonance Mitigation) package.

This package contains the split-out implementation that used to live in `PARM/model.py`.
`PARM/model.py` remains as a compatibility shim for older imports.
"""

from .config import Config
from .control import TorqueCommandFilter, apply_torque_deadband, clamp_torque_correction, commanded_torque
from .data import (
    ParmDataset,
    build_rolling_samples_from_timeseries,
    heuristic_u_labels_from_spectral_features,
    load_aggregate_csv,
    load_any_csv_as_trajectories,
    load_filtered_csv,
    load_openrocket_csv,
    prepare_training_data_from_openrocket_exports,
    risk_score_from_spectral_features,
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
    "TorqueCommandFilter",
    "apply_torque_deadband",
    "build_rolling_samples_from_timeseries",
    "clamp_torque_correction",
    "commanded_torque",
    "export_onnx",
    "heuristic_u_labels_from_spectral_features",
    "load_aggregate_csv",
    "load_any_csv_as_trajectories",
    "load_filtered_csv",
    "load_openrocket_csv",
    "physics_residual_loss",
    "prepare_training_data_from_openrocket_exports",
    "risk_score_from_spectral_features",
    "spectral_features_from_windows",
    "total_loss",
    "train_model",
]
