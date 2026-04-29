from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Config:
    # Rolling window/STFT config
    window_size: int = 128
    fft_bins: int = 64  # number of magnitude bins from the windowed FFT (excluding DC)
    fft_log1p: bool = True

    # Training
    batch_size: int = 128
    epochs: int = 200
    lr: float = 1e-3
    weight_decay: float = 1e-6

    # Physical constants (replace with measured/identified values)
    I: float = 0.025  # rotational inertia [kg*m^2]
    k: float = 12.0  # torsional stiffness [N*m/rad]
    gamma: float = 0.0  # damping coefficient [N*m*s/rad]

    # Map OpenRocket thrust [N] to motor torque [N*m] via an effective lever arm.
    # If you already have motor torque telemetry, set lever_arm_m=0 and provide tau_motor directly.
    lever_arm_m: float = 0.05

    # Controller output limits (torque reduction magnitude)
    u_max: float = 5.0  # [N*m] max magnitude of corrective torque (applied as negative reduction)

    # Loss weights
    lambda_physics: float = 1.0
    lambda_data: float = 0.0  # optional supervised torque targets if available later
    lambda_u_mag: float = 1e-3  # discourage aggressive control

    # Early stopping (on validation total loss)
    early_stop_patience: int = 25
    early_stop_min_delta: float = 1e-4

    device: str = "cuda" if torch.cuda.is_available() else "cpu"

