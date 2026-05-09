from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Config:
    # Rolling window/STFT config
    window_size: int = 128
    fft_bins: int = 64  # number of retained FFT bins (excluding DC)
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
    u_deadband: float = 0.05  # [N*m] corrections smaller than this are commanded as exactly zero
    u_rate_limit_per_s: float = 8.0  # [N*m/s] max correction change rate after deadband
    u_filter_alpha: float = 0.35  # low-pass blend for command smoothing (1 disables smoothing)

    # Heuristic training labels from spectral risk. These are used until real
    # torque-correction labels or closed-loop targets are available.
    use_heuristic_u_labels: bool = True
    risk_low_quantile: float = 0.60
    risk_high_quantile: float = 0.95
    heuristic_u_max_fraction: float = 0.35

    # Loss weights
    lambda_physics: float = 1.0
    lambda_data: float = 2.0
    lambda_u_mag: float = 2e-2  # discourage aggressive control

    # Early stopping (on validation total loss)
    early_stop_patience: int = 25
    early_stop_min_delta: float = 1e-4

    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def spectral_feature_dim(self) -> int:
        """
        Total spectral feature width consumed by the neural controller.

        Layout:
        - acceleration magnitude: fft_bins
        - acceleration phase as cos/sin: 2 * fft_bins
        - accel-vs-thrust cross phase as cos/sin: 2 * fft_bins
        - cross-phase drift inside the window as cos/sin: 2 * fft_bins
        """
        return 7 * int(self.fft_bins)
