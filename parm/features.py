from __future__ import annotations

import numpy as np

from .config import Config


def _hann(n: int) -> np.ndarray:
    if n <= 1:
        return np.ones((n,), dtype=np.float32)
    return np.hanning(n).astype(np.float32)


def spectral_features_from_windows(
    accel_window: np.ndarray,
    thrust_window: np.ndarray,
    cfg: Config,
) -> np.ndarray:
    """
    Computes Jetson-friendly spectral features for resonance and phase-drift detection.

    Feature layout:
    - acceleration FFT magnitude
    - acceleration FFT phase as cos/sin
    - cross phase between acceleration response and thrust forcing as cos/sin
    - cross-phase drift from the older half to the newer half of the window

    Cross phase is computed from accel_fft * conj(thrust_fft), which preserves the
    relative phase between response and forcing without using unstable wrapped angles
    as raw model inputs.
    """
    accel = np.asarray(accel_window, dtype=np.float32)
    thrust = np.asarray(thrust_window, dtype=np.float32)
    if accel.ndim != 1 or thrust.ndim != 1:
        raise ValueError("accel_window and thrust_window must be 1D")
    if len(accel) != cfg.window_size or len(thrust) != cfg.window_size:
        raise ValueError(f"window lengths must be {cfg.window_size}")

    win = _hann(cfg.window_size)
    accel_spec = np.fft.rfft(accel * win)[1:]
    thrust_spec = np.fft.rfft(thrust * win)[1:]

    accel_spec = accel_spec[: cfg.fft_bins]
    thrust_spec = thrust_spec[: cfg.fft_bins]
    if accel_spec.shape[0] < cfg.fft_bins:
        pad = cfg.fft_bins - accel_spec.shape[0]
        accel_spec = np.pad(accel_spec, (0, pad))
        thrust_spec = np.pad(thrust_spec, (0, pad))

    eps = np.float32(1e-6)
    mag = np.abs(accel_spec).astype(np.float32)
    if cfg.fft_log1p:
        mag = np.log1p(mag)

    parts = [mag]

    accel_abs = np.maximum(np.abs(accel_spec).astype(np.float32), eps)
    parts.append((np.real(accel_spec).astype(np.float32) / accel_abs).astype(np.float32))
    parts.append((np.imag(accel_spec).astype(np.float32) / accel_abs).astype(np.float32))

    cross = accel_spec * np.conj(thrust_spec)
    cross_abs = np.maximum(np.abs(cross).astype(np.float32), eps)
    parts.append((np.real(cross).astype(np.float32) / cross_abs).astype(np.float32))
    parts.append((np.imag(cross).astype(np.float32) / cross_abs).astype(np.float32))

    half = cfg.window_size // 2
    older_accel = np.fft.rfft(accel[:half] * _hann(half))[1:]
    newer_accel = np.fft.rfft(accel[-half:] * _hann(half))[1:]
    older_thrust = np.fft.rfft(thrust[:half] * _hann(half))[1:]
    newer_thrust = np.fft.rfft(thrust[-half:] * _hann(half))[1:]

    older_cross = older_accel * np.conj(older_thrust)
    newer_cross = newer_accel * np.conj(newer_thrust)
    drift = newer_cross * np.conj(older_cross)
    drift = drift[: cfg.fft_bins]
    if drift.shape[0] < cfg.fft_bins:
        drift = np.pad(drift, (0, cfg.fft_bins - drift.shape[0]))

    drift_abs = np.maximum(np.abs(drift).astype(np.float32), eps)
    parts.append((np.real(drift).astype(np.float32) / drift_abs).astype(np.float32))
    parts.append((np.imag(drift).astype(np.float32) / drift_abs).astype(np.float32))

    return np.concatenate(parts).astype(np.float32)
