from __future__ import annotations

import numpy as np

from .config import Config


def _hann(n: int) -> np.ndarray:
    if n <= 1:
        return np.ones((n,), dtype=np.float32)
    return np.hanning(n).astype(np.float32)


def stft_features_from_window(
    accel_window: np.ndarray,
    cfg: Config,
) -> np.ndarray:
    """
    Computes a compact frequency representation for a single rolling window of vertical acceleration.

    For training and embedded friendliness we approximate the STFT-at-each-timestep described in the
    writeup using a windowed FFT over the rolling window (equivalent to a single STFT frame).

    Returns: (cfg.fft_bins,) float32 magnitude spectrum (DC removed).
    """
    x = np.asarray(accel_window, dtype=np.float32)
    if x.ndim != 1:
        raise ValueError("accel_window must be 1D")
    if len(x) != cfg.window_size:
        raise ValueError(f"accel_window length must be {cfg.window_size}")

    xw = x * _hann(cfg.window_size)
    spec = np.fft.rfft(xw)
    mag = np.abs(spec).astype(np.float32)

    # drop DC
    mag = mag[1:]

    if cfg.fft_log1p:
        mag = np.log1p(mag)

    if cfg.fft_bins is not None:
        mag = mag[: cfg.fft_bins]
        if mag.shape[0] < cfg.fft_bins:
            mag = np.pad(mag, (0, cfg.fft_bins - mag.shape[0]))

    return mag.astype(np.float32)

