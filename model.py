"""
PARM (Predictive Adaptive Resonance Mitigation) model.

Aligned with the project writeup:
- Training data comes from OpenRocket CSV exports.
- Four key physical parameters are used: time, vertical acceleration, thrust, vertical velocity.
- A rolling window of vertical acceleration is transformed via STFT (implemented as a windowed FFT
  for efficiency) to produce a frequency representation.
- The PINN ingests the four parameters + STFT features and outputs a corrective torque command
  that temporarily reduces torque when approaching resonance.
- Physics-informed training constrains corrections using the rotational equation of motion.

Notes:
- OpenRocket does not provide torsional states (phi, phi_dot, phi_ddot). This PINN learns a latent
  torsional response phi(t) and enforces the governing equation via autograd time-derivatives.
- STFT/FFT feature extraction is kept OUTSIDE the exported ONNX model to keep inference latency low
  on embedded targets; ONNX exports the neural network controller only.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


# -----------------------------
# Config
# -----------------------------

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
    I: float = 0.025       # rotational inertia [kg*m^2]
    k: float = 12.0        # torsional stiffness [N*m/rad]
    gamma: float = 0.0     # damping coefficient [N*m*s/rad]

    # Map OpenRocket thrust [N] to motor torque [N*m] via an effective lever arm.
    # If you already have motor torque telemetry, set lever_arm_m=0 and provide tau_motor directly.
    lever_arm_m: float = 0.05

    # Controller output limits (torque reduction magnitude)
    u_max: float = 5.0  # [N*m] max magnitude of corrective torque (applied as negative reduction)

    # Loss weights
    lambda_physics: float = 1.0
    lambda_data: float = 0.0   # optional supervised torque targets if available later
    lambda_u_mag: float = 1e-3  # discourage aggressive control

    device: str = "cuda" if torch.cuda.is_available() else "cpu"


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


def build_rolling_samples_from_timeseries(
    t: np.ndarray,
    accel_z: np.ndarray,
    thrust: np.ndarray,
    v_z: np.ndarray,
    cfg: Config,
    u_label: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """
    Builds rolling-window samples.

    Each sample i (for i >= window_size) contains:
    - scalar physical inputs at time i: time, accel_z, thrust, v_z
    - STFT/FFT features from accel_z[i-window_size:i]
    - optional supervised label u_label[i]
    """
    t = np.asarray(t, dtype=np.float32)
    accel_z = np.asarray(accel_z, dtype=np.float32)
    thrust = np.asarray(thrust, dtype=np.float32)
    v_z = np.asarray(v_z, dtype=np.float32)

    n = len(t)
    if not (len(accel_z) == len(thrust) == len(v_z) == n):
        raise ValueError("time, accel_z, thrust, v_z must have the same length")
    if n <= cfg.window_size:
        raise ValueError("timeseries shorter than rolling window")

    n_samples = n - cfg.window_size
    scalar_x = np.zeros((n_samples, 4), dtype=np.float32)
    stft_x = np.zeros((n_samples, cfg.fft_bins), dtype=np.float32)

    y = None
    if u_label is not None:
        u_label = np.asarray(u_label, dtype=np.float32)
        if len(u_label) != n:
            raise ValueError("u_label length must match timeseries length")
        y = np.zeros((n_samples, 1), dtype=np.float32)

    for j, i in enumerate(range(cfg.window_size, n)):
        scalar_x[j, 0] = t[i]
        scalar_x[j, 1] = accel_z[i]
        scalar_x[j, 2] = thrust[i]
        scalar_x[j, 3] = v_z[i]

        stft_x[j] = stft_features_from_window(accel_z[i - cfg.window_size : i], cfg)

        if y is not None:
            y[j, 0] = u_label[i]

    return {
        "scalar_x": scalar_x,
        "stft_x": stft_x,
        "y": y if y is not None else np.zeros((n_samples, 1), dtype=np.float32),
    }


class ParmDataset(Dataset):
    """
    Each item corresponds to a timestep (after rolling window warmup) for a single flight trajectory.
    """

    def __init__(self, scalar_x: np.ndarray, stft_x: np.ndarray, y: np.ndarray):
        self.scalar_x = torch.tensor(scalar_x, dtype=torch.float32)
        self.stft_x = torch.tensor(stft_x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return self.scalar_x.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "scalar_x": self.scalar_x[idx],  # (4,)
            "stft_x": self.stft_x[idx],      # (fft_bins,)
            "y": self.y[idx],                # (1,)
        }


class ParmPINN(nn.Module):
    """
    Physics-Informed Neural Network controller core.

    Input:
      - scalar_x: (B, 4) = [time, vertical_accel, thrust, vertical_velocity]
      - stft_x:   (B, fft_bins)

    Output:
      - phi_pred: (B, 1) latent torsional response estimate
      - u_pred:   (B, 1) corrective torque (negative torque reduction, bounded)
    """

    def __init__(self, fft_bins: int, hidden: int = 128):
        super().__init__()
        in_dim = 4 + fft_bins

        self.backbone = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )

        self.phi_head = nn.Linear(hidden, 1)

        # Resonance gate and magnitude head; combined to produce a bounded negative torque reduction.
        self.res_gate = nn.Sequential(nn.Linear(hidden, 1), nn.Sigmoid())  # (0..1)
        self.u_mag = nn.Sequential(nn.Linear(hidden, 1), nn.Sigmoid())     # (0..1)

    def forward(self, scalar_x: torch.Tensor, stft_x: torch.Tensor, u_max: float) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([scalar_x, stft_x], dim=-1)
        h = self.backbone(x)

        phi = self.phi_head(h)
        gate = self.res_gate(h)
        mag = self.u_mag(h)

        # Negative torque reduction: 0 (no reduction) down to -u_max.
        u = -(u_max * gate * mag)
        return phi, u


def _tau_motor_from_thrust(thrust: torch.Tensor, cfg: Config) -> torch.Tensor:
    # thrust: (B, 1) or (B,)
    return thrust * float(cfg.lever_arm_m)


def physics_residual_loss(
    t: torch.Tensor,
    phi: torch.Tensor,
    u: torch.Tensor,
    thrust: torch.Tensor,
    cfg: Config,
) -> torch.Tensor:
    """
    Enforces the rotational equation of motion:

      I * phi_ddot + gamma * phi_dot + k * phi = tau_motor + u

    where tau_motor is estimated from OpenRocket thrust via cfg.lever_arm_m.

    phi_dot and phi_ddot are computed via autograd w.r.t. time.
    """
    # Ensure shapes (B, 1)
    if t.ndim == 1:
        t = t.unsqueeze(-1)
    if thrust.ndim == 1:
        thrust = thrust.unsqueeze(-1)

    # Autograd derivatives (batch-wise)
    ones = torch.ones_like(phi)
    phi_dot = torch.autograd.grad(phi, t, grad_outputs=ones, create_graph=True, retain_graph=True)[0]
    phi_ddot = torch.autograd.grad(phi_dot, t, grad_outputs=torch.ones_like(phi_dot), create_graph=True, retain_graph=True)[0]

    tau_motor = _tau_motor_from_thrust(thrust, cfg)
    lhs = float(cfg.I) * phi_ddot + float(cfg.gamma) * phi_dot + float(cfg.k) * phi
    rhs = tau_motor + u
    residual = lhs - rhs
    return torch.mean(residual ** 2)


def total_loss(batch: Dict[str, torch.Tensor], model: ParmPINN, cfg: Config) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    scalar_x = batch["scalar_x"].to(cfg.device)  # (B, 4)
    stft_x = batch["stft_x"].to(cfg.device)      # (B, fft_bins)
    y = batch["y"].to(cfg.device)                # (B, 1) (placeholder if unavailable)

    # time needs gradients for autograd-based derivatives
    t = scalar_x[:, 0:1].clone().detach().requires_grad_(True)
    accel_z = scalar_x[:, 1:2]
    thrust = scalar_x[:, 2:3]
    v_z = scalar_x[:, 3:4]

    # Replace time column with gradient-enabled t
    scalar_x_g = torch.cat([t, accel_z, thrust, v_z], dim=-1)

    phi_pred, u_pred = model(scalar_x_g, stft_x, u_max=float(cfg.u_max))

    phys = physics_residual_loss(t=t, phi=phi_pred, u=u_pred, thrust=thrust, cfg=cfg)
    data = torch.mean((u_pred - y) ** 2)
    u_mag = torch.mean(u_pred ** 2)

    loss = cfg.lambda_physics * phys + cfg.lambda_data * data + cfg.lambda_u_mag * u_mag

    parts = {"physics": phys.detach(), "data": data.detach(), "u_mag": u_mag.detach()}
    return loss, parts


# -----------------------------
# Training
# -----------------------------

def train_model(model, train_loader, val_loader, cfg):
    model.to(cfg.device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(cfg.epochs):
        model.train()

        train_losses = []
        train_phys = []

        for batch in train_loader:
            optimizer.zero_grad()

            loss, parts = total_loss(batch, model, cfg)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_losses.append(loss.item())
            train_phys.append(parts["physics"].item())

        model.eval()
        val_losses = []
        val_phys = []

        with torch.no_grad():
            for batch in val_loader:
                loss, parts = total_loss(batch, model, cfg)
                val_losses.append(loss.item())
                val_phys.append(parts["physics"].item())

        avg_train = np.mean(train_losses)
        avg_val = np.mean(val_losses)

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_state = model.state_dict()

        if epoch % 10 == 0:
            print(
                f"Epoch {epoch:04d} | "
                f"Train Loss: {avg_train:.6f} | "
                f"Val Loss: {avg_val:.6f} | "
                f"Train Phys: {np.mean(train_phys):.6f} | "
                f"Val Phys: {np.mean(val_phys):.6f}"
            )

    model.load_state_dict(best_state)
    return model


def _parse_openrocket_header(header_line: str) -> List[str]:
    # Header line looks like: "# Time (s),Altitude (m),...,Vertical acceleration (m/s²),...,Thrust (N),..."
    h = header_line.strip()
    if h.startswith("#"):
        h = h[1:].strip()
    return [c.strip() for c in h.split(",")]


def load_openrocket_csv(path: str | Path) -> Dict[str, np.ndarray]:
    """
    Loads a single OpenRocket export CSV and returns the four PARM inputs:
    - time
    - vertical_acceleration
    - thrust
    - vertical_velocity

    The file contains comment lines starting with '#', including the column header.
    """
    path = Path(path)
    header_cols: Optional[List[str]] = None
    rows: List[List[float]] = []

    with path.open("r", newline="", encoding="utf-8") as f:
        for line in f:
            if line.startswith("# Time"):
                header_cols = _parse_openrocket_header(line)
                continue
            if line.startswith("#") or not line.strip():
                continue
            # data row
            parts = [p.strip() for p in line.split(",")]
            # Defensive: some OpenRocket rows may have trailing commas; ignore empty tail
            parts = [p for p in parts if p != ""]
            try:
                rows.append([float(x) if x.lower() != "nan" else float("nan") for x in parts])
            except ValueError:
                # Skip malformed rows (rare; e.g., if OpenRocket changes formatting)
                continue

    if header_cols is None or not rows:
        raise ValueError(f"Could not parse OpenRocket CSV header/data from: {path}")

    data = np.asarray(rows, dtype=np.float32)

    def col(name: str) -> np.ndarray:
        try:
            idx = header_cols.index(name)
        except ValueError as e:
            raise ValueError(f"Missing required column '{name}' in {path.name}") from e
        return data[:, idx]

    t = col("Time (s)")
    v_z = col("Vertical velocity (m/s)")
    a_z = col("Vertical acceleration (m/s²)")
    thrust = col("Thrust (N)")

    # Drop NaNs at the beginning if present (e.g., pre-launch oddities); keep aligned slices.
    mask = np.isfinite(t) & np.isfinite(v_z) & np.isfinite(a_z) & np.isfinite(thrust)
    if not np.all(mask):
        t = t[mask]
        v_z = v_z[mask]
        a_z = a_z[mask]
        thrust = thrust[mask]

    return {"t": t, "v_z": v_z, "a_z": a_z, "thrust": thrust}


def prepare_training_data_from_openrocket_exports(
    export_paths: Sequence[str | Path],
    cfg: Config,
    u_label_by_path: Optional[Dict[str, np.ndarray]] = None,
) -> Dict[str, np.ndarray]:
    """
    Converts one or more OpenRocket CSV exports into a single training matrix.

    If you later produce supervised torque-correction labels (e.g., from a control heuristic),
    provide them in u_label_by_path keyed by path string.
    """
    scalar_all: List[np.ndarray] = []
    stft_all: List[np.ndarray] = []
    y_all: List[np.ndarray] = []

    for p in export_paths:
        d = load_openrocket_csv(p)
        labels = None
        if u_label_by_path is not None:
            labels = u_label_by_path.get(str(p))

        built = build_rolling_samples_from_timeseries(
            t=d["t"],
            accel_z=d["a_z"],
            thrust=d["thrust"],
            v_z=d["v_z"],
            cfg=cfg,
            u_label=labels,
        )
        scalar_all.append(built["scalar_x"])
        stft_all.append(built["stft_x"])
        y_all.append(built["y"])

    return {
        "scalar_x": np.concatenate(scalar_all, axis=0),
        "stft_x": np.concatenate(stft_all, axis=0),
        "y": np.concatenate(y_all, axis=0),
    }


# -----------------------------
# ONNX Export
# -----------------------------

class ParmONNXWrapper(nn.Module):
    """
    Wraps ParmPINN so ONNX export has a simple signature.
    Inputs are the four scalars + precomputed STFT features.
    """

    def __init__(self, core: ParmPINN, cfg: Config):
        super().__init__()
        self.core = core
        self.u_max = float(cfg.u_max)

    def forward(self, scalar_x: torch.Tensor, stft_x: torch.Tensor) -> torch.Tensor:
        _, u = self.core(scalar_x, stft_x, u_max=self.u_max)
        return u


def export_onnx(model: ParmPINN, cfg: Config, path: str = "parm_controller.onnx"):
    model.eval()
    wrapper = ParmONNXWrapper(model, cfg).to(cfg.device).eval()

    dummy_scalar = torch.randn(1, 4, device=cfg.device)
    dummy_stft = torch.randn(1, cfg.fft_bins, device=cfg.device)

    torch.onnx.export(
        wrapper,
        (dummy_scalar, dummy_stft),
        path,
        input_names=["scalar_x", "stft_x"],
        output_names=["torque_correction"],
        dynamic_axes={
            "scalar_x": {0: "batch_size"},
            "stft_x": {0: "batch_size"},
            "torque_correction": {0: "batch_size"},
        },
        opset_version=17,
    )

    print(f"Exported model to {path}")


# -----------------------------
# Main Training Entry
# -----------------------------

def main_openrocket_exports(export_paths: Sequence[str | Path]) -> ParmPINN:
    """
    Train PARM from OpenRocket export CSV(s).
    """
    cfg = Config()
    built = prepare_training_data_from_openrocket_exports(export_paths, cfg)

    scalar_x = built["scalar_x"]
    stft_x = built["stft_x"]
    y = built["y"]

    sx_train, sx_val, fx_train, fx_val, y_train, y_val = train_test_split(
        scalar_x, stft_x, y, test_size=0.2, random_state=42
    )

    train_ds = ParmDataset(sx_train, fx_train, y_train)
    val_ds = ParmDataset(sx_val, fx_val, y_val)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, drop_last=False)

    model = ParmPINN(fft_bins=cfg.fft_bins)
    model = train_model(model, train_loader, val_loader, cfg)

    torch.save(model.state_dict(), "parm_controller.pt")
    export_onnx(model, cfg, path="parm_controller.onnx")
    return model