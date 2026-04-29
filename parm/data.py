from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .config import Config
from .features import stft_features_from_window


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
            "stft_x": self.stft_x[idx],  # (fft_bins,)
            "y": self.y[idx],  # (1,)
        }


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


def load_filtered_csv(path: str | Path) -> Dict[str, np.ndarray]:
    """
    Loads a *filtered* OpenRocket CSV (no comment lines) with columns:
      Time (s), Vertical velocity (m/s), Vertical acceleration (m/s²), Thrust (N)

    This matches the simplified exports produced by OpenRocket-Automation's filtered outputs.
    """
    path = Path(path)
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"No header found in: {path}")

        required = ["Time (s)", "Vertical velocity (m/s)", "Vertical acceleration (m/s²)", "Thrust (N)"]
        missing = [c for c in required if c not in reader.fieldnames]
        if missing:
            raise ValueError(f"{path.name} missing columns: {missing}")

        t: List[float] = []
        v_z: List[float] = []
        a_z: List[float] = []
        thrust: List[float] = []

        for row in reader:
            try:
                t.append(float(row["Time (s)"]))
                v_z.append(float(row["Vertical velocity (m/s)"]))
                a_z.append(float(row["Vertical acceleration (m/s²)"]))
                thrust.append(float(row["Thrust (N)"]))
            except (TypeError, ValueError):
                continue

    tt = np.asarray(t, dtype=np.float32)
    vz = np.asarray(v_z, dtype=np.float32)
    az = np.asarray(a_z, dtype=np.float32)
    th = np.asarray(thrust, dtype=np.float32)

    mask = np.isfinite(tt) & np.isfinite(vz) & np.isfinite(az) & np.isfinite(th)
    if not np.all(mask):
        tt, vz, az, th = tt[mask], vz[mask], az[mask], th[mask]

    return {"t": tt, "v_z": vz, "a_z": az, "thrust": th}


def split_filtered_timeseries_on_time_reset(d: Dict[str, np.ndarray]) -> List[Dict[str, np.ndarray]]:
    """
    Splits a filtered timeseries dict into trajectories when time resets/decreases.

    This enables a single "mega CSV" that is a concatenation of many simulations back-to-back,
    without requiring an explicit `sim_id` column.
    """
    t = np.asarray(d["t"], dtype=np.float32)
    if t.size == 0:
        return []

    # Indices where a new trajectory starts (time goes backwards).
    breaks = np.nonzero(t[1:] < t[:-1])[0] + 1
    starts = np.concatenate([np.array([0], dtype=int), breaks])
    ends = np.concatenate([breaks, np.array([t.size], dtype=int)])

    out: List[Dict[str, np.ndarray]] = []
    for s, e in zip(starts, ends):
        if e - s <= 1:
            continue
        out.append(
            {
                "t": d["t"][s:e],
                "a_z": d["a_z"][s:e],
                "thrust": d["thrust"][s:e],
                "v_z": d["v_z"][s:e],
            }
        )
    return out


def load_aggregate_csv(path: str | Path) -> List[Dict[str, np.ndarray]]:
    """
    Loads an aggregated CSV produced by `PARM/aggregate_split.py`:
      sim_id, Time (s), Vertical acceleration (m/s²), Thrust (N), Vertical velocity (m/s)

    Returns a list of trajectories (one per sim_id), so rolling windows do not cross simulations.
    """
    path = Path(path)
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"No header found in: {path}")

        required = ["sim_id", "Time (s)", "Vertical acceleration (m/s²)", "Thrust (N)", "Vertical velocity (m/s)"]
        missing = [c for c in required if c not in reader.fieldnames]
        if missing:
            raise ValueError(f"{path.name} missing columns: {missing}")

        grouped: Dict[str, Dict[str, List[float]]] = {}
        for row in reader:
            sid = (row.get("sim_id") or "").strip()
            if not sid:
                continue
            try:
                t = float(row["Time (s)"])
                a = float(row["Vertical acceleration (m/s²)"])
                th = float(row["Thrust (N)"])
                vz = float(row["Vertical velocity (m/s)"])
            except (TypeError, ValueError):
                continue

            g = grouped.setdefault(sid, {"t": [], "a_z": [], "thrust": [], "v_z": []})
            g["t"].append(t)
            g["a_z"].append(a)
            g["thrust"].append(th)
            g["v_z"].append(vz)

    trajectories: List[Dict[str, np.ndarray]] = []
    for sid, g in grouped.items():
        t = np.asarray(g["t"], dtype=np.float32)
        a_z = np.asarray(g["a_z"], dtype=np.float32)
        thrust = np.asarray(g["thrust"], dtype=np.float32)
        v_z = np.asarray(g["v_z"], dtype=np.float32)

        mask = np.isfinite(t) & np.isfinite(a_z) & np.isfinite(thrust) & np.isfinite(v_z)
        if not np.all(mask):
            t, a_z, thrust, v_z = t[mask], a_z[mask], thrust[mask], v_z[mask]

        # Ensure monotonic time order inside each sim_id
        if t.size > 1:
            order = np.argsort(t)
            t, a_z, thrust, v_z = t[order], a_z[order], thrust[order], v_z[order]

        trajectories.append({"t": t, "a_z": a_z, "thrust": thrust, "v_z": v_z, "sim_id": np.array([sid])})

    return trajectories


def load_any_csv_as_trajectories(path: str | Path) -> List[Dict[str, np.ndarray]]:
    """
    Auto-detects supported CSV formats and returns a list of trajectories.
    Supported:
      - OpenRocket full export (comment-prefixed with '# Time ...')
      - Filtered single-simulation CSV (simple 4-column header)
      - Aggregated split CSV with `sim_id`
    """
    path = Path(path)
    # Peek first non-empty line to detect format
    first = ""
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.strip():
                first = line.strip()
                break

    if first.startswith("#"):
        return [load_openrocket_csv(path)]
    if first.lower().startswith("sim_id,") or first.lower().startswith("sim_id ,"):
        return load_aggregate_csv(path)
    # otherwise treat as filtered/simple csv
    d = load_filtered_csv(path)
    # If this is a concatenation of sims, time often resets (e.g., back to 0). Split into trajectories.
    # If no resets are present, this returns a single trajectory.
    split = split_filtered_timeseries_on_time_reset(d)
    return split if split else [d]


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
        labels = None
        if u_label_by_path is not None:
            labels = u_label_by_path.get(str(p))

        trajectories = load_any_csv_as_trajectories(p)
        for d in trajectories:
            try:
                built = build_rolling_samples_from_timeseries(
                    t=d["t"],
                    accel_z=d["a_z"],
                    thrust=d["thrust"],
                    v_z=d["v_z"],
                    cfg=cfg,
                    u_label=labels,
                )
            except ValueError as e:
                # Common case: very short trajectories (or truncated files) shorter than window size.
                # Skip them so training can proceed.
                if "shorter than rolling window" in str(e):
                    continue
                raise

            scalar_all.append(built["scalar_x"])
            stft_all.append(built["stft_x"])
            y_all.append(built["y"])

    if not scalar_all:
        raise ValueError("No usable trajectories found (all shorter than window_size or empty).")

    return {
        "scalar_x": np.concatenate(scalar_all, axis=0),
        "stft_x": np.concatenate(stft_all, axis=0),
        "y": np.concatenate(y_all, axis=0),
    }

