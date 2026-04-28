"""
Train the PARM PINN controller from OpenRocket CSV exports.

Usage (PowerShell):
  python -m PARM.train --exports "OpenRocket-Automation\\data\\exports\\*.csv"

Outputs:
  - parm_controller.pt
  - parm_controller.onnx
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from PARM.model import Config, ParmDataset, ParmPINN, export_onnx, prepare_training_data_from_openrocket_exports, train_model

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader


def _expand_exports(patterns: List[str]) -> List[Path]:
    out: List[Path] = []
    for pat in patterns:
        p = Path(pat)
        if any(ch in pat for ch in ["*", "?", "["]):
            out.extend(sorted(Path().glob(pat)))
        elif p.is_dir():
            out.extend(sorted(p.glob("*.csv")))
        else:
            out.append(p)
    # Deduplicate while preserving order
    seen = set()
    uniq = []
    for p in out:
        rp = str(p.resolve())
        if rp not in seen:
            seen.add(rp)
            uniq.append(p)
    return uniq


def main() -> None:
    ap = argparse.ArgumentParser(description="Train PARM controller from OpenRocket exports.")
    ap.add_argument(
        "--exports",
        nargs="+",
        required=True,
        help="CSV file(s), directories, or glob(s) to OpenRocket exports (e.g. data/exports/*.csv).",
    )

    # Core config overrides (keep minimal; edit Config in model.py for deeper tuning)
    ap.add_argument("--window-size", type=int, default=128)
    ap.add_argument("--fft-bins", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)

    ap.add_argument("--I", type=float, default=0.025, help="Torsional inertia [kg*m^2]")
    ap.add_argument("--k", type=float, default=12.0, help="Torsional stiffness [N*m/rad]")
    ap.add_argument("--gamma", type=float, default=0.0, help="Torsional damping [N*m*s/rad]")
    ap.add_argument("--lever-arm-m", type=float, default=0.05, help="Thrust->torque lever arm [m]")
    ap.add_argument("--u-max", type=float, default=5.0, help="Max corrective torque magnitude [N*m]")

    ap.add_argument("--out-pt", type=str, default="parm_controller.pt")
    ap.add_argument("--out-onnx", type=str, default="parm_controller.onnx")

    args = ap.parse_args()

    export_paths = _expand_exports(args.exports)
    if not export_paths:
        raise SystemExit("No export CSVs found.")

    cfg = Config(
        window_size=args.window_size,
        fft_bins=args.fft_bins,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        I=args.I,
        k=args.k,
        gamma=args.gamma,
        lever_arm_m=args.lever_arm_m,
        u_max=args.u_max,
    )

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

    torch.save(model.state_dict(), args.out_pt)
    export_onnx(model, cfg, path=args.out_onnx)

    print(f"Saved PyTorch weights to {args.out_pt}")
    print(f"Saved ONNX model to {args.out_onnx}")
    print(f"Trained from {len(export_paths)} export file(s) and {scalar_x.shape[0]} samples.")


if __name__ == "__main__":
    main()

