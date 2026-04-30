"""
Train the PARM PINN controller from OpenRocket CSV exports.

Usage (PowerShell, from inside PARM/):
  python -m tools.train --exports "data\\aggregate\\train.csv"

Outputs:
  - artifacts\\weights\\parm_controller.pt
  - artifacts\\onnx\\parm_controller.onnx
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from parm import (  # type: ignore
    Config,
    ParmDataset,
    ParmPINN,
    export_onnx,
    prepare_training_data_from_openrocket_exports,
    train_model,
)

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

    # Core config overrides (keep minimal; edit Config in PARM/parm/config.py for deeper tuning)
    ap.add_argument("--window-size", type=int, default=128)
    ap.add_argument("--fft-bins", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42, help="Random seed for splits/shuffling.")

    ap.add_argument("--val-frac", type=float, default=0.1, help="Validation fraction (of total samples).")
    ap.add_argument("--test-frac", type=float, default=0.1, help="Test fraction (of total samples).")

    ap.add_argument("--I", type=float, default=0.025, help="Torsional inertia [kg*m^2]")
    ap.add_argument("--k", type=float, default=12.0, help="Torsional stiffness [N*m/rad]")
    ap.add_argument("--gamma", type=float, default=0.0, help="Torsional damping [N*m*s/rad]")
    ap.add_argument("--lever-arm-m", type=float, default=0.05, help="Thrust->torque lever arm [m]")
    ap.add_argument("--u-max", type=float, default=5.0, help="Max corrective torque magnitude [N*m]")

    ap.add_argument("--early-stop-patience", type=int, default=25, help="Stop if val loss plateaus for N epochs (0 disables).")
    ap.add_argument("--early-stop-min-delta", type=float, default=1e-4, help="Required val-loss improvement to reset patience.")

    ap.add_argument("--out-pt", type=str, default=str(Path("artifacts") / "weights" / "parm_controller.pt"))
    ap.add_argument("--out-onnx", type=str, default=str(Path("artifacts") / "onnx" / "parm_controller.onnx"))

    args = ap.parse_args()

    if args.val_frac < 0 or args.test_frac < 0 or (args.val_frac + args.test_frac) >= 1.0:
        raise SystemExit("--val-frac and --test-frac must be >=0 and sum to < 1.0")

    export_paths = _expand_exports(args.exports)
    if not export_paths:
        raise SystemExit("No export CSVs found.")

    # Make runs reproducible.
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

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
        early_stop_patience=args.early_stop_patience,
        early_stop_min_delta=args.early_stop_min_delta,
    )

    built = prepare_training_data_from_openrocket_exports(export_paths, cfg)
    scalar_x = built["scalar_x"]
    spectral_x = built["spectral_x"]
    y = built["y"]

    # Random splits happen on the *rolling-window samples* (not raw CSV rows).
    # This keeps time/FFT features meaningful while preventing simulations from being clustered in training.
    test_size = float(args.test_frac)
    val_size = float(args.val_frac)

    sx_tmp, sx_test, fx_tmp, fx_test, y_tmp, y_test = train_test_split(
        scalar_x,
        spectral_x,
        y,
        test_size=test_size,
        random_state=args.seed,
        shuffle=True,
    )

    # val is a fraction of the remaining pool
    val_of_tmp = val_size / (1.0 - test_size) if (1.0 - test_size) > 0 else 0.0
    sx_train, sx_val, fx_train, fx_val, y_train, y_val = train_test_split(
        sx_tmp,
        fx_tmp,
        y_tmp,
        test_size=val_of_tmp,
        random_state=args.seed,
        shuffle=True,
    )

    train_ds = ParmDataset(sx_train, fx_train, y_train)
    val_ds = ParmDataset(sx_val, fx_val, y_val)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, drop_last=False)

    model = ParmPINN(fft_bins=cfg.fft_bins, spectral_feature_dim=cfg.spectral_feature_dim)
    model = train_model(model, train_loader, val_loader, cfg)

    out_pt = Path(args.out_pt)
    out_onnx = Path(args.out_onnx)
    out_pt.parent.mkdir(parents=True, exist_ok=True)
    out_onnx.parent.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), out_pt)
    export_onnx(model, cfg, path=str(out_onnx))

    print(f"Saved PyTorch weights to {out_pt.as_posix()}")
    print(f"Saved ONNX model to {out_onnx.as_posix()}")
    print(
        f"Trained from {len(export_paths)} export file(s) and {scalar_x.shape[0]} samples "
        f"(train={len(train_ds)}, val={len(val_ds)}, test={sx_test.shape[0]})."
    )


if __name__ == "__main__":
    main()
