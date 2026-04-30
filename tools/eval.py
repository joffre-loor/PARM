"""
Evaluate a trained PARM controller on a held-out dataset.

Typical usage (from inside PARM/):
  python -m tools.eval --weights "artifacts\\weights\\parm_controller.pt" --exports "data\\aggregate\\test.csv"

This reports:
  - mean total loss (same objective used in training)
  - mean physics residual loss
  - basic stats of the controller output u (torque reduction)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader

from parm import (  # type: ignore
    Config,
    ParmDataset,
    ParmPINN,
    prepare_training_data_from_openrocket_exports,
    total_loss,
)


@torch.no_grad()
def _u_stats(model: ParmPINN, loader: DataLoader, cfg: Config) -> Dict[str, float]:
    # u depends on forward only; no autograd needed here
    model.eval()
    us = []
    for batch in loader:
        scalar_x = batch["scalar_x"].to(cfg.device)
        spectral_x = batch["spectral_x"].to(cfg.device)
        _, u = model(scalar_x, spectral_x, u_max=float(cfg.u_max))
        us.append(u.detach().cpu().numpy().reshape(-1))
    u = np.concatenate(us) if us else np.array([], dtype=np.float32)
    if u.size == 0:
        return {"n": 0}
    return {
        "n": int(u.size),
        "mean": float(np.mean(u)),
        "std": float(np.std(u)),
        "min": float(np.min(u)),
        "max": float(np.max(u)),
        "p01": float(np.quantile(u, 0.01)),
        "p50": float(np.quantile(u, 0.50)),
        "p99": float(np.quantile(u, 0.99)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate PARM model on held-out data.")
    ap.add_argument("--weights", required=True, help="Path to .pt weights saved by train.py")
    ap.add_argument(
        "--exports",
        nargs="+",
        required=True,
        help="CSV(s) / glob(s) to evaluate on (e.g. data/aggregate/test.csv)",
    )
    ap.add_argument("--window-size", type=int, default=128)
    ap.add_argument("--fft-bins", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--u-max", type=float, default=5.0)
    ap.add_argument("--out-json", type=str, default=str(Path("artifacts") / "metrics" / "eval_metrics.json"))
    args = ap.parse_args()

    cfg = Config(
        window_size=args.window_size,
        fft_bins=args.fft_bins,
        batch_size=args.batch_size,
        u_max=args.u_max,
    )

    # Expand inputs similarly to train.py behavior (simple glob support)
    paths = []
    for pat in args.exports:
        p = Path(pat)
        if any(ch in pat for ch in ["*", "?", "["]):
            paths.extend(sorted(Path().glob(pat)))
        elif p.is_dir():
            paths.extend(sorted(p.glob("*.csv")))
        else:
            paths.append(p)
    if not paths:
        raise SystemExit("No eval CSVs found.")

    built = prepare_training_data_from_openrocket_exports(paths, cfg)
    ds = ParmDataset(built["scalar_x"], built["spectral_x"], built["y"])
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, drop_last=False)

    model = ParmPINN(fft_bins=cfg.fft_bins, spectral_feature_dim=cfg.spectral_feature_dim).to(cfg.device)
    state = torch.load(args.weights, map_location=cfg.device)
    model.load_state_dict(state)

    # Loss evaluation requires autograd (physics residual uses time derivatives)
    model.eval()
    losses = []
    phys = []
    with torch.enable_grad():
        for batch in loader:
            loss, parts = total_loss(batch, model, cfg)
            losses.append(float(loss.detach().cpu().item()))
            phys.append(float(parts["physics"].detach().cpu().item()))

    metrics = {
        "files": [str(p.as_posix()) for p in paths],
        "samples": int(len(ds)),
        "mean_loss": float(np.mean(losses)) if losses else None,
        "mean_physics": float(np.mean(phys)) if phys else None,
        "u_stats": _u_stats(model, loader, cfg),
    }

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"Wrote {out_json.as_posix()}")


if __name__ == "__main__":
    main()

