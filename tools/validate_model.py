"""
Validate a trained PARM model beyond raw PINN loss.

This is an open-loop validation pass: it proves command behavior on held-out
trajectories, not closed-loop physical mitigation after the command is applied.

Typical usage:
  python -m tools.validate_model --weights artifacts\\weights\\parm_controller.pt --exports data\\aggregate\\test.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

from parm import (  # type: ignore
    Config,
    ParmDataset,
    ParmPINN,
    apply_torque_deadband,
    prepare_training_data_from_openrocket_exports,
    risk_score_from_spectral_features,
    total_loss,
)


def _expand_paths(patterns: List[str]) -> List[Path]:
    out: List[Path] = []
    for pat in patterns:
        p = Path(pat)
        if any(ch in pat for ch in ["*", "?", "["]):
            out.extend(sorted(Path().glob(pat)))
        elif p.is_dir():
            out.extend(sorted(p.glob("*.csv")))
        else:
            out.append(p)
    return out


@torch.no_grad()
def _predict_u(model: ParmPINN, loader: DataLoader, cfg: Config) -> np.ndarray:
    model.eval()
    us: List[np.ndarray] = []
    for batch in loader:
        scalar_x = batch["scalar_x"].to(cfg.device)
        spectral_x = batch["spectral_x"].to(cfg.device)
        _, u = model(scalar_x, spectral_x, u_max=float(cfg.u_max))
        us.append(u.detach().cpu().numpy().reshape(-1))
    return np.concatenate(us).astype(np.float32) if us else np.array([], dtype=np.float32)


def _loss_metrics(model: ParmPINN, loader: DataLoader, cfg: Config) -> Dict[str, float | None]:
    losses: List[float] = []
    phys: List[float] = []
    model.eval()
    with torch.enable_grad():
        for batch in loader:
            loss, parts = total_loss(batch, model, cfg)
            losses.append(float(loss.detach().cpu().item()))
            phys.append(float(parts["physics"].detach().cpu().item()))
    return {
        "mean_loss": float(np.mean(losses)) if losses else None,
        "mean_physics": float(np.mean(phys)) if phys else None,
    }


def _u_stats(u: np.ndarray) -> Dict[str, float | int]:
    if u.size == 0:
        return {"n": 0}
    abs_u = np.abs(u)
    return {
        "n": int(u.size),
        "mean": float(np.mean(u)),
        "std": float(np.std(u)),
        "min": float(np.min(u)),
        "max": float(np.max(u)),
        "p01": float(np.quantile(u, 0.01)),
        "p50": float(np.quantile(u, 0.50)),
        "p99": float(np.quantile(u, 0.99)),
        "abs_mean": float(np.mean(abs_u)),
        "abs_std": float(np.std(abs_u)),
        "abs_cv": float(np.std(abs_u) / max(np.mean(abs_u), 1e-9)),
    }


def _conditioned_u_stats(u: np.ndarray, cfg: Config) -> Dict[str, float | int]:
    conditioned = np.asarray([apply_torque_deadband(float(v), cfg) for v in u], dtype=np.float32)
    active = np.abs(conditioned) > 0.0
    stats = _u_stats(conditioned)
    stats.update(
        {
            "active_count": int(np.sum(active)),
            "zero_count": int(conditioned.size - np.sum(active)),
            "active_fraction": float(np.mean(active)) if conditioned.size else 0.0,
            "deadband": float(cfg.u_deadband),
        }
    )
    return stats


def _risk_metrics(spectral_x: np.ndarray, cfg: Config) -> Dict[str, Any]:
    bins = int(cfg.fft_bins)
    mag = spectral_x[:, :bins]
    cross_cos = spectral_x[:, 3 * bins : 4 * bins]
    drift_cos = spectral_x[:, 5 * bins : 6 * bins]

    band_energy = np.max(mag, axis=1)
    cross_alignment = np.max(cross_cos, axis=1)
    phase_stability = np.max(drift_cos, axis=1)
    risk_score = risk_score_from_spectral_features(spectral_x, cfg)
    high_thr = float(np.quantile(risk_score, 0.80))
    low_thr = float(np.quantile(risk_score, 0.20))
    high_mask = risk_score >= high_thr
    low_mask = risk_score <= low_thr

    return {
        "risk_score": risk_score.astype(np.float32),
        "high_mask": high_mask,
        "low_mask": low_mask,
        "thresholds": {"low_p20": low_thr, "high_p80": high_thr},
        "components": {
            "band_energy_mean": float(np.mean(band_energy)),
            "band_energy_p95": float(np.quantile(band_energy, 0.95)),
            "cross_alignment_mean": float(np.mean(cross_alignment)),
            "phase_stability_mean": float(np.mean(phase_stability)),
            "energy_weight": 0.70,
            "cross_alignment_weight": 0.15,
            "phase_stability_weight": 0.15,
        },
    }


def _command_behavior(u: np.ndarray, risk: Dict[str, Any], y: np.ndarray) -> Dict[str, Any]:
    abs_u = np.abs(u)
    abs_y = np.abs(y.reshape(-1))
    risk_score = risk["risk_score"]
    high_mask = risk["high_mask"]
    low_mask = risk["low_mask"]

    high_abs = abs_u[high_mask]
    low_abs = abs_u[low_mask]
    mean_high = float(np.mean(high_abs)) if high_abs.size else None
    mean_low = float(np.mean(low_abs)) if low_abs.size else None
    ratio = None
    delta = None
    if mean_high is not None and mean_low is not None:
        ratio = float(mean_high / max(mean_low, 1e-9))
        delta = float(mean_high - mean_low)

    corr = None
    if u.size > 1 and float(np.std(abs_u)) > 1e-12 and float(np.std(risk_score)) > 1e-12:
        corr = float(np.corrcoef(risk_score, abs_u)[0, 1])

    label_corr = None
    if u.size > 1 and float(np.std(abs_u)) > 1e-12 and float(np.std(abs_y)) > 1e-12:
        label_corr = float(np.corrcoef(abs_y, abs_u)[0, 1])

    label_mae = float(np.mean(np.abs(u - y.reshape(-1)))) if u.size else None

    return {
        "risk_u_correlation": corr,
        "label_u_correlation": label_corr,
        "label_mae": label_mae,
        "mean_abs_label": float(np.mean(abs_y)) if abs_y.size else None,
        "mean_abs_u_high_risk": mean_high,
        "mean_abs_u_low_risk": mean_low,
        "high_vs_low_abs_u_ratio": ratio,
        "high_vs_low_abs_u_delta": delta,
        "high_risk_samples": int(np.sum(high_mask)),
        "low_risk_samples": int(np.sum(low_mask)),
    }


def _verdict(loss: Dict[str, Any], safety: Dict[str, Any], u_stats: Dict[str, Any], behavior: Dict[str, Any]) -> Dict[str, Any]:
    issues: List[str] = []
    warnings: List[str] = []

    if not safety["passed"]:
        issues.extend(safety["violations"])

    if loss["mean_loss"] is None or not np.isfinite(loss["mean_loss"]):
        issues.append("loss is missing or non-finite")

    abs_cv = float(u_stats.get("abs_cv", 0.0))
    if abs_cv < 0.10:
        warnings.append("torque correction is nearly constant; model may not be risk-sensitive enough")

    corr = behavior.get("risk_u_correlation")
    if corr is None:
        warnings.append("could not compute risk/u correlation")
    elif corr < 0.10:
        warnings.append("torque correction is weakly correlated with the heuristic risk score")

    label_corr = behavior.get("label_u_correlation")
    if label_corr is not None and label_corr < 0.50:
        warnings.append("torque correction does not closely follow the heuristic risk labels")

    ratio = behavior.get("high_vs_low_abs_u_ratio")
    if ratio is not None and ratio < 1.10:
        warnings.append("high-risk windows do not receive much more correction than low-risk windows")

    return {
        "passed": not issues,
        "status": "pass_with_warnings" if not issues and warnings else ("pass" if not issues else "fail"),
        "issues": issues,
        "warnings": warnings,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate PARM command behavior on held-out data.")
    ap.add_argument("--weights", required=True, help="Path to trained .pt weights.")
    ap.add_argument("--exports", nargs="+", required=True, help="Held-out CSV(s), directory, or glob(s).")
    ap.add_argument("--window-size", type=int, default=128)
    ap.add_argument("--fft-bins", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--u-max", type=float, default=5.0)
    ap.add_argument("--u-deadband", type=float, default=0.05)
    ap.add_argument("--u-rate-limit-per-s", type=float, default=8.0)
    ap.add_argument("--u-filter-alpha", type=float, default=0.35)
    ap.add_argument("--lambda-physics", type=float, default=1.0)
    ap.add_argument("--lambda-data", type=float, default=2.0)
    ap.add_argument("--lambda-u-mag", type=float, default=2e-2)
    ap.add_argument("--risk-low-quantile", type=float, default=0.60)
    ap.add_argument("--risk-high-quantile", type=float, default=0.95)
    ap.add_argument("--heuristic-u-max-fraction", type=float, default=0.35)
    ap.add_argument("--out-json", default=str(Path("artifacts") / "metrics" / "validation_report.json"))
    args = ap.parse_args()

    cfg = Config(
        window_size=args.window_size,
        fft_bins=args.fft_bins,
        batch_size=args.batch_size,
        u_max=args.u_max,
        u_deadband=args.u_deadband,
        u_rate_limit_per_s=args.u_rate_limit_per_s,
        u_filter_alpha=args.u_filter_alpha,
        lambda_physics=args.lambda_physics,
        lambda_data=args.lambda_data,
        lambda_u_mag=args.lambda_u_mag,
        risk_low_quantile=args.risk_low_quantile,
        risk_high_quantile=args.risk_high_quantile,
        heuristic_u_max_fraction=args.heuristic_u_max_fraction,
    )
    paths = _expand_paths(args.exports)
    if not paths:
        raise SystemExit("No validation CSVs found.")

    built = prepare_training_data_from_openrocket_exports(paths, cfg)
    ds = ParmDataset(built["scalar_x"], built["spectral_x"], built["y"])
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, drop_last=False)

    model = ParmPINN(fft_bins=cfg.fft_bins, spectral_feature_dim=cfg.spectral_feature_dim).to(cfg.device)
    state = torch.load(args.weights, map_location=cfg.device)
    model.load_state_dict(state)

    loss = _loss_metrics(model, loader, cfg)
    u = _predict_u(model, loader, cfg)
    u_stats = _u_stats(u)
    conditioned_u_stats = _conditioned_u_stats(u, cfg)
    risk = _risk_metrics(built["spectral_x"], cfg)
    behavior = _command_behavior(u, risk, built["y"])

    tol = 1e-6
    violations: List[str] = []
    if np.any(u > tol):
        violations.append("positive torque correction found")
    if np.any(u < -float(cfg.u_max) - tol):
        violations.append("torque correction below -u_max found")
    safety = {
        "passed": not violations,
        "violations": violations,
        "positive_count": int(np.sum(u > tol)),
        "below_negative_limit_count": int(np.sum(u < -float(cfg.u_max) - tol)),
    }

    report = {
        "files": [str(p.as_posix()) for p in paths],
        "samples": int(len(ds)),
        "config": {
            "window_size": int(cfg.window_size),
            "fft_bins": int(cfg.fft_bins),
            "spectral_feature_dim": int(cfg.spectral_feature_dim),
            "u_max": float(cfg.u_max),
            "u_deadband": float(cfg.u_deadband),
            "u_rate_limit_per_s": float(cfg.u_rate_limit_per_s),
            "u_filter_alpha": float(cfg.u_filter_alpha),
            "risk_low_quantile": float(cfg.risk_low_quantile),
            "risk_high_quantile": float(cfg.risk_high_quantile),
            "heuristic_u_max_fraction": float(cfg.heuristic_u_max_fraction),
        },
        "loss": loss,
        "safety": safety,
        "u_stats": u_stats,
        "conditioned_u_stats": conditioned_u_stats,
        "risk": {
            "thresholds": risk["thresholds"],
            "components": risk["components"],
        },
        "command_behavior": behavior,
    }
    report["verdict"] = _verdict(loss, safety, u_stats, behavior)

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))
    print(f"Wrote {out_json.as_posix()}")


if __name__ == "__main__":
    main()
