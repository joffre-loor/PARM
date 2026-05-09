"""
Find and plot a held-out PARM behavior example for README/demo use.

The script scans test trajectories, runs the trained model, applies the same
deadband used by deployment, and picks a trajectory with a clear active window
and mostly-zero correction elsewhere.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

from parm import (  # type: ignore
    Config,
    ParmPINN,
    apply_torque_deadband,
    build_rolling_samples_from_timeseries,
    load_any_csv_as_trajectories,
    risk_score_from_spectral_features,
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
def _predict_u(model: ParmPINN, scalar_x: np.ndarray, spectral_x: np.ndarray, cfg: Config) -> np.ndarray:
    model.eval()
    sx = torch.tensor(scalar_x, dtype=torch.float32, device=cfg.device)
    fx = torch.tensor(spectral_x, dtype=torch.float32, device=cfg.device)
    _, u = model(sx, fx, u_max=float(cfg.u_max))
    return u.detach().cpu().numpy().reshape(-1).astype(np.float32)


def _active_segments(active: np.ndarray) -> List[tuple[int, int]]:
    segments: List[tuple[int, int]] = []
    start = None
    for i, v in enumerate(active):
        if v and start is None:
            start = i
        elif not v and start is not None:
            segments.append((start, i))
            start = None
    if start is not None:
        segments.append((start, len(active)))
    return segments


def _score_candidate(t: np.ndarray, risk: np.ndarray, u_conditioned: np.ndarray) -> Dict[str, Any]:
    active = np.abs(u_conditioned) > 0.0
    active_fraction = float(np.mean(active)) if active.size else 0.0
    if active.size == 0 or not np.any(active):
        return {"score": -1.0, "active_fraction": active_fraction}

    segments = _active_segments(active)
    longest = max((e - s for s, e in segments), default=0)
    active_abs = np.abs(u_conditioned[active])
    inactive_abs = np.abs(u_conditioned[~active]) if np.any(~active) else np.array([0.0], dtype=np.float32)
    risk_active = risk[active]
    risk_inactive = risk[~active] if np.any(~active) else np.array([0.0], dtype=np.float32)

    # Prefer examples that are mostly idle, have a nontrivial active segment,
    # and show higher risk during active windows.
    idle_score = max(0.0, 1.0 - abs(active_fraction - 0.15) / 0.15)
    separation = float(np.mean(risk_active) - np.mean(risk_inactive))
    strength = float(np.quantile(active_abs, 0.95) - np.mean(inactive_abs))
    score = idle_score + 2.0 * separation + 0.25 * strength + min(longest / 80.0, 1.0)

    return {
        "score": float(score),
        "active_fraction": active_fraction,
        "longest_active_samples": int(longest),
        "mean_risk_active": float(np.mean(risk_active)),
        "mean_risk_inactive": float(np.mean(risk_inactive)),
        "p95_active_abs_u": float(np.quantile(active_abs, 0.95)),
    }


def _write_csv(path: Path, t: np.ndarray, risk: np.ndarray, raw_u: np.ndarray, conditioned_u: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = ["time_s,risk_score,raw_torque_correction_nm,conditioned_torque_correction_nm"]
    for vals in zip(t, risk, raw_u, conditioned_u):
        rows.append(f"{float(vals[0]):.6g},{float(vals[1]):.6g},{float(vals[2]):.6g},{float(vals[3]):.6g}")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _risk_components(spectral_x: np.ndarray, cfg: Config) -> Dict[str, np.ndarray]:
    bins = int(cfg.fft_bins)
    mag = spectral_x[:, :bins]
    cross_cos = spectral_x[:, 3 * bins : 4 * bins]
    drift_cos = spectral_x[:, 5 * bins : 6 * bins]

    band_energy = np.max(mag, axis=1)
    cross_alignment = np.max(cross_cos, axis=1)
    phase_stability = np.max(drift_cos, axis=1)

    def norm01(v: np.ndarray) -> np.ndarray:
        lo = float(np.quantile(v, 0.05))
        hi = float(np.quantile(v, 0.95))
        return np.clip((v - lo) / max(hi - lo, 1e-9), 0.0, 1.0).astype(np.float32)

    return {
        "band_energy": band_energy.astype(np.float32),
        "phase_alignment": cross_alignment.astype(np.float32),
        "phase_stability": phase_stability.astype(np.float32),
        "band_energy_norm": norm01(band_energy),
        "phase_alignment_norm": norm01(cross_alignment),
        "phase_stability_norm": norm01(phase_stability),
    }


def _write_plot(path: Path, t: np.ndarray, risk: np.ndarray, raw_u: np.ndarray, conditioned_u: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width = 1000
    height = 520
    left = 70
    right = 70
    top = 54
    bottom = 62
    plot_w = width - left - right
    plot_h = height - top - bottom

    t0 = float(np.min(t))
    t1 = float(np.max(t))
    u_min = min(float(np.min(raw_u)), float(np.min(conditioned_u)), -0.1)
    u_max = 0.02

    def sx(v: float) -> float:
        return left + (float(v) - t0) / max(t1 - t0, 1e-9) * plot_w

    def sy_risk(v: float) -> float:
        return top + (1.0 - np.clip(float(v), 0.0, 1.0)) * plot_h

    def sy_u(v: float) -> float:
        return top + (u_max - float(v)) / max(u_max - u_min, 1e-9) * plot_h

    def polyline(xs: np.ndarray, ys: np.ndarray, yfunc) -> str:
        step = max(1, int(np.ceil(xs.size / 900)))
        pts = " ".join(f"{sx(x):.2f},{yfunc(y):.2f}" for x, y in zip(xs[::step], ys[::step]))
        return pts

    spans = []
    active = np.abs(conditioned_u) > 0.0
    if np.any(active):
        for start, end in _active_segments(active):
            x = sx(float(t[start]))
            w = max(1.0, sx(float(t[end - 1])) - x)
            spans.append(f'<rect x="{x:.2f}" y="{top}" width="{w:.2f}" height="{plot_h}" fill="#d62728" opacity="0.08"/>')

    risk_pts = polyline(t, risk, sy_risk)
    raw_pts = polyline(t, raw_u, sy_u)
    conditioned_pts = polyline(t, conditioned_u, sy_u)

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="white"/>
  <text x="{width / 2:.0f}" y="28" text-anchor="middle" font-family="Arial" font-size="18" fill="#222">PARM held-out test example</text>
  <text x="{width / 2:.0f}" y="48" text-anchor="middle" font-family="Arial" font-size="12" fill="#555">Correction remains zero until risk rises, then commands temporary torque reduction</text>
  {''.join(spans)}
  <rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="none" stroke="#cccccc"/>
  <line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#888"/>
  <line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#888"/>
  <line x1="{left + plot_w}" y1="{top}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#888"/>
  <polyline points="{risk_pts}" fill="none" stroke="#1f77b4" stroke-width="2"/>
  <polyline points="{raw_pts}" fill="none" stroke="#999999" stroke-width="1.3" opacity="0.65"/>
  <polyline points="{conditioned_pts}" fill="none" stroke="#d62728" stroke-width="2.2"/>
  <text x="{left + plot_w / 2:.0f}" y="{height - 18}" text-anchor="middle" font-family="Arial" font-size="13" fill="#333">Time (s)</text>
  <text x="20" y="{top + plot_h / 2:.0f}" transform="rotate(-90 20 {top + plot_h / 2:.0f})" text-anchor="middle" font-family="Arial" font-size="13" fill="#1f77b4">Risk score</text>
  <text x="{width - 20}" y="{top + plot_h / 2:.0f}" transform="rotate(90 {width - 20} {top + plot_h / 2:.0f})" text-anchor="middle" font-family="Arial" font-size="13" fill="#d62728">Torque correction (N*m)</text>
  <text x="{left}" y="{height - 38}" font-family="Arial" font-size="11" fill="#555">{t0:.2f}s</text>
  <text x="{left + plot_w}" y="{height - 38}" text-anchor="end" font-family="Arial" font-size="11" fill="#555">{t1:.2f}s</text>
  <text x="{left - 8}" y="{top + 5}" text-anchor="end" font-family="Arial" font-size="11" fill="#1f77b4">1.0</text>
  <text x="{left - 8}" y="{top + plot_h}" text-anchor="end" font-family="Arial" font-size="11" fill="#1f77b4">0.0</text>
  <text x="{left + plot_w + 8}" y="{sy_u(0):.2f}" font-family="Arial" font-size="11" fill="#d62728">0</text>
  <text x="{left + plot_w + 8}" y="{sy_u(u_min):.2f}" font-family="Arial" font-size="11" fill="#d62728">{u_min:.2f}</text>
  <rect x="{left + plot_w - 220}" y="{top + 12}" width="205" height="68" fill="white" stroke="#dddddd"/>
  <line x1="{left + plot_w - 205}" y1="{top + 30}" x2="{left + plot_w - 170}" y2="{top + 30}" stroke="#1f77b4" stroke-width="2"/>
  <text x="{left + plot_w - 162}" y="{top + 34}" font-family="Arial" font-size="12" fill="#333">Risk score</text>
  <line x1="{left + plot_w - 205}" y1="{top + 50}" x2="{left + plot_w - 170}" y2="{top + 50}" stroke="#999999" stroke-width="1.3"/>
  <text x="{left + plot_w - 162}" y="{top + 54}" font-family="Arial" font-size="12" fill="#333">Raw u</text>
  <line x1="{left + plot_w - 205}" y1="{top + 70}" x2="{left + plot_w - 170}" y2="{top + 70}" stroke="#d62728" stroke-width="2.2"/>
  <text x="{left + plot_w - 162}" y="{top + 74}" font-family="Arial" font-size="12" fill="#333">Conditioned u</text>
</svg>
"""
    path.write_text(svg, encoding="utf-8")


def _write_zoom_plot(
    path: Path,
    t: np.ndarray,
    risk: np.ndarray,
    components: Dict[str, np.ndarray],
    raw_u: np.ndarray,
    conditioned_u: np.ndarray,
) -> Dict[str, float | int]:
    path.parent.mkdir(parents=True, exist_ok=True)
    active = np.abs(conditioned_u) > 0.0
    if np.any(active):
        segments = _active_segments(active)
        start, end = max(segments, key=lambda p: p[1] - p[0])
    else:
        peak = int(np.argmax(risk))
        start, end = max(0, peak - 20), min(len(t), peak + 21)

    pad = max(20, int((end - start) * 0.75))
    z0 = max(0, start - pad)
    z1 = min(len(t), end + pad)

    tz = t[z0:z1]
    risk_z = risk[z0:z1]
    energy_z = components["band_energy_norm"][z0:z1]
    align_z = components["phase_alignment_norm"][z0:z1]
    stable_z = components["phase_stability_norm"][z0:z1]
    raw_z = raw_u[z0:z1]
    cond_z = conditioned_u[z0:z1]

    width = 1000
    height = 620
    left = 76
    right = 82
    top = 58
    bottom = 64
    plot_w = width - left - right
    plot_h = height - top - bottom
    t0 = float(np.min(tz))
    t1 = float(np.max(tz))
    u_min = min(float(np.min(raw_z)), float(np.min(cond_z)), -0.1)
    u_max = 0.02

    def sx(v: float) -> float:
        return left + (float(v) - t0) / max(t1 - t0, 1e-9) * plot_w

    def sy01(v: float) -> float:
        return top + (1.0 - np.clip(float(v), 0.0, 1.0)) * plot_h

    def syu(v: float) -> float:
        return top + (u_max - float(v)) / max(u_max - u_min, 1e-9) * plot_h

    def pts(xs: np.ndarray, ys: np.ndarray, yfunc) -> str:
        return " ".join(f"{sx(x):.2f},{yfunc(y):.2f}" for x, y in zip(xs, ys))

    spans = []
    active_z = np.abs(cond_z) > 0.0
    if np.any(active_z):
        for s, e in _active_segments(active_z):
            x = sx(float(tz[s]))
            w = max(1.0, sx(float(tz[e - 1])) - x)
            spans.append(f'<rect x="{x:.2f}" y="{top}" width="{w:.2f}" height="{plot_h}" fill="#d62728" opacity="0.08"/>')

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="white"/>
  <text x="{width / 2:.0f}" y="30" text-anchor="middle" font-family="Arial" font-size="18" fill="#222">PARM decision-window view</text>
  <text x="{width / 2:.0f}" y="50" text-anchor="middle" font-family="Arial" font-size="12" fill="#555">Correction is based on spectral energy, phase alignment, and phase stability over the rolling window</text>
  {''.join(spans)}
  <rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="none" stroke="#cccccc"/>
  <line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#888"/>
  <line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#888"/>
  <line x1="{left + plot_w}" y1="{top}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#888"/>
  <polyline points="{pts(tz, energy_z, sy01)}" fill="none" stroke="#2ca02c" stroke-width="1.8"/>
  <polyline points="{pts(tz, align_z, sy01)}" fill="none" stroke="#9467bd" stroke-width="1.8"/>
  <polyline points="{pts(tz, stable_z, sy01)}" fill="none" stroke="#ff7f0e" stroke-width="1.8"/>
  <polyline points="{pts(tz, risk_z, sy01)}" fill="none" stroke="#1f77b4" stroke-width="2.6"/>
  <polyline points="{pts(tz, raw_z, syu)}" fill="none" stroke="#999999" stroke-width="1.2" opacity="0.65"/>
  <polyline points="{pts(tz, cond_z, syu)}" fill="none" stroke="#d62728" stroke-width="2.4"/>
  <text x="{left + plot_w / 2:.0f}" y="{height - 18}" text-anchor="middle" font-family="Arial" font-size="13" fill="#333">Time (s)</text>
  <text x="22" y="{top + plot_h / 2:.0f}" transform="rotate(-90 22 {top + plot_h / 2:.0f})" text-anchor="middle" font-family="Arial" font-size="13" fill="#333">Normalized decision signals</text>
  <text x="{width - 24}" y="{top + plot_h / 2:.0f}" transform="rotate(90 {width - 24} {top + plot_h / 2:.0f})" text-anchor="middle" font-family="Arial" font-size="13" fill="#d62728">Torque correction (N*m)</text>
  <text x="{left}" y="{height - 40}" font-family="Arial" font-size="11" fill="#555">{t0:.2f}s</text>
  <text x="{left + plot_w}" y="{height - 40}" text-anchor="end" font-family="Arial" font-size="11" fill="#555">{t1:.2f}s</text>
  <text x="{left - 8}" y="{top + 5}" text-anchor="end" font-family="Arial" font-size="11" fill="#333">1.0</text>
  <text x="{left - 8}" y="{top + plot_h}" text-anchor="end" font-family="Arial" font-size="11" fill="#333">0.0</text>
  <text x="{left + plot_w + 8}" y="{syu(0):.2f}" font-family="Arial" font-size="11" fill="#d62728">0</text>
  <text x="{left + plot_w + 8}" y="{syu(u_min):.2f}" font-family="Arial" font-size="11" fill="#d62728">{u_min:.2f}</text>
  <rect x="{left + plot_w - 250}" y="{top + 12}" width="235" height="128" fill="white" stroke="#dddddd"/>
  <line x1="{left + plot_w - 232}" y1="{top + 31}" x2="{left + plot_w - 197}" y2="{top + 31}" stroke="#2ca02c" stroke-width="1.8"/>
  <text x="{left + plot_w - 188}" y="{top + 35}" font-family="Arial" font-size="12" fill="#333">Resonance-band energy</text>
  <line x1="{left + plot_w - 232}" y1="{top + 52}" x2="{left + plot_w - 197}" y2="{top + 52}" stroke="#9467bd" stroke-width="1.8"/>
  <text x="{left + plot_w - 188}" y="{top + 56}" font-family="Arial" font-size="12" fill="#333">Forcing/response phase</text>
  <line x1="{left + plot_w - 232}" y1="{top + 73}" x2="{left + plot_w - 197}" y2="{top + 73}" stroke="#ff7f0e" stroke-width="1.8"/>
  <text x="{left + plot_w - 188}" y="{top + 77}" font-family="Arial" font-size="12" fill="#333">Phase stability</text>
  <line x1="{left + plot_w - 232}" y1="{top + 94}" x2="{left + plot_w - 197}" y2="{top + 94}" stroke="#1f77b4" stroke-width="2.6"/>
  <text x="{left + plot_w - 188}" y="{top + 98}" font-family="Arial" font-size="12" fill="#333">Combined risk score</text>
  <line x1="{left + plot_w - 232}" y1="{top + 115}" x2="{left + plot_w - 197}" y2="{top + 115}" stroke="#d62728" stroke-width="2.4"/>
  <text x="{left + plot_w - 188}" y="{top + 119}" font-family="Arial" font-size="12" fill="#333">Conditioned correction</text>
</svg>
"""
    path.write_text(svg, encoding="utf-8")
    return {
        "zoom_start_time_s": float(tz[0]),
        "zoom_end_time_s": float(tz[-1]),
        "zoom_samples": int(tz.size),
        "zoom_active_samples": int(np.sum(active_z)),
        "zoom_peak_risk": float(np.max(risk_z)),
        "zoom_min_conditioned_u": float(np.min(cond_z)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate a PARM README showcase example.")
    ap.add_argument("--weights", default=str(Path("artifacts") / "weights" / "parm_controller.pt"))
    ap.add_argument("--exports", nargs="+", default=[str(Path("data") / "aggregate" / "test.csv")])
    ap.add_argument("--window-size", type=int, default=128)
    ap.add_argument("--fft-bins", type=int, default=64)
    ap.add_argument("--u-max", type=float, default=5.0)
    ap.add_argument("--u-deadband", type=float, default=0.05)
    ap.add_argument("--out-dir", default=str(Path("artifacts") / "examples"))
    args = ap.parse_args()

    cfg = Config(window_size=args.window_size, fft_bins=args.fft_bins, u_max=args.u_max, u_deadband=args.u_deadband)
    model = ParmPINN(fft_bins=cfg.fft_bins, spectral_feature_dim=cfg.spectral_feature_dim).to(cfg.device)
    model.load_state_dict(torch.load(args.weights, map_location=cfg.device))

    candidates: List[Dict[str, Any]] = []
    for path in _expand_paths(args.exports):
        for traj_idx, traj in enumerate(load_any_csv_as_trajectories(path)):
            if len(traj["t"]) <= cfg.window_size:
                continue
            try:
                built = build_rolling_samples_from_timeseries(
                    t=traj["t"], accel_z=traj["a_z"], thrust=traj["thrust"], v_z=traj["v_z"], cfg=cfg
                )
            except ValueError:
                continue

            t = built["scalar_x"][:, 0]
            risk = risk_score_from_spectral_features(built["spectral_x"], cfg)
            raw_u = _predict_u(model, built["scalar_x"], built["spectral_x"], cfg)
            conditioned_u = np.asarray([apply_torque_deadband(float(v), cfg) for v in raw_u], dtype=np.float32)
            score = _score_candidate(t, risk, conditioned_u)
            if score["score"] < 0.0:
                continue
            sid = str(traj.get("sim_id", np.array([f"trajectory_{traj_idx}"]))[0])
            candidates.append(
                {
                    "source": str(path.as_posix()),
                    "sim_id": sid,
                    "trajectory_index": int(traj_idx),
                    "score": score,
                    "t": t,
                    "risk": risk,
                    "components": _risk_components(built["spectral_x"], cfg),
                    "raw_u": raw_u,
                    "conditioned_u": conditioned_u,
                }
            )

    if not candidates:
        raise SystemExit("No usable showcase candidate found.")

    best = max(candidates, key=lambda c: c["score"]["score"])
    out_dir = Path(args.out_dir)
    png_path = out_dir / "parm_showcase_example.svg"
    zoom_path = out_dir / "parm_showcase_decision_window.svg"
    csv_path = out_dir / "parm_showcase_example.csv"
    json_path = out_dir / "parm_showcase_example.json"

    _write_plot(png_path, best["t"], best["risk"], best["raw_u"], best["conditioned_u"])
    zoom_summary = _write_zoom_plot(
        zoom_path, best["t"], best["risk"], best["components"], best["raw_u"], best["conditioned_u"]
    )
    _write_csv(csv_path, best["t"], best["risk"], best["raw_u"], best["conditioned_u"])

    active = np.abs(best["conditioned_u"]) > 0.0
    summary = {
        "source": best["source"],
        "sim_id": best["sim_id"],
        "samples": int(best["t"].size),
        "active_samples": int(np.sum(active)),
        "zero_samples": int(best["t"].size - np.sum(active)),
        "active_fraction": float(np.mean(active)),
        "risk_mean_active": float(np.mean(best["risk"][active])) if np.any(active) else None,
        "risk_mean_inactive": float(np.mean(best["risk"][~active])) if np.any(~active) else None,
        "raw_u_min": float(np.min(best["raw_u"])),
        "raw_u_median": float(np.median(best["raw_u"])),
        "conditioned_u_min": float(np.min(best["conditioned_u"])),
        "conditioned_u_median": float(np.median(best["conditioned_u"])),
        "conditioned_u_max": float(np.max(best["conditioned_u"])),
        "plot": str(png_path.as_posix()),
        "decision_window_plot": str(zoom_path.as_posix()),
        "csv": str(csv_path.as_posix()),
        **zoom_summary,
    }
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"Wrote {png_path.as_posix()}")
    print(f"Wrote {zoom_path.as_posix()}")
    print(f"Wrote {csv_path.as_posix()}")
    print(f"Wrote {json_path.as_posix()}")


if __name__ == "__main__":
    main()
