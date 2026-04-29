"""
Build shuffled train/val/test sample datasets from one or more OpenRocket CSV exports.

Why this exists:
- Do NOT shuffle raw CSV rows (breaks rolling-window FFT features).
- Instead, build rolling-window samples (scalar_x, stft_x, y placeholder), then shuffle/split samples.

Usage (from inside PARM/):
  python -m tools.build_dataset --exports "data\\aggregate\\train.csv"

Outputs (default):
  artifacts/datasets/train.npz
  artifacts/datasets/val.npz
  artifacts/datasets/test.npz
  artifacts/datasets/manifest.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np

from parm import Config, prepare_training_data_from_openrocket_exports  # type: ignore


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
    uniq: List[Path] = []
    for pp in out:
        rp = str(pp.resolve())
        if rp not in seen:
            seen.add(rp)
            uniq.append(pp)
    return uniq


def _split_indices(n: int, val_frac: float, test_frac: float, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if n <= 0:
        return (np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=int))
    if val_frac < 0 or test_frac < 0 or (val_frac + test_frac) >= 1.0:
        raise ValueError("val_frac and test_frac must be >=0 and sum to < 1.0")

    rng = np.random.default_rng(seed)
    idx = np.arange(n, dtype=int)
    rng.shuffle(idx)

    n_test = int(round(n * test_frac))
    n_val = int(round(n * val_frac))
    n_test = max(0, min(n, n_test))
    n_val = max(0, min(n - n_test, n_val))

    test_idx = idx[:n_test]
    val_idx = idx[n_test : n_test + n_val]
    train_idx = idx[n_test + n_val :]
    return train_idx, val_idx, test_idx


def _save_npz(path: Path, scalar_x: np.ndarray, stft_x: np.ndarray, y: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, scalar_x=scalar_x, stft_x=stft_x, y=y)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build shuffled train/val/test sample datasets for PARM.")
    ap.add_argument("--exports", nargs="+", required=True, help="CSV file(s), directory, or glob(s) to exports.")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for sample shuffling/splits.")
    ap.add_argument("--val-frac", type=float, default=0.1, help="Validation fraction (of total samples).")
    ap.add_argument("--test-frac", type=float, default=0.1, help="Test fraction (of total samples).")

    ap.add_argument("--window-size", type=int, default=128)
    ap.add_argument("--fft-bins", type=int, default=64)

    ap.add_argument("--out-dir", type=str, default=str(Path("artifacts") / "datasets"))
    args = ap.parse_args()

    export_paths = _expand_exports(args.exports)
    if not export_paths:
        raise SystemExit("No export CSVs found.")

    cfg = Config(window_size=args.window_size, fft_bins=args.fft_bins)
    built = prepare_training_data_from_openrocket_exports(export_paths, cfg)

    scalar_x = built["scalar_x"]
    stft_x = built["stft_x"]
    y = built["y"]

    n = int(scalar_x.shape[0])
    train_idx, val_idx, test_idx = _split_indices(n=n, val_frac=float(args.val_frac), test_frac=float(args.test_frac), seed=int(args.seed))

    out_dir = Path(args.out_dir)
    _save_npz(out_dir / "train.npz", scalar_x[train_idx], stft_x[train_idx], y[train_idx])
    _save_npz(out_dir / "val.npz", scalar_x[val_idx], stft_x[val_idx], y[val_idx])
    _save_npz(out_dir / "test.npz", scalar_x[test_idx], stft_x[test_idx], y[test_idx])

    manifest = {
        "exports": [str(p.as_posix()) for p in export_paths],
        "config": {"window_size": int(cfg.window_size), "fft_bins": int(cfg.fft_bins)},
        "seed": int(args.seed),
        "val_frac": float(args.val_frac),
        "test_frac": float(args.test_frac),
        "counts": {"samples": n, "train": int(train_idx.size), "val": int(val_idx.size), "test": int(test_idx.size)},
        "files": {
            "train": str((out_dir / "train.npz").as_posix()),
            "val": str((out_dir / "val.npz").as_posix()),
            "test": str((out_dir / "test.npz").as_posix()),
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(json.dumps(manifest["counts"], indent=2))
    print(f"Wrote {out_dir.as_posix()}")


if __name__ == "__main__":
    main()

