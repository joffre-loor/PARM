"""
Split OpenRocket filtered CSVs into PARM train/val/test folders.

Splits by FILE (simulation) to avoid leakage across timesteps from the same trajectory.

Usage (from inside PARM/):
  python -m tools.split_dataset --src "..\\OpenRocket-Automation\\data\\filtered" --dst "data" --seed 42

By default, this COPIES files (safe). Use --move to move instead.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class SplitConfig:
    train: float
    val: float
    test: float
    seed: int
    move: bool


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _list_csvs(src: Path) -> List[Path]:
    if not src.exists():
        raise FileNotFoundError(src)
    if src.is_file():
        return [src]
    return sorted([p for p in src.glob("*.csv") if p.is_file()])


def _split_counts(n: int, train: float, val: float, test: float) -> Tuple[int, int, int]:
    if n <= 0:
        return (0, 0, 0)
    # Deterministic rounding: allocate train, then val, remainder to test
    n_train = int(round(n * train))
    n_val = int(round(n * val))
    n_train = max(0, min(n, n_train))
    n_val = max(0, min(n - n_train, n_val))
    n_test = n - n_train - n_val
    return (n_train, n_val, n_test)


def split_files(files: List[Path], cfg: SplitConfig) -> Dict[str, List[Path]]:
    import random

    rng = random.Random(cfg.seed)
    files = list(files)
    rng.shuffle(files)

    n_train, n_val, n_test = _split_counts(len(files), cfg.train, cfg.val, cfg.test)
    train_files = files[:n_train]
    val_files = files[n_train : n_train + n_val]
    test_files = files[n_train + n_val :]

    assert len(train_files) + len(val_files) + len(test_files) == len(files)
    assert len(test_files) == n_test

    return {"train": train_files, "val": val_files, "test": test_files}


def materialize_split(split: Dict[str, List[Path]], dst_root: Path, move: bool) -> Dict[str, List[Dict]]:
    dst_root.mkdir(parents=True, exist_ok=True)
    manifest: Dict[str, List[Dict]] = {"train": [], "val": [], "test": []}

    op = shutil.move if move else shutil.copy2

    for part, files in split.items():
        part_dir = dst_root / part
        part_dir.mkdir(parents=True, exist_ok=True)
        for src_path in files:
            dst_path = part_dir / src_path.name
            op(str(src_path), str(dst_path))
            manifest[part].append(
                {
                    "file": str(dst_path.as_posix()),
                    "source": str(src_path.as_posix()),
                    "sha256": _sha256_file(dst_path),
                    "bytes": dst_path.stat().st_size,
                }
            )

    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description="Split filtered OpenRocket CSVs into train/val/test.")
    ap.add_argument("--src", required=True, help="Source folder containing filtered CSVs (or a single CSV).")
    ap.add_argument("--dst", default="data", help="Destination data folder (contains train/val/test).")
    ap.add_argument("--train", type=float, default=0.8)
    ap.add_argument("--val", type=float, default=0.1)
    ap.add_argument("--test", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--move", action="store_true", help="Move files instead of copying.")
    args = ap.parse_args()

    total = args.train + args.val + args.test
    if abs(total - 1.0) > 1e-6:
        raise SystemExit(f"--train + --val + --test must sum to 1.0 (got {total})")

    src = Path(args.src)
    dst = Path(args.dst)

    files = _list_csvs(src)
    if not files:
        raise SystemExit(f"No CSV files found in {src}")

    cfg = SplitConfig(train=args.train, val=args.val, test=args.test, seed=args.seed, move=bool(args.move))
    split = split_files(files, cfg)
    manifest = materialize_split(split, dst, move=cfg.move)

    meta = {
        "config": asdict(cfg),
        "source": str(src.resolve().as_posix()),
        "destination": str(dst.resolve().as_posix()),
        "counts": {k: len(v) for k, v in split.items()},
    }

    (dst / "split_manifest.json").write_text(json.dumps({"meta": meta, "files": manifest}, indent=2), encoding="utf-8")

    print("Split complete.")
    print(json.dumps(meta["counts"], indent=2))
    print(f"Manifest written to: {(dst / 'split_manifest.json').as_posix()}")


if __name__ == "__main__":
    main()

