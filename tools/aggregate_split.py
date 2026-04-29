"""
Aggregate split CSVs into one file per split.

Input:
  data/{train,val,test}/*.csv  (filtered OpenRocket outputs)

Output (default):
  data/aggregate/train.csv
  data/aggregate/val.csv
  data/aggregate/test.csv

The output keeps ONLY the 4 PARM columns, in a consistent order:
  Time (s), Vertical acceleration (m/s²), Thrust (N), Vertical velocity (m/s)

Each aggregated file also includes a `sim_id` column (source filename) so you can
trace rows back to a simulation if needed.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List


REQUIRED_COLS = [
    "Time (s)",
    "Vertical acceleration (m/s²)",
    "Thrust (N)",
    "Vertical velocity (m/s)",
]


def _read_rows(path: Path) -> Iterable[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"No header found in {path}")
        missing = [c for c in REQUIRED_COLS if c not in reader.fieldnames]
        if missing:
            raise ValueError(f"{path.name} missing columns: {missing}")
        yield from reader


def aggregate_folder(src_dir: Path, out_path: Path) -> int:
    files = sorted([p for p in src_dir.glob("*.csv") if p.is_file()])
    out_path.parent.mkdir(parents=True, exist_ok=True)

    out_cols = ["sim_id"] + REQUIRED_COLS
    n_rows = 0

    with out_path.open("w", newline="", encoding="utf-8") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=out_cols)
        writer.writeheader()

        for p in files:
            sim_id = p.stem
            for row in _read_rows(p):
                writer.writerow(
                    {
                        "sim_id": sim_id,
                        "Time (s)": row["Time (s)"],
                        "Vertical acceleration (m/s²)": row["Vertical acceleration (m/s²)"],
                        "Thrust (N)": row["Thrust (N)"],
                        "Vertical velocity (m/s)": row["Vertical velocity (m/s)"],
                    }
                )
                n_rows += 1

    return n_rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Aggregate split CSVs into one file per split.")
    ap.add_argument("--data", default="data", help="PARM data root containing train/val/test directories.")
    ap.add_argument("--out", default="data/aggregate", help="Output directory for aggregated CSVs.")
    args = ap.parse_args()

    data_root = Path(args.data)
    out_root = Path(args.out)

    splits = ["train", "val", "test"]
    for split in splits:
        src_dir = data_root / split
        if not src_dir.exists():
            raise SystemExit(f"Missing split folder: {src_dir}")
        out_path = out_root / f"{split}.csv"
        n = aggregate_folder(src_dir, out_path)
        print(f"{split}: wrote {n} rows -> {out_path.as_posix()}")


if __name__ == "__main__":
    main()

