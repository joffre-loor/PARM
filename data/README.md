# PARM datasets

Put your datasets here so the `PARM` repo stays self-contained.

Recommended layout:

- `data/train/`: training CSVs
- `data/val/`: validation CSVs
- `data/test/`: held-out test CSVs

By default, `PARM/.gitignore` ignores `data/` so you can keep large exports out of git.
If you want to version small sample datasets, remove or narrow the ignore rule.

