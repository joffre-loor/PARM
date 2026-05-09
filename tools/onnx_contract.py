"""
Print the PARM ONNX input/output contract and optionally run one inference.

This script does not require onnxruntime for inspection. If onnxruntime is
installed, pass --run-sample to execute a zero-valued sample input.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import onnx


def _shape(value_info: Any) -> list[str | int]:
    dims = []
    for dim in value_info.type.tensor_type.shape.dim:
        if dim.dim_param:
            dims.append(dim.dim_param)
        else:
            dims.append(int(dim.dim_value))
    return dims


def inspect_onnx(path: Path) -> Dict[str, Any]:
    model = onnx.load(path)
    return {
        "path": str(path.as_posix()),
        "ir_version": int(model.ir_version),
        "opset": [{"domain": o.domain or "ai.onnx", "version": int(o.version)} for o in model.opset_import],
        "inputs": [{"name": i.name, "shape": _shape(i)} for i in model.graph.input],
        "outputs": [{"name": o.name, "shape": _shape(o)} for o in model.graph.output],
        "metadata": {p.key: p.value for p in model.metadata_props},
    }


def run_sample(path: Path, scalar_dim: int, spectral_dim: int) -> Dict[str, Any]:
    try:
        import onnxruntime as ort
    except ImportError as e:
        raise SystemExit("onnxruntime is not installed; install it to use --run-sample") from e

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    scalar_x = np.zeros((1, scalar_dim), dtype=np.float32)
    spectral_x = np.zeros((1, spectral_dim), dtype=np.float32)
    out = session.run(None, {"scalar_x": scalar_x, "spectral_x": spectral_x})
    return {"torque_correction": out[0].astype(float).tolist()}


def main() -> None:
    ap = argparse.ArgumentParser(description="Inspect PARM ONNX input/output contract.")
    ap.add_argument("--onnx", default=str(Path("artifacts") / "onnx" / "parm_controller.onnx"))
    ap.add_argument("--run-sample", action="store_true")
    args = ap.parse_args()

    path = Path(args.onnx)
    report = inspect_onnx(path)
    if args.run_sample:
        spectral_dim = int(report["metadata"].get("parm.spectral_feature_dim", 448))
        report["sample_inference"] = run_sample(path, scalar_dim=4, spectral_dim=spectral_dim)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
