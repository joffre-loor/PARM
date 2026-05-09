from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from .config import Config
from .network import ParmPINN


class ParmONNXWrapper(nn.Module):
    """
    Wraps ParmPINN so ONNX export has a simple signature.
    Inputs are the four scalars + precomputed spectral features.
    """

    def __init__(self, core: ParmPINN, cfg: Config):
        super().__init__()
        self.core = core
        self.u_max = float(cfg.u_max)

    def forward(self, scalar_x: torch.Tensor, spectral_x: torch.Tensor) -> torch.Tensor:
        _, u = self.core(scalar_x, spectral_x, u_max=self.u_max)
        return u


def export_onnx(model: ParmPINN, cfg: Config, path: str = "parm_controller.onnx") -> None:
    model.eval()
    wrapper = ParmONNXWrapper(model, cfg).to(cfg.device).eval()

    dummy_scalar = torch.randn(1, 4, device=cfg.device)
    dummy_spectral = torch.randn(1, cfg.spectral_feature_dim, device=cfg.device)

    torch.onnx.export(
        wrapper,
        (dummy_scalar, dummy_spectral),
        path,
        input_names=["scalar_x", "spectral_x"],
        output_names=["torque_correction"],
        dynamic_axes={
            "scalar_x": {0: "batch_size"},
            "spectral_x": {0: "batch_size"},
            "torque_correction": {0: "batch_size"},
        },
        opset_version=17,
        dynamo=False,
    )

    try:
        import onnx

        onnx_model = onnx.load(path)
        metadata = {
            "parm.inputs": "scalar_x float32[batch,4]; spectral_x float32[batch,spectral_feature_dim]",
            "parm.scalar_x": "[time_s, vertical_acceleration_m_per_s2, thrust_n, vertical_velocity_m_per_s]",
            "parm.spectral_x": "7*fft_bins phase-aware rolling FFT features from acceleration and thrust windows",
            "parm.spectral_feature_dim": str(int(cfg.spectral_feature_dim)),
            "parm.fft_bins": str(int(cfg.fft_bins)),
            "parm.window_size": str(int(cfg.window_size)),
            "parm.output": "torque_correction float32[batch,1], raw negative torque reduction in N*m",
            "parm.postprocess": "apply deadband, smoothing, rate limit, then clamp nominal_torque + correction to [min_torque, nominal_torque]",
        }
        existing = {p.key: p for p in onnx_model.metadata_props}
        for key, value in metadata.items():
            prop = existing.get(key)
            if prop is None:
                prop = onnx_model.metadata_props.add()
                prop.key = key
            prop.value = value
        onnx.save(onnx_model, path)
    except Exception as e:
        meta_path = Path(path).with_suffix(".metadata_warning.txt")
        meta_path.write_text(f"ONNX metadata was not written: {e}\n", encoding="utf-8")

    print(f"Exported model to {path}")
