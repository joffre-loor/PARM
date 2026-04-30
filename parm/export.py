from __future__ import annotations

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
    )

    print(f"Exported model to {path}")
