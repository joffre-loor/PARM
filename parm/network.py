from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


class ParmPINN(nn.Module):
    """
    Physics-Informed Neural Network controller core.

    Input:
      - scalar_x: (B, 4) = [time, vertical_accel, thrust, vertical_velocity]
      - spectral_x: (B, spectral_feature_dim)

    Output:
      - phi_pred: (B, 1) latent torsional response estimate
      - u_pred:   (B, 1) corrective torque (negative torque reduction, bounded)
    """

    def __init__(self, fft_bins: int, spectral_feature_dim: int, hidden: int = 128):
        super().__init__()
        in_dim = 4 + int(spectral_feature_dim)

        self.backbone = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )

        self.phi_head = nn.Linear(hidden, 1)

        # Resonance gate and magnitude head; combined to produce a bounded negative torque reduction.
        self.res_gate = nn.Sequential(nn.Linear(hidden, 1), nn.Sigmoid())  # (0..1)
        self.u_mag = nn.Sequential(nn.Linear(hidden, 1), nn.Sigmoid())  # (0..1)

    def forward(
        self, scalar_x: torch.Tensor, spectral_x: torch.Tensor, u_max: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([scalar_x, spectral_x], dim=-1)
        h = self.backbone(x)

        phi = self.phi_head(h)
        gate = self.res_gate(h)
        mag = self.u_mag(h)

        # Negative torque reduction: 0 (no reduction) down to -u_max.
        u = -(u_max * gate * mag)
        return phi, u
