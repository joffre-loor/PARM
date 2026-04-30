from __future__ import annotations

from typing import Dict, Tuple

import torch

from .config import Config
from .network import ParmPINN


def _tau_motor_from_thrust(thrust: torch.Tensor, cfg: Config) -> torch.Tensor:
    # thrust: (B, 1) or (B,)
    return thrust * float(cfg.lever_arm_m)


def physics_residual_loss(
    t: torch.Tensor,
    phi: torch.Tensor,
    u: torch.Tensor,
    thrust: torch.Tensor,
    cfg: Config,
) -> torch.Tensor:
    """
    Enforces the rotational equation of motion:

      I * phi_ddot + gamma * phi_dot + k * phi = tau_motor + u

    where tau_motor is estimated from OpenRocket thrust via cfg.lever_arm_m.

    phi_dot and phi_ddot are computed via autograd w.r.t. time.
    """
    # Ensure shapes (B, 1)
    if t.ndim == 1:
        t = t.unsqueeze(-1)
    if thrust.ndim == 1:
        thrust = thrust.unsqueeze(-1)

    # Autograd derivatives (batch-wise)
    ones = torch.ones_like(phi)
    phi_dot = torch.autograd.grad(phi, t, grad_outputs=ones, create_graph=True, retain_graph=True)[0]
    phi_ddot = torch.autograd.grad(
        phi_dot, t, grad_outputs=torch.ones_like(phi_dot), create_graph=True, retain_graph=True
    )[0]

    tau_motor = _tau_motor_from_thrust(thrust, cfg)
    lhs = float(cfg.I) * phi_ddot + float(cfg.gamma) * phi_dot + float(cfg.k) * phi
    rhs = tau_motor + u
    residual = lhs - rhs
    return torch.mean(residual**2)


def total_loss(batch: Dict[str, torch.Tensor], model: ParmPINN, cfg: Config) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    scalar_x = batch["scalar_x"].to(cfg.device)  # (B, 4)
    spectral_x = batch["spectral_x"].to(cfg.device)  # (B, spectral_feature_dim)
    y = batch["y"].to(cfg.device)  # (B, 1) (placeholder if unavailable)

    # time needs gradients for autograd-based derivatives
    t = scalar_x[:, 0:1].clone().detach().requires_grad_(True)
    accel_z = scalar_x[:, 1:2]
    thrust = scalar_x[:, 2:3]
    v_z = scalar_x[:, 3:4]

    # Replace time column with gradient-enabled t
    scalar_x_g = torch.cat([t, accel_z, thrust, v_z], dim=-1)

    phi_pred, u_pred = model(scalar_x_g, spectral_x, u_max=float(cfg.u_max))

    phys = physics_residual_loss(t=t, phi=phi_pred, u=u_pred, thrust=thrust, cfg=cfg)
    data = torch.mean((u_pred - y) ** 2)
    u_mag = torch.mean(u_pred**2)

    loss = cfg.lambda_physics * phys + cfg.lambda_data * data + cfg.lambda_u_mag * u_mag

    parts = {"physics": phys.detach(), "data": data.detach(), "u_mag": u_mag.detach()}
    return loss, parts
