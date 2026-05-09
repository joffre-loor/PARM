from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import Config


def clamp_torque_correction(u: float, cfg: Config) -> float:
    """Clamp correction to the supported reduction-only range [-u_max, 0]."""
    return float(np.clip(float(u), -float(cfg.u_max), 0.0))


def apply_torque_deadband(u: float, cfg: Config) -> float:
    """
    Convert tiny model reductions into exactly zero.

    The neural model can output very small negative values near safe windows. For
    the actuator command, those should be treated as no correction.
    """
    u = clamp_torque_correction(u, cfg)
    return 0.0 if abs(u) < float(cfg.u_deadband) else u


def commanded_torque(nominal_torque: float, correction: float, min_torque: float, cfg: Config) -> float:
    """
    Apply a reduction-only correction to nominal torque with final safety clamps.
    """
    u = apply_torque_deadband(correction, cfg)
    return float(np.clip(float(nominal_torque) + u, float(min_torque), float(nominal_torque)))


@dataclass
class TorqueCommandFilter:
    """
    Stateful post-processor for onboard PARM commands.

    Use one instance per control loop. It applies:
    - reduction-only clamp
    - deadband to exact zero
    - low-pass smoothing
    - rate limiting
    """

    cfg: Config
    previous_u: float = 0.0

    def reset(self) -> None:
        self.previous_u = 0.0

    def update(self, raw_u: float, dt_s: float) -> float:
        target = apply_torque_deadband(raw_u, self.cfg)

        alpha = float(np.clip(float(self.cfg.u_filter_alpha), 0.0, 1.0))
        smoothed = self.previous_u + alpha * (target - self.previous_u)

        max_step = max(float(self.cfg.u_rate_limit_per_s), 0.0) * max(float(dt_s), 0.0)
        if max_step > 0.0:
            smoothed = float(np.clip(smoothed, self.previous_u - max_step, self.previous_u + max_step))

        self.previous_u = clamp_torque_correction(smoothed, self.cfg)
        return self.previous_u

    def command_torque(self, nominal_torque: float, raw_u: float, dt_s: float, min_torque: float) -> float:
        filtered_u = self.update(raw_u=raw_u, dt_s=dt_s)
        return float(np.clip(float(nominal_torque) + filtered_u, float(min_torque), float(nominal_torque)))
