"""
Minimal Jetson-style PARM runtime wrapper.

This script shows the deployment pattern:
1. Keep rolling acceleration/thrust windows in flight software.
2. Convert those windows to `spectral_x`.
3. Run the ONNX controller.
4. Apply deadband/smoothing/rate-limit before changing torque.

Install runtime dependency on Jetson:
  pip install onnxruntime

For NVIDIA TensorRT deployment, keep this same wrapper shape but replace the
`OnnxParmRuntime._run_onnx()` implementation with a TensorRT engine call.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Deque, Optional

import numpy as np

from parm import Config, TorqueCommandFilter, spectral_features_from_windows


class OnnxParmRuntime:
    """
    Stateful PARM runtime wrapper for a flight-control loop.

    The ONNX model is stateless. This class owns the sliding windows and command
    conditioning so the rest of flight software can call one `update(...)` method.
    """

    def __init__(
        self,
        onnx_path: str | Path = Path("artifacts") / "onnx" / "parm_controller.onnx",
        cfg: Optional[Config] = None,
        min_torque: float = 0.0,
    ):
        try:
            import onnxruntime as ort
        except ImportError as e:
            raise RuntimeError("onnxruntime is required for this sample runtime: pip install onnxruntime") from e

        self.cfg = cfg or Config()
        self.min_torque = float(min_torque)
        self.accel_window: Deque[float] = deque(maxlen=int(self.cfg.window_size))
        self.thrust_window: Deque[float] = deque(maxlen=int(self.cfg.window_size))
        self.command_filter = TorqueCommandFilter(self.cfg)

        self.session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        self.output_name = "torque_correction"

    @property
    def ready(self) -> bool:
        return len(self.accel_window) == int(self.cfg.window_size) and len(self.thrust_window) == int(self.cfg.window_size)

    def reset(self) -> None:
        self.accel_window.clear()
        self.thrust_window.clear()
        self.command_filter.reset()

    def update(
        self,
        *,
        time_s: float,
        vertical_acceleration_mps2: float,
        thrust_n: float,
        vertical_velocity_mps: float,
        nominal_torque_nm: float,
        dt_s: float,
    ) -> dict[str, float | bool]:
        """
        Add the latest sample and return the current PARM command state.

        Before the rolling windows fill, PARM returns zero correction.
        """
        self.accel_window.append(float(vertical_acceleration_mps2))
        self.thrust_window.append(float(thrust_n))

        if not self.ready:
            return {
                "ready": False,
                "raw_torque_correction_nm": 0.0,
                "conditioned_torque_correction_nm": 0.0,
                "commanded_torque_nm": float(np.clip(nominal_torque_nm, self.min_torque, nominal_torque_nm)),
            }

        scalar_x = np.asarray(
            [[time_s, vertical_acceleration_mps2, thrust_n, vertical_velocity_mps]],
            dtype=np.float32,
        )
        spectral_x = spectral_features_from_windows(
            accel_window=np.asarray(self.accel_window, dtype=np.float32),
            thrust_window=np.asarray(self.thrust_window, dtype=np.float32),
            cfg=self.cfg,
        ).reshape(1, -1)

        raw_u = self._run_onnx(scalar_x=scalar_x, spectral_x=spectral_x)
        conditioned_u = self.command_filter.update(raw_u=raw_u, dt_s=dt_s)
        commanded_torque = float(np.clip(float(nominal_torque_nm) + conditioned_u, self.min_torque, float(nominal_torque_nm)))

        return {
            "ready": True,
            "raw_torque_correction_nm": float(raw_u),
            "conditioned_torque_correction_nm": float(conditioned_u),
            "commanded_torque_nm": commanded_torque,
        }

    def _run_onnx(self, *, scalar_x: np.ndarray, spectral_x: np.ndarray) -> float:
        output = self.session.run(
            [self.output_name],
            {
                "scalar_x": scalar_x.astype(np.float32, copy=False),
                "spectral_x": spectral_x.astype(np.float32, copy=False),
            },
        )[0]
        return float(output[0, 0])


def _demo_without_hardware() -> None:
    """
    Tiny synthetic demo for local smoke testing after installing onnxruntime.
    Replace this with actual sensor reads in flight software.
    """
    runtime = OnnxParmRuntime()
    dt_s = 0.01

    for i in range(180):
        t = i * dt_s
        accel = 2.0 * np.sin(2.0 * np.pi * 18.0 * t)
        thrust = 120.0 + 5.0 * np.sin(2.0 * np.pi * 18.0 * t + 0.4)
        velocity = 40.0
        nominal_torque = 5.0

        state = runtime.update(
            time_s=t,
            vertical_acceleration_mps2=float(accel),
            thrust_n=float(thrust),
            vertical_velocity_mps=velocity,
            nominal_torque_nm=nominal_torque,
            dt_s=dt_s,
        )

        if state["ready"] and i % 10 == 0:
            print(state)


if __name__ == "__main__":
    _demo_without_hardware()
