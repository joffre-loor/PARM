# PARM (Predictive Adaptive Resonance Mitigation)

This folder contains the **PARM controller model** implemented as a **Physics-Informed Neural Network (PINN)** for predicting/mitigating resonance by issuing a **temporary torque reduction** command during “danger windows” (approaching phase drift / resonance buildup).

The implementation lives in `PARM/model.py`.

---

## What the model does (conceptual pipeline)

1. **OpenRocket simulations generate training trajectories**
   - A separate automation workflow (outside this folder) runs many OpenRocket simulations with randomized conditions (wind speed/direction, turbulence, etc.).
   - Each export includes time-series flight data.

2. **Extract 4 key PARM inputs from OpenRocket**
   The training/inference inputs are the four parameters described in the writeup:
   - **Time** `t` (s)
   - **Vertical acceleration** `a_z` (m/s²)
   - **Thrust** `F` (N)
   - **Vertical velocity** `v_z` (m/s)

3. **Rolling-window frequency representation (STFT step)**
   - At each timestep \(i\), PARM looks back over a **rolling window** of the most recent vertical acceleration samples:
     \[
       a_z[i-N], \dots, a_z[i-1]
     \]
   - A Short-Time Fourier Transform (STFT)-style feature is computed from the window.
   - In code, this is implemented as a **Hann-windowed FFT of the rolling window** (a single STFT frame) for efficiency:
     - See `stft_features_from_window()` in `model.py`.
   - The result is a compact **magnitude spectrum feature vector** `stft_x` used to detect frequency content associated with resonance/phase drift.

4. **PINN controller produces a corrective torque command**
   - The neural network ingests:
     - the four physical scalars `[t, a_z, F, v_z]`, and
     - the frequency features `stft_x`
   - It outputs a bounded **corrective torque** \(u\) that is interpreted as a **negative torque reduction**:
     - \(u = 0\) → no reduction (baseline thrust/torque)
     - \(u = -u_{max}\) → maximum commanded reduction

5. **Physics-informed training constrains corrections**
   - The controller is trained to remain consistent with the **rotational equation of motion**:
     \[
       I\phï + \gamma\phi̇ + k\phi = \tau_{motor} + u
     \]
   - OpenRocket does not provide torsional states \(\phi, \phi̇, \phï\). Instead, the network learns a **latent torsional response** \(\phi(t)\) and its derivatives are computed via **autograd with respect to time**.
   - This physics residual is the primary training signal (optionally combined with supervised labels if you later add them).

---

## Model architecture (what’s inside `ParmPINN`)

Class: `ParmPINN` in `model.py`

### Inputs

- **`scalar_x`**: shape `(B, 4)`
  - `[time, vertical_acceleration, thrust, vertical_velocity]`
- **`stft_x`**: shape `(B, fft_bins)`
  - windowed-FFT magnitude features from the rolling acceleration window

### Backbone

A feed-forward MLP with `Tanh` activations:

- input dim = `4 + fft_bins`
- several hidden layers (default hidden size 128)

### Heads / outputs

- **`phi_head`**: predicts latent torsional response \(\phi(t)\)
- **resonance “gate” head**: sigmoid output in \((0,1)\)
- **magnitude head**: sigmoid output in \((0,1)\)

The final corrective torque command is:

\[
u = -u_{max} \cdot gate \cdot mag
\]

This creates a naturally bounded controller output and matches the “reduce torque during danger window, then return to baseline” behavior.

---

## Physics-informed loss (PINN training)

The physics loss enforces:

\[
I\phï + \gamma\phi̇ + k\phi - (\tau_{motor} + u) = 0
\]

Implementation details:

- `phi_dot` and `phi_ddot` are computed as derivatives of `phi_pred` w.r.t. time via `torch.autograd.grad`.
- \(\tau_{motor}\) is estimated from OpenRocket thrust using:
  \[
  \tau_{motor} \approx F \cdot r
  \]
  where `r = Config.lever_arm_m` is an effective lever arm (meters).

Optional additional losses:

- **Supervised torque labels**: if you later create a target correction signal (heuristic controller, logged flight data, etc.), you can supply `u_label` and turn on `Config.lambda_data`.
- **Small-control penalty**: `Config.lambda_u_mag` discourages overly aggressive torque reductions.

---

## Data flow (training)

### OpenRocket CSV parsing

`load_openrocket_csv()` reads the OpenRocket export and extracts columns by name:

- `Time (s)`
- `Vertical acceleration (m/s²)`
- `Thrust (N)`
- `Vertical velocity (m/s)`

### Rolling sample construction

`build_rolling_samples_from_timeseries()` creates one training sample per timestep after the rolling window “warms up”:

- `scalar_x[j] = [t[i], a_z[i], thrust[i], v_z[i]]`
- `stft_x[j] = FFT(a_z[i-window_size : i])`

---

## ONNX export (deployment)

`export_onnx()` exports a simple controller signature:

- **Inputs**
  - `scalar_x` (batch, 4)
  - `stft_x` (batch, fft_bins)
- **Output**
  - `torque_correction` (batch, 1)

Important design choice:

- **STFT/FFT is intentionally NOT inside the ONNX graph**.
  - This keeps the exported controller small and predictable.
  - It also reduces inference latency on embedded targets, where FFT can be implemented with a highly optimized DSP library.

---

## How to train (quick start)

From the repository root (recommended: use the CLI trainer in `PARM/train.py`):

```bash
python -m PARM.train --exports "OpenRocket-Automation\\data\\exports\\*.csv"
```

This produces:

- `parm_controller.pt` (PyTorch weights)
- `parm_controller.onnx` (frozen controller for deterministic inference)

You can also train directly from Python using `main_openrocket_exports()` in `model.py`.

---

## What you must tune for your motor / structure

These are the parameters that most directly depend on your physical system:

- **`Config.I`**: torsional inertia (kg·m²)
- **`Config.k`**: torsional stiffness (N·m/rad)
- **`Config.gamma`**: torsional damping (N·m·s/rad)
- **`Config.lever_arm_m`**: maps thrust to equivalent torque (m)
  - Set to the effective moment arm between the thrust line and torsion axis, *or* set to `0` if you supply motor torque directly.
- **`Config.u_max`**: max torque reduction magnitude (N·m)
  - Should reflect actuator limits and allowable torque reduction.
- **`Config.window_size`**: rolling window length (samples)
  - Should span multiple cycles of the resonance you care about.
- **`Config.fft_bins`**: number of frequency bins retained
  - Pick enough resolution to isolate your resonance band without wasting compute.

---

## Current limitations / assumptions

- OpenRocket trajectories are used as training data. Controller quality depends on how accurately OpenRocket represents:
  - thrust curve variations
  - environmental disturbances
  - acceleration dynamics relevant to torsional resonance
- The torque-excitation model is simplified (`tau_motor ≈ thrust * lever_arm_m`). If resonance is primarily driven by **torque ripple** rather than thrust, consider replacing this with a more direct motor torque model or logged torque telemetry.
- The “STFT” step is implemented as a single-frame windowed FFT per timestep, which is a practical approximation of a full multi-frame STFT for embedded friendliness.

