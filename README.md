# PARM (Predictive Adaptive Resonance Mitigation)

This folder contains the **PARM controller model** implemented as a **Physics-Informed Neural Network (PINN)** for predicting/mitigating resonance by issuing a **temporary torque reduction** command during “danger windows” (approaching phase drift / resonance buildup).

The implementation lives in the `PARM/parm/` package. `PARM/model.py` re-exports the current API surface.

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

3. **Rolling-window spectral representation (STFT step)**
- At each timestep \(i\), PARM looks back over a **rolling window** of the most recent vertical acceleration and thrust samples:
     \[
       a_z[i-N], \dots, a_z[i-1]
     \]
   - A Short-Time Fourier Transform (STFT)-style feature is computed from the windows.
   - In code, this is implemented as a **Hann-windowed FFT of the rolling window** (a single STFT frame) for efficiency:
  - See `spectral_features_from_windows()` in `parm/features.py` (also re-exported from `model.py`).
   - The result is a compact feature vector `spectral_x` used to detect resonance and phase drift:
     - acceleration FFT magnitude
     - acceleration phase as cosine/sine pairs
     - thrust-vs-acceleration cross phase as cosine/sine pairs
     - within-window cross-phase drift as cosine/sine pairs

4. **PINN controller produces a corrective torque command**
   - The neural network ingests:
     - the four physical scalars `[t, a_z, F, v_z]`, and
     - the phase-aware spectral features `spectral_x`
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

Class: `ParmPINN` in `parm/network.py` (also re-exported from `model.py`)

### Inputs

- **`scalar_x`**: shape `(B, 4)`
  - `[time, vertical_acceleration, thrust, vertical_velocity]`
- **`spectral_x`**: shape `(B, Config.spectral_feature_dim)`
  - default shape is `(B, Config.spectral_feature_dim)`, currently `7 * fft_bins`
  - contains acceleration magnitude, acceleration phase, thrust-vs-acceleration cross phase, and within-window phase drift

### Backbone

A feed-forward MLP with `Tanh` activations:

- input dim = `4 + Config.spectral_feature_dim`
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
- `spectral_x[j] = spectral_features_from_windows(a_z[i-window_size : i], thrust[i-window_size : i])`

---

## ONNX export (deployment)

`export_onnx()` exports a simple controller signature:

- **Inputs**
  - `scalar_x` (batch, 4)
  - `spectral_x` (batch, `Config.spectral_feature_dim`)
- **Output**
  - `torque_correction` (batch, 1)

Important design choice:

- **STFT/FFT is intentionally NOT inside the ONNX graph**.
  - This keeps the exported controller small and predictable.
  - It also reduces inference latency on embedded targets, where FFT can be implemented with a highly optimized DSP library.
  - On Jetson, compute the rolling FFT/cross-phase features outside the ONNX model, then pass the fixed feature vector into the controller.

---

## How to train (quick start)

From inside the `PARM/` repo (recommended: use the CLI trainer in `tools/train.py`):

```bash
python -m tools.train --exports "data\\train\\*.csv"
```

If you deleted the per-simulation split files and kept only the aggregated files, you can train from:

```bash
python -m tools.train --exports "data\\aggregate\\train.csv"
```

This produces:

- `artifacts/weights/parm_controller.pt` (PyTorch weights)
- `artifacts/onnx/parm_controller.onnx` (frozen controller for deterministic inference)

PARM will create rolling-window samples and then do randomized train/val/test splits over those samples.
Avoid shuffling raw CSV rows yourself; it breaks the time-based rolling-window FFT features.

### (Optional) Build shuffled sample datasets

If you want to materialize the shuffled/split samples for inspection/debugging:

```bash
python -m tools.build_dataset --exports "data\\aggregate\\train.csv" --seed 42 --val-frac 0.1 --test-frac 0.1
```

This writes:
- `artifacts/datasets/train.npz`
- `artifacts/datasets/val.npz`
- `artifacts/datasets/test.npz`
- `artifacts/datasets/manifest.json`

---

## How to test (offline evaluation)

After training, evaluate on held-out data (recommended: `data/aggregate/test.csv`):

```bash
python -m tools.eval --weights "artifacts\\weights\\parm_controller.pt" --exports "data\\aggregate\\test.csv"
```

This prints and writes `artifacts/metrics/eval_metrics.json` containing:
- mean total loss
- mean physics residual loss
- basic statistics of the controller output (torque reduction)

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
