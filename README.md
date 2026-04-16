# MPC on F1Tenth Gym

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1RgHlU39CTojtIGw5r1O9V6uC9l_DRx5a?usp=sharing)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![CasADi](https://img.shields.io/badge/Solver-CasADi%20%2F%20IPOPT-orange)](https://web.casadi.org/)
[![F1Tenth Gym](https://img.shields.io/badge/Sim-F1Tenth%20Gym-red)](https://f1tenth.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

**Constrained Linear MPC for autonomous racing on the F1Tenth Gym simulator.**  
A pure-Python, ROS-free implementation that formulates a receding-horizon optimal
control problem with track-corridor safety constraints and solves it online using
[CasADi](https://web.casadi.org/) + IPOPT at every control interval.

---

## Table of Contents

1. [Overview](#overview)
2. [Repository Structure](#repository-structure)
3. [Getting Started](#getting-started)
4. [Component Reference](#component-reference)
   - [mpc_config - Hyperparameter Dataclass](#mpc_config--hyperparameter-dataclass)
   - [State - Vehicle State Dataclass](#state--vehicle-state-dataclass)
   - [wrap_angle - Angle Utility](#wrap_angle--angle-utility)
   - [calc_speed_profile - Adaptive Speed Planner](#calc_speed_profile--adaptive-speed-planner)
   - [HeadlessMPC - Core MPC Controller](#headlessmpc--core-mpc-controller)
     - [_load_waypoints](#_load_waypoints)
     - [_load_border_coeffs](#_load_border_coeffs)
     - [_linear_mpc_prob_init](#_linear_mpc_prob_init)
     - [_get_model_matrix](#_get_model_matrix)
     - [update_state](#update_state)
     - [_predict_motion](#_predict_motion)
     - [calc_ref_trajectory](#calc_ref_trajectory)
     - [_linear_mpc_prob_solve](#_linear_mpc_prob_solve)
     - [_linear_mpc_control](#_linear_mpc_control)
     - [update_yaw](#update_yaw)
   - [GymMPCRunner - Simulation Harness](#gymmprunner--simulation-harness)
5. [Data Files](#data-files)
6. [Notes](#NOTES)
7. [Quick Example](#quick-example)

---

## Overview


The pipeline follows a classical receding-horizon loop:

```
waypoints.csv  ──►  HeadlessMPC  ──►  CasADi / IPOPT  ──►  gym action
border_coeffs.csv ──► corridor constraints ──►  safety projection
```

At every MPC interval the controller:

1. Locates the nearest reference point on the pre-loaded raceline.
2. Builds a `TK`-step reference trajectory sampled at spacing `dlk`.
3. Linearises the kinematic bicycle model around the predicted path.
4. Solves the resulting **Quadratic Program** subject to actuation limits and
   optional track-corridor constraints.
5. Applies the first element of the optimal control sequence (`steer`, `speed`).

---


## Repository Structure

```
examples/
├── mpc_gym_colab.py            # Main MPC implementation (this file)
├── waypoints.csv               # Pre-recorded Spielberg raceline  (x, y, yaw)
├── Spielberg_border_coeffs.csv # Track wall / corridor coefficients
├── waypoint_follow.py          # Pure-pursuit baseline
├── run_in_empty_track.py       # Empty-map test helper
├── random_trackgen.py          # Random track generation utility
└── video_recording.py          # Standalone video capture helper
```

---

## Getting Started

### Installation

```bash
pip install f1tenth-gym casadi scipy imageio pillow
```

### Run

```bash
python mpc_gym_colab.py
```

Outputs written to the working directory:

| File | Description |
|---|---|
| `mpc_run.mp4` | Annotated simulation video (5 fps summary) |
| `cte_plot.png` | Cross-track error over simulation time |

---

## Component Reference

### `mpc_config` - Hyperparameter Dataclass

```python
@dataclass
class mpc_config:
    NXK: int = 4        # state dimension  [x, y, v, yaw]
    NU:  int = 2        # control dimension [accel, steer]
    TK:  int = 10       # horizon steps (= 2 s at DTK=0.2 s)
    ...
```

| Parameter | Default | Role |
|---|---|---|
| `NXK` | 4 | State vector size: `[x, y, v, yaw]` |
| `NU` | 2 | Control vector size: `[accel, steer]` |
| `TK` | 10 | MPC horizon (steps) |
| `DTK` | 0.2 s | MPC timestep |
| `dlk` | 0.03 m | Reference waypoint spacing |
| `WB` | 0.33 m | Vehicle wheelbase |
| `Rk` | diag(1,1) | Control effort weight |
| `Rdk` | diag(0.6,0.6) | Control rate-of-change weight |
| `Qk` | diag(50,50,5,10) | Stage state-error weight |
| `Qfk` | diag(500,500,5,50) | Terminal state-error weight |
| `MAX_STEER` | ±0.4189 rad | Steering angle limit |
| `MAX_ACCEL` | 3.0 m/s² | Acceleration limit |
| `CORRIDOR_WIDTH` | 0.33 m | Fixed metric safety buffer inset inward from each wall |
| `NMPC_CTE_THRESHOLD` | `(2.2−0.33)/2` m | \|CTE\| above which NMPC activates |
| `NMPC_HYSTERESIS` | 5 intervals | MPC intervals NMPC stays active after CTE drops below threshold |

---

### `State` - Vehicle State Dataclass

```python
@dataclass
class State:
    x:     float = 0.0   # world-frame X position (m)
    y:     float = 0.0   # world-frame Y position (m)
    delta: float = 0.0   # current steering angle (rad)
    v:     float = 0.0   # longitudinal speed (m/s)
    yaw:   float = 0.0   # heading angle (rad)
```

Mutable container passed through `update_state`, `_predict_motion`, and
`calc_ref_trajectory`.

---

### `wrap_angle` - Angle Utility

```python
def wrap_angle(a):
    return np.arctan2(np.sin(a), np.cos(a))
```

Used throughout the yaw error computation inside `_linear_mpc_prob_solve` to prevent the solver
from seeing artificially large heading errors across the ±π discontinuity.

---

### `calc_speed_profile` - Adaptive Speed Planner

```python
sp = calc_speed_profile(cx, cy, cyaw)
```

Constructs a per-waypoint target speed array using three-zone logic:

| Zone | Condition | Speed |
|---|---|---|
| **Straight** | `abs(Δyaw) ≤ CURVE_THRESHOLD` everywhere nearby | 5.0 m/s |
| **Entry boost** | Within `ENTRY_BOOST_COUNT` steps before a curve | 5.0 m/s |
| **Curve** | `abs(Δyaw) > 0.01` | 2.5 m/s |
| **Exit ramp** | Within `EXIT_RAMP_COUNT` steps after a curve | linearly ramped 2.5 → 5.0 m/s |

The resulting `sp` array is passed as reference speed `ref_traj[2, :]` to the MPC
solver and used as a fallback when IPOPT has not yet produced a valid solution.

---

### `HeadlessMPC` - Core MPC Controller

```python
mpc = HeadlessMPC(
    waypoints_csv    = "waypoints.csv",
    border_coeffs_csv= "Spielberg_border_coeffs.csv",  # optional
)
```

---

#### `_load_waypoints`

```python
wp = self._load_waypoints("waypoints.csv")  # → np.ndarray (N, 3)
```

Reads `x, y, yaw` columns from CSV, then:

1. **Unwraps** the yaw column with `np.unwrap` to remove ±π jumps.
2. **Closes the loop** : If the gap between the last and first point exceeds 5 cm,
   linearly interpolates a bridge segment.
3. **Resamples at 3 cm spacing** along cumulative arc-length so that every
   subsequent index calculation is uniform in distance.

Output shape: `(N, 3)` where columns are `[x, y, yaw]`.

---

#### `_load_border_coeffs`

```python
bc = self._load_border_coeffs("Spielberg_border_coeffs.csv")  # → np.ndarray (N, 4)
```

Reads per-waypoint track boundary data from CSV columns
`a, b, c_center, c_left, c_right`, then:

1. **Resamples** coefficients to match the uniformly-spaced waypoint array using
   `np.interp` on fractional indices.
2. **Computes track wall XY** coordinates by offsetting the raceline along the
   normal vector `(a, b)` by `(c_left − c_center)` and `(c_right − c_center)`.
3. **Computes corridor bounds**  insets `CORRIDOR_WIDTH` metres from each wall
   inward: `tight_cl = c_left − w`, `tight_cr = c_right + w`.  Where the track is
   narrower than `2w` the limit collapses to the raceline center so the constraint
   stays feasible and the car follows the trajectory as closely as geometry allows.

   The reference trajectory itself is set to the **geometric track center**
   (midpoint of `c_left` and `c_right`), not the original optimised raceline.

Return shape: `(N, 4)` — columns `[a, b, tight_cl, tight_cr]`.

---
#### `_linear_mpc_prob_init`

All matrices that change between solves are declared as `ca.Opti.parameter` so IPOPT can be
warm-started without rebuilding the symbolic graph.

**Decision variables:**

| Symbol | Shape | Description |
|---|---|---|
| `xk` | `(NX, T+1)` | State trajectory over the horizon |
| `uk` | `(NU, T)` | Control trajectory over the horizon |

**Parameters (set every solve):**

| Symbol | Shape | Description |
|---|---|---|
| `x0k` | `(NX,)` | Current vehicle state |
| `ref_traj_k` | `(NX, T+1)` | Reference state trajectory |
| `Ak_param` | `(NX·T, NX·T)` | Block-diagonal linearised A matrices |
| `Bk_param` | `(NX·T, NU·T)` | Block-diagonal linearised B matrices |
| `Ck_param` | `(NX·T,)` | Affine linearisation offset |
| `border_{a,b,cl,cr}_k` | `(T+1,)` | Per-step corridor constraints (if enabled) |

**Cost function:**

$$J = u^T R u + e_x^T Q e_x + \Delta u^T R_d \Delta u$$

where $e_x = x - x_\text{ref}$ and $\Delta u$ are consecutive control differences.

**Constraints:**

- Linearised dynamics: $x_{t+1} = A_t x_t + B_t u_t + C_t$
- Acceleration rate: $|\Delta a| \leq 5\ \text{m/s}^2/\text{step}$
- Steering rate: $|\Delta \delta| \leq \dot\delta_\text{max} \cdot \Delta t$
- Speed bounds: $v \in [0,\ 5]\ \text{m/s}$
- Actuation bounds: $|a| \leq 3\ \text{m/s}^2$, $|\delta| \leq 0.4189\ \text{rad}$
- Corridor (optional): $a_k x_k + b_k y_k \in [c_{r,k},\ c_{l,k}]$

---

#### `_get_model_matrix`

```python
A, B, C = mpc._get_model_matrix(v=3.0, phi=0.1, delta=0.05)
```

Returns the first-order Taylor expansion of the kinematic bicycle model around
operating point `(v, phi, delta)`:

$$\dot x = f(x, u) \approx A x + B u + C$$

| Matrix | Non-zero entries | Physical meaning |
|---|---|---|
| `A[0,2]` | $\Delta t \cos\phi$ | Speed → X displacement |
| `A[0,3]` | $-\Delta t\ * v \sin\phi$ | Yaw → X displacement |
| `A[1,2]` | $\Delta t \sin\phi$ | Speed → Y displacement |
| `A[1,3]` | $\Delta t\ * v \cos\phi$ | Yaw → Y displacement |
| `A[3,2]` | $\Delta t * \tan\delta / L$ | Speed → yaw rate |
| `B[2,0]` | $\Delta t$ | Acceleration → speed |
| `B[3,1]` | $\Delta t\* v / (L * \cos^2\delta)$ | Steering → yaw rate |

---

#### `update_state`

```python
state = mpc.update_state(state, a=1.0, delta=0.05)
```

Integrates the **discrete kinematic bicycle model** one timestep forward:

$$x \mathrel{+}= v \cos\psi \cdot \Delta t, \quad
  y \mathrel{+}= v \sin\psi \cdot \Delta t$$
$$\psi \mathrel{+}= \frac{v}{L}\tan\delta \cdot \Delta t, \quad
  v \leftarrow \text{clip}(v + a\ * \Delta t,\ v_\min,\ v_\max)$$

Used inside `_predict_motion` to roll out the warm-start prediction.

---

#### `_predict_motion`

```python
path_predict = mpc._predict_motion(x0, oa, od, xref)
```

Simulates `TK` steps of the bicycle model using the previous MPC solution
`(oa, od)` as inputs, producing a `(NX, TK+1)` predicted state array.
This array is the linearisation point for `_get_model_matrix`, making the QP an
**iterative linear MPC** (sequential linearisation).

---

#### `calc_ref_trajectory`

```python
ref_traj = mpc.calc_ref_trajectory(state, cx, cy, cyaw, sp)
```

Computes the `(NX, TK+1)` reference trajectory from the global waypoint arrays:

1. **Nearest-point search** - searches `N_IND_SEARCH` waypoints ahead of
   `target_ind` to find and advance the closest point.
2. **Look-ahead spacing** - converts current speed to a step-size in waypoint
   indices: `dind = v·DTK / dlk`.
3. **Index list** - wraps around the closed loop using modular arithmetic so the
   car races multiple laps without resetting.

The index list is cached in `_last_ind_list` for use in the corridor constraint
lookup inside `_linear_mpc_prob_solve`.

---

#### `_linear_mpc_prob_solve`

```python
oa, odelta = mpc._linear_mpc_prob_solve(ref_traj, path_predict, x0)
```

Core solve call.  Sequence of operations per interval:


1. **Warm start** - shifts the previous solution one step forward as the initial
   guess via `opti.set_initial`.
2. **Solve** - calls `opti.solve()`; on failure falls back to `opti.debug.value`
   to extract the last iterate.
3. Returns `oa` (acceleration sequence) and `odelta` (steering sequence).

---

#### `_linear_mpc_control`

```python
oa, odelta = mpc._linear_mpc_control(ref_path, x0)
```

Thin wrapper that runs `_predict_motion` on the carry-forward solution first, then
calls `_linear_mpc_prob_solve`.  Called only when the LMPC mode is active.

---

#### `mpc_control`

```python
oa, odelta, mode = mpc.mpc_control(ref_path, x0, cte=cte)
```

Single switching entry-point called by `GymMPCRunner` every MPC interval.  Selects
the active solver based on `|CTE|`:

| Condition | Active solver |
|---|---|
| `\|CTE\| < NMPC_CTE_THRESHOLD` | **LMPC** — fast linearised QP |
| `\|CTE\| ≥ NMPC_CTE_THRESHOLD` | **NMPC** activated; hysteresis counter set to `NMPC_HYSTERESIS` |
| Hysteresis countdown > 0 | Stay on **NMPC** even after CTE drops |

When NMPC is active it runs alone with **no LMPC fallback**.  On an IPOPT failure
it carries forward its own last debug iterate.  The two solvers maintain separate
warm-start caches (`_prev_xk`/`_prev_uk` for LMPC, `_prev_xk_nl`/`_prev_uk_nl`
for NMPC) and never share state.

The NMPC problem enforces the **full nonlinear kinematic bicycle dynamics**
symbolically in CasADi per horizon step — no A/B/C linearisation:

$$x_{t+1} = x_t + v_t \cos\psi_t \cdot \Delta t, \quad
  y_{t+1} = y_t + v_t \sin\psi_t \cdot \Delta t$$
$$v_{t+1} = v_t + a_t \cdot \Delta t, \quad
  \psi_{t+1} = \psi_t + \frac{v_t}{L}\tan\delta_t \cdot \Delta t$$

Returns `(oa, odelta, mode_str)` where `mode_str` is `'LMPC'` or `'NMPC'`.
The active mode is shown on the simulation HUD (green = LMPC, magenta = NMPC)
and logged in `self._log` under the key `'mode'`.


---

#### `update_yaw`

```python
yaw_continuous = mpc.update_yaw(yaw_raw)
```

Maintains a running `yaw_offset` to unwrap the gym's wrapped `(−π, π]` yaw output
into a monotonically-evolving heading angle.

---

### `GymMPCRunner` — Simulation Harness

```python
runner = GymMPCRunner(
    waypoints_csv     = "waypoints.csv",
    map_name          = "Spielberg",
    border_coeffs_csv = "Spielberg_border_coeffs.csv",
)
runner.run(max_sim_seconds=720.0)
runner.save_mp4("mpc_run.mp4", fps=5)
runner.plot_cte("cte_plot.png")
```

Owns the gym environment lifecycle, the MPC call cadence, the frame capture
pipeline, and all output artefacts.

---


## Data Files

### `waypoints.csv`

Three-column CSV with header `x,y,yaw`.  Stores the pre-recorded Spielberg raceline
sampled at the original logging frequency.  `_load_waypoints` resamples it to a
uniform 3 cm arc-length grid internally.

### `Spielberg_border_coeffs.csv`

Seven-column CSV with headers `w_tr_right, w_tr_left,a,b,c_center,c_left,c_right`.  

| Column | Meaning |
|---|---|
| `a`, `b` | Unit normal vector components at each waypoint |
| `c_center` | Signed projection of the raceline onto the normal |
| `c_left` | Signed projection of the left wall onto the normal |
| `c_right` | Signed projection of the right wall onto the normal |


---

## NOTES

### Speed

| Want | Change |
|---|---|
| Faster straight-line speed | Increase `MAX_SPEED` and `sp[straight]` in `calc_speed_profile` |
| Tighter cornering | Lower `CURVE_SPEED` and `CURVE_THRESHOLD` |
| Smoother speed transitions | Increase `EXIT_RAMP_COUNT` |

### Tracking accuracy vs. comfort

| Want | Change |
|---|---|
| Tighter path tracking | Increase `Qk[0,0]`, `Qk[1,1]` (position weights) |
| Smoother steering | Increase `Rdk[1,1]` (steer rate weight) |
| Faster speed convergence | Increase `Qk[2,2]` (speed weight) |
| Sharper heading response | Increase `Qk[3,3]` (yaw weight) |

### Safety corridor

Increasing `CORRIDOR_FRACTION` to a value of 1.0
constrains the car to stay on the raceline exactly.

### Solver performance

| Symptom | Remediation |
|---|---|
| Lack of moving indicator | Prediction horizon needs to dynamically updated |
| Infeasibility | Corridor control based on width of the car |
| Oscillation | Increase `Rdk` (control rate weight) |

---

## Quick Example

```python
from mpc_gym_colab import GymMPCRunner

runner = GymMPCRunner(
    waypoints_csv     = "waypoints.csv",
    map_name          = "Spielberg",
    border_coeffs_csv = "Spielberg_border_coeffs.csv",
)

# Run for 60 simulated seconds
runner.run(max_sim_seconds=60.0)

# Save outputs
runner.save_mp4("mpc_run.mp4", fps=5)
runner.plot_cte("cte_plot.png")
```

Expected console output (first few lines):

```
 Border constraints loaded: 3847 rows
Running MPC in gym for up to 60s sim-time ...
   step   sim_t         x         y   v_obs   v_cmd  steer_deg       CTE
      0    0.00s     0.000     0.000   0.000   5.000       0.00    0.0000
    100    1.00s     1.203     0.041   3.871   4.112       2.31    0.0218
    200    2.00s     3.917    -0.012   4.953   4.890      -0.87   -0.0043
...
Simulation finished.  sim-time=60.00s  wall-time=18.3s  steps=6000  log-entries=6000
CTE plot saved: cte_plot.png
Writing 60 annotated frames → mpc_run.mp4 ...
Saved: mpc_run.mp4  (1.4 MB)
```
