<p align="center">
  <img src="assets/amrita_logo.jpeg" alt="Amrita Vishwa Vidyapeetham" width="450"/>
</p>

<h1 align="center">Koopman-Operator-Based Model Predictive Control for Payload-Carrying and Wind-Disturbed Quadrotors</h1>
<h3 align="center">A Comparative Study Against Classical PID and LQR Control</h3>

<p align="center"><b>Group: CD7 &nbsp;|&nbsp; Course: Drones, Semester 5</b></p>

---

## Team Members

| Name | Roll Number | Email |
|---|---|---|
| Rupanshi Sangwan | CB.SC.U4AIE24262 | cb.sc.u4aie24262@cb.amrita.students.edu |
| Devana Madhavan | CB.SC.U4AIE24213 | cb.sc.u4aie24213@cb.amrita.students.edu |
| Indraneel R | CB.SC.U4AIE24323 | cb.sc.u4aie24323@cb.amrita.students.edu |
| Adithya U K | CB.SC.U4AIE24302 | cb.sc.u4aie24302@cb.amrita.students.edu |

---

## Repository Contents

| Folder / File | Description |
|---|---|
| `data/` | Logged flight data (`nominal`, `windy`, `payload`) and trained Koopman models |
| `src/koopman/` | Data collection and EDMD Koopman model training |
| `src/mpc/` | MPC controller, LQR controller, and scenario runners |
| `src/env_selector/` | Environment (nominal/windy/payload) classifier |
| `src/analysis/` | Metrics computation and validation scripts |
| `results/` | All `.npz` flight logs, tracking plots, and metrics CSVs for MPC, PID, and LQR |
| `results/analysis/` | Final metrics tables and the analysis-specific findings notes |
| `results/plots_updated/` | Corrected, current Z-tracking comparison plots (all 3 controllers, all 6 scenarios) |
| `README.md` | This file - full project report |

---

## Abstract

Model Predictive Control (MPC) built on a Koopman-operator linearization of nonlinear
quadrotor dynamics has been proposed in recent literature as a way to obtain fast,
real-time-capable control without deriving an exact first-principles model of the
system. This project replicates the Koopman-based control approach of Oh et al.
(IEEE Access, 2024)  including their Extended Dynamic Mode Decomposition (EDMD)
system identification and environment-selector architecture  in the
`gym-pybullet-drones` simulation environment, and extends it in two directions absent
from the base paper: (i) systematic disturbance testing under both wind and a
mid-flight payload addition, and (ii) a genuine comparison against classical PID and
Linear-Quadratic-Regulator (LQR) baseline controllers. Through iterative empirical
testing, we identify and diagnose a persistent orientation instability in the
Koopman-MPC controller  the drone tips and loses attitude control consistently
within one second of flight, independent of any applied disturbance. Isolated hover
tests (zero climb, zero disturbance) reproduce the same failure, indicating the
instability is a structural property of the Euler-angle/quaternion-based Koopman
lifting function rather than a consequence of the disturbance scenarios themselves,
consistent with known singularity issues described in SE(3)-structured Koopman-MPC
literature. Under a fair, identical target across all three controllers, PID and LQR
both reliably reach and hold the commanded height across every tested scenario, while
Koopman-MPC does not complete any scenario and does not outperform either baseline on
any tracking metric. We report this as an evidenced limitation of the replicated
approach and identify the specific representational change (rotation-matrix-based
lifting, as opposed to Euler-angle/quaternion-based lifting) that the literature
suggests would resolve it.

---

## 1. Introduction

Multirotor drones are increasingly deployed for applications " package delivery,
inspection, agriculture ” where payload mass and wind conditions vary unpredictably
during flight. Classical model-based controllers (PID, LQR) are simple and robust but
are typically tuned around a fixed nominal model of the vehicle; when the true
dynamics deviate substantially (e.g., due to a sudden mass change), performance can
degrade. Data-driven control offers an alternative: rather than deriving the
dynamics analytically, a model is *learned* directly from input/output flight data.

The Koopman operator provides a principled way to do this: it represents a nonlinear
dynamical system as a (possibly infinite-dimensional) *linear* operator acting on a
space of observable functions of the state. In practice, a finite-dimensional
approximation is obtained via Extended Dynamic Mode Decomposition (EDMD), and the
resulting linear model is used inside a standard, convex Model Predictive Control
(MPC) formulation "” avoiding the nonlinear optimization that a first-principles
model would otherwise require.

**Base paper.** This project replicates and extends Oh, Lee, and Moon (2024),
*"Koopman-Based Control System for Quadrotors in Noisy Environments"* (IEEE Access),
which trains separate Koopman models for nominal and windy flight conditions, uses a
nearest-mean *environment selector* to pick the active model in real time, and drives
an MPC controller with the selected model. The base paper validates its approach
under hover-level attitude tracking and windy conditions only; it does not test
payload disturbances, and it does not compare against a classical control baseline
(its own "baseline" is an ablation of its own selector, not an independent
controller).

**This project's contribution / novelty.**

1. **Payload disturbance testing**  a mid-flight mass addition (not present in the
   base paper), tested independently and in combination with wind.
2. **A genuine classical-control baseline**  PID (via `gym-pybullet-drones`'
   built-in `DSLPIDControl`) and a discrete-time LQR controller, evaluated on the
   *same* six disturbance scenarios as the Koopman-MPC controller, with matched
   targets and a crash-aware metrics pipeline.
3. **An evidenced limitation finding**  through systematic isolation testing, we
   show the Koopman-MPC controller's instability is independent of disturbance type
   or target height, and connect it to a known representational issue in the
   literature.

---

## 2. Methodology

The project pipeline consists of six stages.

### 2.1 Flight Data Collection

Flight data is collected in `gym-pybullet-drones` (CF2X quadrotor model, PyBullet
physics) under three conditions:

- **Nominal**  no external disturbance.
- **Windy**  a constant horizontal force disturbance.
- **Payload**  additional mass introduced mid-flight.

A PID-stabilized controller (`DSLPIDControl`) flies to randomized position/height
setpoints during data collection. This was found necessary empirically: driving the
four motors with independent random RPM noise (a natural first choice for exciting
system dynamics) caused the drone to tumble (roll standard deviation $\sigma_\phi
\approx 0.73$ rad, full $\pm\pi$ excursions) rather than fly "” data collected this way
is unusable for identifying *flight* dynamics. Flying to randomized setpoints under
PID stabilization keeps the drone upright ($\sigma_\phi \approx 0.08$ rad) while still
covering a wide height range for the Koopman model to learn from.

At each timestep the logged state vector is

$$
\mathbf{x} = \begin{bmatrix} x & y & z & q_1 & q_2 & q_3 & q_4 & \phi & \theta & \psi & v_x & v_y & v_z & \omega_x & \omega_y & \omega_z \end{bmatrix}^\top \in \mathbb{R}^{16}
$$

(position, quaternion, roll/pitch/yaw, linear velocity, angular velocity), alongside
the 4-dimensional motor RPM command $\mathbf{u} \in \mathbb{R}^4$.

### 2.2 Koopman Model Training (EDMD)

Let $\mathbf{x}[k]$ be the state at discrete time step $k$. The Koopman operator
$\mathcal{K}$ acts on observable functions $\psi$ of the state such that

$$
\mathcal{K}\,\psi(\mathbf{x}[k]) = \psi(\mathbf{x}[k+1])
$$

In its finite-dimensional, control-affine EDMD approximation, a *lifting function*
$\psi(\mathbf{x})$ maps the raw state into a higher-dimensional space in which the
dynamics are approximately linear:

$$
\psi(\mathbf{x}[k+1]) \approx A\,\psi(\mathbf{x}[k]) + B\,\mathbf{u}[k]
$$

The lifting function used in this project is

$$
\psi(\mathbf{x}) = \Big[\ \mathbf{x},\ \ \mathbf{x}^{\circ 2},\ \ \{x_i x_j\}_{(i,j)\in P}\ \Big]
$$

where $\mathbf{x}^{\circ 2}$ denotes the elementwise square and $P$ is the set of
pairwise cross-terms among the orientation and angular-velocity dimensions (indices
3"“15). Cross-terms were found necessary empirically: elementwise squares alone fit
translational states well but fit rotational states poorly, consistent with the fact
that rigid-body rotational dynamics (Euler's equation, Eq. 3) are inherently coupled
across axes.

$(A, B)$ are obtained via regularized least squares on normalized data:

$$
(A, B) = \arg\min_{A,B} \sum_{k} \left\lVert \psi(\mathbf{x}[k+1]) - A\,\psi(\mathbf{x}[k]) - B\,\mathbf{u}[k] \right\rVert_2^2 + \lambda \lVert [A\ B] \rVert_F^2
$$

with $\lambda = 10^{-4}$. Separate $(A,B)$ pairs are trained independently for the
nominal, windy, and payload conditions (matching the base paper's approach of
training one Koopman model per environment, rather than one model spanning all
conditions).

**Underlying physics (for reference).** Discretely, the translational dynamics
follow Newton's second law and the rotational dynamics follow Euler's rotation
equation:

$$
m\,\dot{\mathbf{v}} = R(\mathbf{q})\,[0,0,T]^\top - m g\,\hat{\mathbf{z}} + \mathbf{F}_{\text{dist}}
$$

$$
I\,\dot{\boldsymbol{\omega}} = \boldsymbol{\tau} - \boldsymbol{\omega} \times (I\,\boldsymbol{\omega})
$$

where $T$ is total thrust, $R(\mathbf{q})$ is the rotation matrix corresponding to
quaternion $\mathbf{q}$, $I$ is the inertia tensor, and $\mathbf{F}_{\text{dist}}$
represents an external disturbance force (e.g. wind). The Koopman/EDMD model does not
use this equation directly  it is learned from data  but it explains why
orientation/angular-velocity cross-terms are needed in the lifting function: the
$\boldsymbol{\omega} \times (I\boldsymbol{\omega})$ term is itself a bilinear coupling
between angular-velocity components.

### 2.3 MPC Controller

At each control step, given the current state **x₀** and a target state **x<sub>ref</sub>**, the controller solves a receding-horizon quadratic program over the lifted dynamics. The lifted trajectory is *substituted directly* into the
cost (rather than carried as a free optimization variable subject to equality
constraints) "” an early formulation using free lifted-state variables was found to be
numerically fragile, causing the QP solver to fail on nearly every call in practice:

$$
\min_{\mathbf{u}_{0:H-1}} \ \sum_{t=0}^{H-1} \Big[\, Q \left\lVert W \odot \big(\psi_{t+1} - \psi_{\text{ref}}\big) \right\rVert_2^2 + R \lVert \mathbf{u}_t \rVert_2^2 + R_\Delta \lVert \mathbf{u}_t - \mathbf{u}_{t-1} \rVert_2^2 \Big]
$$

$$
\text{s.t.} \quad \psi_{t+1} = A\,\psi_t + B\,\mathbf{u}_t, \qquad |\mathbf{u}_t| \le U_{\max}, \qquad |\mathbf{u}_t - \mathbf{u}_{t-1}| \le \Delta U_{\max}
$$

where $W$ is a per-dimension weight vector that penalizes orientation error
(quaternion, roll, pitch) more heavily than other states "” added after empirically
observing that a uniform weight allowed the controller to sacrifice orientation
stability for marginal height-tracking gains, leading to the drone tipping over.
$H = 6$ is the prediction horizon; the QP is solved with `cvxpy`/OSQP at each control
step and only the first control action $\mathbf{u}_0$ is applied (receding-horizon
control), matching the base paper's MPC formulation in structure.

### 2.4 Environment Selector

Following the base paper's Algorithm 1, a nearest-mean classifier identifies the
active environment. A precision vector $\boldsymbol{\lambda}_c$ is computed per
condition $c \in \{\text{nominal}, \text{windy}, \text{payload}\}$ as the mean of the
roll/pitch/yaw and angular-velocity dimensions over that condition's training data.
At runtime, over a rolling 5-second window (240 steps at 48 Hz, matching the base
paper), the selector computes

$$
\hat{c} = \arg\min_{c} \left\lVert \bar{\mathbf{y}}_{\text{window}} - \boldsymbol{\lambda}_c \right\rVert_2
$$

and switches the active Koopman model $(A_{\hat c}, B_{\hat c})$ accordingly.

### 2.5 Disturbance Scenario Testing

Six scenarios are tested, extending the base paper's four:

| # | Scenario | Selector |
|---|---|---|
| 1 | Nominal environment | On |
| 2 | Windy environment | **Off** (ablation) |
| 3 | Windy environment | On |
| 4 | Nominal â†’ windy transition (mid-flight) | On |
| 5 | Payload added mid-flight | On |
| 6 | Payload + wind combined | On |

Scenarios 5 and 6 (payload) are not present in the base paper.

### 2.6 PID / LQR Baseline

A PID baseline uses `gym-pybullet-drones`' built-in `DSLPIDControl`. An LQR baseline
linearizes the height-channel dynamics about hover,

$$
\begin{bmatrix} z \\ v_z \end{bmatrix}_{k+1} = \begin{bmatrix} 1 & \Delta t \\ 0 & 1 \end{bmatrix} \begin{bmatrix} z \\ v_z \end{bmatrix}_k + \begin{bmatrix} 0 \\ \Delta t / m \end{bmatrix} \delta T_k
$$

and solves the discrete algebraic Riccati equation for the optimal gain $K$ minimizing
$\sum_k \mathbf{x}_k^\top Q \mathbf{x}_k + R\,\delta T_k^2$. Both baselines are run
through the *identical* six scenarios, targeting the *identical* height setpoint as
the Koopman-MPC controller "” this consistency was verified explicitly after an
earlier version of the experiment was found to have a target-height mismatch between
controllers (see Â§5).

---

## 3. Results

### 3.1 Final Comparison Metrics

Computed with crash-aware truncation (metrics are computed only up to the first
failure event  ground contact, a roll/pitch flip beyond 90Â°, or a simulator reset 
so that post-crash artifacts do not silently corrupt the reported numbers):

| Scenario | Controller | RMSE (m) | Settling Time (s) | Control Effort (RMS RPM) | Max Transient Dev. (m) | Crashed | Failure Time (s) |
|---|---|---|---|---|---|---|---|
| 1 | MPC | 0.106 | - | 3199.6 | 0.187 | **Yes** | 0.50 |
| 1 | PID | 0.062 | 4.04 | 14.8 | 0.187 | No | - |
| 1 | LQR | **0.060** | 3.92 | 18.5 | 0.187 | No | - |
| 2 | MPC | 0.118 | - | 3321.8 | 0.187 | **Yes** | 0.42 |
| 2 | PID | 0.062 | 4.00 | 951.9 | 0.187 | No | - |
| 2 | LQR | **0.060** | 3.92 | 19.5 | 0.187 | No | - |
| 3 | MPC | 0.118 | - | 3321.8 | 0.187 | **Yes** | 0.42 |
| 3 | PID | 0.062 | 4.00 | 951.9 | 0.187 | No | - |
| 3 | LQR | **0.060** | 3.92 | 19.5 | 0.187 | No | - |
| 4 | MPC | 0.106 | - | 3199.6 | - | **Yes** | 0.50 |
| 4 | PID | 0.062 | 0.00 | 608.0 | 0.003 | No | - |
| 4 | LQR | **0.060** | 0.00 | 18.9 | 0.0001 | No | - |
| 5 | MPC | 0.106 | - | 3199.6 | - | **Yes** | 0.50 |
| 5 | PID | 0.079 | - | 1749.6 | 0.073 | No | - |
| 5 | LQR | **0.062** | - | 1748.4 | 0.027 | No | - |
| 6 | MPC | 0.118 | - | 3321.8 | 0.187 | **Yes** | 0.42 |
| 6 | PID | 0.079 | - | 1853.0 | 0.187 | No | - |
| 6 | LQR | **0.062** | - | 1746.1 | 0.187 | No | - |

**Bold** marks the best (lowest) value per scenario per metric. Across every metric,
in every scenario, PID or LQR wins; Koopman-MPC does not win a single metric in any
scenario, and does not complete any scenario (see completion rate, Â§3.2).

### 3.2 Completion Rate

Koopman-MPC completes only 5"“7% of every 15-second run before failing; PID and LQR
complete 100% of every scenario.

*(Figure: `results/analysis/completion_rate_comparison.png`)*

### 3.3 Z-Tracking Comparison (All Controllers, All Scenarios)

*(Figures: `results/plots_updated/scenario_1_z_tracking.png` through
`scenario_6_z_tracking.png`)*

PID and LQR both climb smoothly to the 0.3 m target and hold steady, including
sensible recovery after mid-flight payload addition (scenario 5/6: PID settles â‰ˆ0.23
m, LQR â‰ˆ0.27 m post-payload  both closer to target than a naively re-tuned
controller would achieve without adaptation). Koopman-MPC climbs briefly, then tips
past 90Â° roll and lands, in every scenario.

### 3.4 Isolated Hover Instability (Root-Cause Diagnostic)

To determine whether the Koopman-MPC failure was specific to disturbance conditions
or aggressive height targets, an isolated test was run: the controller was asked to
hold its own *starting* height (zero commanded climb) with *zero* disturbance
applied. The instability persisted  roll grew unbounded even under this minimal-demand
condition, ruling out disturbance response or target aggressiveness as the cause.

*(Figure: `results/plots/notebook_hover_instability.png`)*

This isolates the failure to the controller/model itself "” specifically, to the
Euler-angle/quaternion-based orientation representation used in the Koopman lifting
function "” rather than to any disturbance-testing methodology in this project.

---

## 4. Discussion

The base paper's Koopman-MPC approach, faithfully replicated here (same state
representation, same EDMD lifting structure, same environment-selector logic), is
reproducible and its data-driven modeling pipeline (Â§2.1-Â§2.2) achieves low one-step
prediction error on position, velocity, and orientation states (normalized RMSE
$\approx 0.19$-$0.27$ across conditions after correcting known excitation-signal
instabilities during data collection). The failure occurs specifically at the *control*
stage: the linear MPC controller, operating on this learned model, cannot maintain
attitude stability over a sustained flight, independent of disturbance type.

This is consistent with a known, published limitation of Euler-angle/quaternion-based
Koopman lifting functions: Narayanan et al. (SE(3) Koopman-MPC, IFAC-PapersOnLine,
2023) specifically avoid this representation, instead lifting with observables built
from the rotation matrix $R \in SO(3)$ directly (e.g. $R\boldsymbol{\omega}$,
$R\boldsymbol{\omega}^{\circ 2}$), citing exactly the kind of orientation singularity
this project's diagnostics independently reproduce. Migrating to that representation
is identified as the concrete next step for this project, though it requires
retraining the Koopman model on a re-derived state representation and is left as
future work to preserve fidelity to the base paper's stated methodology.

---

## 5. Verification and Reproducibility Notes

In the interest of transparency, two significant bugs were found and corrected during
this project's own internal verification process, prior to reporting final results:

1. **Target-height mismatch.** `mpc_controller.py`'s target was found to be
   mismatched (0.1125 m  the drone's own starting height) against the 0.3 m target
   used by the PID/LQR baselines and the metrics pipeline, invalidating any
   RMSE/tracking comparison computed before the fix. Corrected by unifying the target
   to 0.3 m across all three controllers and the metrics script, and regenerating all
   18 scenario result files.
2. **Payload data corruption.** Payload-condition training data was found to have
   roll instability ($\sigma_\phi \approx 1.5$-$2.1$ rad) traced to repeated calls to
   PyBullet's `changeDynamics()` destabilizing the simulator, and separately to the
   randomized height range (0.1-1.2 m) being unachievable for a 37%-overloaded drone.
   Fixed by calling `changeDynamics()` exactly once per mass change and using a
   condition-specific, achievable height range (0.1"“0.6 m) for payload data
   collection; payload data, Koopman model, and environment-selector precision values
   were all regenerated.
3. **Environment selector unreliable on payload.** The environment selector's
   RPY/angular-velocity-only signature was found, through wider empirical testing
   across 12 non-overlapping windows per condition, to be statistically unreliable
   for distinguishing the payload condition from nominal (payload's distance to its
   own precision mean: 0.0216; distance to nominal's precision mean: 0.0223 —
   effectively indistinguishable). This was corrected by adding a normalized thrust
   feature (mean commanded RPM as a fractional deviation from hover) to the
   signature, since a genuinely overloaded drone requires measurably more thrust to
   hold the same attitude. Post-fix, payload's distance to its own mean (0.092) is
   clearly and consistently smaller than its distance to nominal (0.098) or windy
   (0.114) across all 12 test windows. This fix does not alter the headline results
   in §3, since Koopman-MPC crashes at t≈0.42–0.50s in every scenario, before the
   selector's first 5-second classification window ever executes — MPC never lives
   long enough for this bug to have affected its own reported numbers.

All reported results in Â§3 reflect the corrected pipeline. Git tags
`before-target-and-payload-fix` and `corrected-final-results` mark the exact commits
before and after these corrections for full reproducibility.

## 5.1 Preliminary Investigation: Rotation-Matrix-Based Lifting (Future Work, Started)

Following the diagnosis in §4 and the SE(3) Koopman-MPC literature (Narayanan et al.,
2023), a preliminary investigation was carried out into whether replacing the
Euler-angle/quaternion-based lifting function with a rotation-matrix-based one would
resolve the orientation instability. This section reports that investigation
honestly, as a **started but not completed** line of work, not a finished result.

**What was done.** Rotation matrices $R \in SO(3)$ were computed directly from the
existing quaternion data (no new flight data was required for this step, since a
quaternion and its corresponding rotation matrix describe the same physical
orientation). A new, more compact lifting function was built, following the
literature's approach:

$$
\psi_{\text{rotmat}}(\mathbf{x}) = \big[\ \mathbf{p},\ \ \mathbf{v},\ \ \text{vec}(R),\ \ R\boldsymbol{\omega},\ \ R\boldsymbol{\omega}^{\circ 2}\ \big] \in \mathbb{R}^{21}
$$

where $\mathbf{p}$ and $\mathbf{v}$ are position and velocity, $\text{vec}(R)$ is the
flattened $3\times 3$ rotation matrix (9 entries), and $R\boldsymbol{\omega}$,
$R\boldsymbol{\omega}^{\circ 2}$ are the rotation matrix applied to angular velocity
and its elementwise square, respectively — this is a substantially smaller,
physics-motivated lift (21 dimensions) compared to the original 110-dimensional
polynomial expansion. A new Koopman model was trained on this lifting function using
the existing nominal flight data.

**What was found.** Position and velocity prediction remained excellent (RMSE 0.0007
and 0.0302, normalized), consistent with the original model. Overall rotation-matrix
prediction RMSE was 0.1355 — comparable to or better than the original quaternion
model's weaker orientation dimensions. However, when evaluated specifically on the
most tilted 10% of the (calm, PID-stabilized) training data, RMSE increased to 0.2540
— a real, measurable degradation under tilt, though the tilt magnitudes present in
this dataset (up to roughly 8°) are far short of the near-flip orientations
(beyond 90°) observed during the actual MPC failure in §3–4.

**Why this is reported as inconclusive, not as a fix.** The existing training data was
collected under calm, PID-stabilized flight specifically to keep the drone upright
(§2.1), and as a result contains very few examples of extreme tilt. This means the
present test cannot yet distinguish whether the rotation-matrix representation
genuinely avoids the singularity problem under the severe orientations relevant to
the actual failure mode, or whether it too would degrade given data that actually
covers that regime. A properly conclusive test requires training data that
deliberately spans a much wider range of roll/pitch angles, which was not collected
as part of this project.

## 5.2 Future Work

Two concrete, well-scoped next steps follow directly from this project's findings:

1. **Collect extreme-tilt training data and properly re-evaluate the rotation-matrix
   lifting function.** Fly (or safely simulate) trajectories that deliberately cover
   a wide range of roll and pitch angles, well beyond the mild tilts present in the
   current dataset, retrain the rotation-matrix Koopman model on this richer data,
   and re-run the same tilted-subset RMSE evaluation performed in §5.1 to determine
   whether prediction accuracy holds up under genuinely severe orientation, not just
   mild wobble.
2. **If §1 confirms improved accuracy under extreme tilt, rewrite the MPC controller's
   cost function to operate on rotation-matrix error rather than quaternion/Euler-angle
   error.** This requires a genuinely different distance metric between orientations
   (rotation matrices lie on the curved $SO(3)$ manifold, not flat vector space, so
   ordinary Euclidean distance between matrices is not the mathematically correct
   comparison — a manifold-appropriate distance, such as the geodesic distance or the
   trace-based angular distance used in the SE(3) Koopman-MPC literature, would be
   needed), and re-running the full disturbance-scenario comparison (§3) against PID
   and LQR under this new representation to determine whether it resolves the crash
   behavior documented in this report.

---

## 6. Conclusion

This project replicates a Koopman-operator-based MPC control system for quadrotors
(Oh et al., 2024) and extends it with payload disturbance testing and a genuine
classical-control baseline comparison  both absent from the base paper. Under a
rigorously verified, fair experimental setup, the replicated Koopman-MPC controller
is found to be reliably unstable, tipping over within approximately half a second of
flight regardless of disturbance type, while PID and LQR baselines complete every
tested scenario successfully. Isolated testing traces this instability to the
controller's Euler-angle/quaternion-based orientation representation rather than to
any property of the disturbance scenarios themselves, a finding consistent with
published literature on Koopman-MPC representation choices. This is reported as an
honest, evidenced limitation of the replicated approach; migrating to a
rotation-matrix-based Koopman lifting function is identified as the specific,
literature-grounded next step to resolve it.

---

## 7. References

1. Y. Oh, M. H. Lee, and J. Moon, "Koopman-Based Control System for Quadrotors in
   Noisy Environments," *IEEE Access*, vol. 12, pp. 71675"“71684, 2024.
   [https://doi.org/10.1109/ACCESS.2024.3403104](https://doi.org/10.1109/ACCESS.2024.3403104)
2. S. Narayanan et al., "SE(3) Koopman-MPC: Data-driven Learning and Control of
   Quadrotor UAVs," *IFAC-PapersOnLine*, 2023.
3. M. O. Williams, I. G. Kevrekidis, and C. W. Rowley, "A Data"“Driven Approximation
   of the Koopman Operator: Extending Dynamic Mode Decomposition," *Journal of
   Nonlinear Science*, vol. 25, no. 6, pp. 1307"“1346, 2015.


---

<p align="center"><i>Amrita Vishwa Vidyapeetham "” Group CD7 "” Drones, Semester 5</i></p>
