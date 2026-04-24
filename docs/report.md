# Robust Model Predictive Tracking Guidance for Planetary Entry: Integral Action, Sliding-Mode, and Chance-Constrained Formulations

**Draft — AIAA SciTech / AAS/AIAA Astrodynamics Specialist Conference Style**

---

## Abstract

Precision atmospheric entry guidance remains a fundamental challenge in planetary mission design, where unmodeled aerodynamic uncertainties, atmospheric variability, and tight terminal constraints must be reconciled with real-time computational budgets. This paper presents and comparatively evaluates a family of model predictive tracking guidance (MPG) laws for six-degree-of-freedom atmospheric entry, each sharing a common receding-horizon quadratic programming (QP) framework but differing in their objective function structure. Starting from the baseline MPG law of Davami, Lu, and Rosengren (2025), which minimizes a weighted sum of linearized tracking errors along a shrinking prediction horizon, three novel formulations are developed: (i) MPG with integral action (MPG-IA), which augments the cost with a penalty on the accumulated integral of the state-tracking error to suppress steady-state offsets driven by unmodeled disturbances; (ii) sliding-mode MPG (SM-MPG), which adds a quadratic approximation of a sliding-variable penalty to increase convergence aggressiveness; (iii) SM-MPG with a true inter-node sliding variable (SM-MPG-Q4), in which the cost penalizes a linear combination of successive predicted states scaled to mimic the dissipation dynamics of a sliding manifold; and (iv) a combined SM-MPG with integral tracking (SM-MPG-IA) that fuses both ideas under a single convex program. Each formulation is solved efficiently with OSQP and closed-loop performance is assessed through a 100-run Monte Carlo campaign against a high-fidelity Earth entry simulator perturbed by a stochastic GRAM atmosphere model. Results quantify the trade-off between reference-tracking accuracy, terminal position error, velocity error, and real-time computation time across all formulations relative to an open-loop baseline.

---

## I. Introduction

### I.A Problem Motivation

Precision Entry, Descent, and Landing (EDL) is a mission-enabling capability for planetary science, robotic sample return, and crewed exploration alike. Whether targeting the narrow ellipses demanded by Mars 2020's terrain-relative navigation or the steep corridors imposed by human-scale payloads, the guidance algorithm must steer an unpowered or lightly propelled vehicle through the atmosphere with enough accuracy to deliver it to a powered descent initiation (PDI) point that a subsequent lander or descent stage can feasibly reach.

The atmospheric entry phase is characterized by several compounding challenges. First, the aerodynamic force coefficients—lift and drag as functions of angle of attack, Mach number, and altitude—are known only approximately, with dispersions of several percent even after extensive wind-tunnel campaigns and flight testing. Second, the atmosphere itself is time-varying and spatially non-uniform: density and wind profiles deviate from engineering models by amounts that grow with altitude uncertainty and planetary-scale meteorology. Third, the dynamics are highly nonlinear: the drag deceleration scales as ρ(h)v², coupling density uncertainty directly into the velocity and flight-path-angle equations. Fourth, actuator authority is strictly limited; on a bank-angle-steered vehicle, the angle of attack is a slow variable and the bank angle is subject to both box and slew-rate limits. Any guidance law that ignores these constraints risks infeasible commanded maneuvers.

Classical entry guidance laws—such as Apollo's predictor-corrector, the NASA ETPC, and Mars Science Laboratory's Range Control—handle nonlinearity through successive linearization or numerical integration of the equations of motion. They are proven robust for relatively loose terminal requirements (tens of kilometers), but their architecture does not lend itself naturally to systematic robustness against uncertainty distributions, or to the incorporation of additional path constraints. Reaching sub-kilometer terminal precision under Monte Carlo dispersion calls for feedback that actively corrects trajectory deviations faster than the disturbance accumulates them.

### I.B Model Predictive Guidance: Background and Related Work

Model Predictive Control (MPC) has been applied to aerospace guidance since the mid-2000s, with early work targeting ascent trajectory optimization and powered descent. The extension to atmospheric entry is more recent, driven by the improved on-board compute available on deep-space missions and the need for adaptive guidance capable of responding to off-nominal conditions in real time.

Lu and colleagues established a framework for analytical predictor-corrector guidance that linearizes the entry dynamics about a reference trajectory and then corrects the control policy via a scalar drag-integral condition. While analytically elegant, this approach addresses only one degree of freedom in the terminal state (downrange) and does not directly account for multi-state covariance or constraint coupling.

The model predictive tracking guidance (MPG) paradigm, as formulated by **Davami, Lu, and Rosengren (2025)** [Ref. 1], casts entry guidance as a receding-horizon QP over linearized error dynamics. The key insight is that, given a pre-computed nominal (optimal) trajectory, the entry dynamics can be linearized about that nominal at each guidance update, yielding a discrete linear time-varying (LTV) prediction model. A QP minimizing a weighted integral of the predicted tracking error is then solved at each guidance cycle—typically at 1–2 Hz—with the first computed control correction applied to the vehicle. The resulting optimization is small enough to solve in milliseconds with an interior-point or active-set solver, yet captures the full six-state coupling of the entry dynamics.

The MPG law of [Ref. 1] demonstrated that receding-horizon tracking substantially reduces terminal dispersions relative to open-loop replay of the nominal controls, even under GRAM-based atmospheric uncertainty. However, the baseline formulation raises several open questions:

1. **Steady-state offset.** If a persistent disturbance (e.g., a systematic density bias) is present, the standard MPG incurs a constant tracking error whose magnitude depends on the open-loop prediction horizon and the disturbance bandwidth. Integral action is a classical remedy in linear control theory but has not been systematically integrated into receding-horizon entry guidance.

2. **Convergence rate.** The MPG objective is a quadratic penalty on nodal tracking errors, which provides asymptotically stable behavior but does not prescribe a minimum convergence rate. For large initial deviations from the reference—arising, for example, from a large state jump at entry interface—a faster-converging law is desirable.

3. **Probabilistic constraint satisfaction.** The nominal MPG does not formally account for the probability that predicted state trajectories satisfy box constraints (altitude bands, cross-range limits, heating constraints). A chance-constrained formulation, in which probabilistic tubes tighten the QP constraints to guarantee a prescribed confidence level of feasibility, offers a principled route to safe guidance under stochastic uncertainty.

This paper addresses all three questions by introducing a family of MPG extensions that augment the baseline cost function while preserving its convex QP structure, common prediction model, and constraint set.

### I.C Contributions of This Work

The specific contributions of this paper are:

1. **MPG with Integral Action (MPG-IA):** A formulation that augments the MPG objective with a quadratic penalty on the running integral of the state-tracking error, accumulating across guidance updates. We derive the integral propagation matrices consistent with the Heun/RK2 discretization used in the prediction model, show that the integral cost adds only $n \times (N+1)$ additional terms to the QP Hessian (where $n = 6$ and $N$ is the horizon length), and demonstrate that the stationary solution of the resulting receding-horizon law asymptotically rejects constant disturbances.

2. **Sliding-Mode MPG (SM-MPG):** Two formulations are introduced. The first (SM-MPG-State) adds a state-tracking penalty with independently tuned weights that increase aggressiveness beyond the baseline. The second and more principled formulation (SM-MPG-Q4) introduces a true inter-node sliding variable $s_k = C_s(\delta x_{k+1} - \lambda \delta x_k)$, where $\lambda \in [0, 1]$ governs the decay rate of the sliding manifold. Minimizing $\sum_k \|s_k\|^2_{W_s}$ in the QP objective induces a closed-loop dissipation analogous to the reaching law of classical sliding-mode control, but entirely within the convex QP framework—avoiding the discontinuities and chattering associated with traditional SMC.

3. **Combined SM-MPG with Integral Action (SM-MPG-IA):** A fully augmented formulation combining the SM-MPG-State terms, the integral-of-error cost, and a penalty on the actual control increment. This formulation targets both convergence rate and steady-state accuracy simultaneously.

4. **Monte Carlo Validation:** A 100-run Monte Carlo campaign using a high-fidelity six-state entry simulator with GRAM stochastic atmosphere perturbations quantitatively compares all five formulations (Baseline MPG, MPG-IA, SM-MPG-State, SM-MPG-Q4, SM-MPG-IA) against an open-loop baseline, reporting terminal position error, velocity error, and solver timing statistics.

### I.D Paper Organization

Section II formulates the entry dynamics and reference trajectory problem. Section III reviews the baseline MPG law and its QP structure. Sections IV–VI present the three families of novel formulations in detail. Section VII describes the Monte Carlo simulation setup. Section VIII presents results. Section IX concludes.

---

## II. Entry Dynamics and Reference Trajectory

### II.A Equations of Motion

Consider a rigid vehicle entering a spherically symmetric, rotating Earth atmosphere. The state vector under a bank-angle steering law is

$$
x = \begin{bmatrix} h & \phi & \theta & v & \gamma & \psi \end{bmatrix}^\top \in \mathbb{R}^6,
$$

where $h$ is altitude above the reference ellipsoid (m), $\phi$ is longitude (rad), $\theta$ is geodetic latitude (rad), $v$ is relative airspeed (m/s), $\gamma$ is the relative flight-path angle (rad), and $\psi$ is the relative heading angle (rad) measured clockwise from North. The control vector is

$$
u = \begin{bmatrix} \alpha & \beta \end{bmatrix}^\top,
$$

where $\alpha$ is the angle of attack (rad) and $\beta$ is the bank angle (rad). The continuous-time equations of motion are

$$
\dot{h} = v \sin\gamma,
$$

$$
\dot{\phi} = \frac{v \cos\gamma \sin\psi}{(R_E + h)\cos\theta},
$$

$$
\dot{\theta} = \frac{v \cos\gamma \cos\psi}{R_E + h},
$$

$$
\dot{v} = -D - g\sin\gamma + \Omega_E^2(R_E + h)\cos\theta(\sin\gamma\cos\theta - \cos\gamma\sin\theta\cos\psi),
$$

$$
\dot{\gamma} = \frac{1}{v}\left[L\cos\beta - g\cos\gamma + \frac{v^2\cos\gamma}{R_E + h} + 2\Omega_E v \cos\theta\sin\psi + \Omega_E^2(R_E + h)\cos\theta(\cos\gamma\cos\theta + \sin\gamma\sin\theta\cos\psi)\right],
$$

$$
\dot{\psi} = \frac{1}{v\cos\gamma}\left[L\sin\beta + \frac{v^2\cos\gamma\sin\psi\tan\theta}{R_E + h} - 2\Omega_E v(\cos\theta\cos\psi\tan\gamma - \sin\theta) + \frac{\Omega_E^2(R_E + h)\sin\theta\cos\theta\sin\psi}{\cos\gamma}\right],
$$

where $R_E$ is the Earth's mean radius, $\Omega_E$ is Earth's rotation rate, $g = \mu/(R_E + h)^2$ is the local gravitational acceleration, and $L$, $D$ are the lift and drag accelerations (m/s²):

$$
D = \frac{1}{2}\rho(h) v_r^2 \frac{S_{\rm ref}}{m} C_D(\alpha), \qquad L = \frac{1}{2}\rho(h) v_r^2 \frac{S_{\rm ref}}{m} C_L(\alpha),
$$

with $\rho(h)$ the atmospheric density (from the GRAM model), $v_r$ the vehicle velocity relative to the atmosphere (including wind perturbations), $S_{\rm ref}$ the aerodynamic reference area, $m$ the vehicle mass, and $C_D(\alpha)$, $C_L(\alpha)$ polynomial fits to the aerodynamic database as functions of angle of attack.

The aerodynamic coefficients are modeled as

$$
C_D(\alpha) = c_{D,0} + c_{D,1}\alpha + c_{D,2}\alpha^2, \qquad C_L(\alpha) = c_{L,0} + c_{L,1}\alpha,
$$

where coefficients $c_{(\cdot)}$ are obtained from CFD/wind-tunnel data for the specific vehicle geometry. This formulation, written compactly as $\dot{x} = f(x, u)$, is the basis for all guidance laws developed here.

### II.B Actuator Constraints

The vehicle is subject to amplitude and slew-rate limits on both control channels:

$$
\alpha_{\min} \leq \alpha \leq \alpha_{\max}, \qquad \beta_{\min} \leq \beta \leq \beta_{\max},
$$

$$
|\dot{\alpha}| \leq \dot{\alpha}_{\max}, \qquad |\dot{\beta}| \leq \dot{\beta}_{\max}.
$$

Numerically, $\alpha \in [-90^\circ, 90^\circ]$, $\beta \in [-89^\circ, 89^\circ]$, $\dot{\alpha}_{\max} = 30\ ^\circ/\text{s}$, $\dot{\beta}_{\max} = 20\ ^\circ/\text{s}$.

### II.C Nominal Reference Trajectory

The reference trajectory $\{x_{\rm ref}(t), u_{\rm ref}(t)\}$ is computed off-line as the solution to a constrained optimal control problem (OCP):

$$
\min_{x(\cdot), u(\cdot)} \quad J_{\rm OCP}(x, u)
$$

$$
\text{subject to} \quad \dot{x} = f(x, u),\ x(t_0) = x_0,\ x(t_f) \in \mathcal{X}_f,\ u(t) \in \mathcal{U},
$$

where $J_{\rm OCP}$ is a user-specified mission objective (e.g., minimize terminal position error or heat load), $\mathcal{X}_f$ encodes terminal state requirements (target altitude, latitude, longitude, velocity, flight-path angle), and $\mathcal{U}$ represents the actuator constraints. The OCP is solved offline (e.g., via direct transcription or sequential convex programming) for nominal atmospheric and vehicle parameters.

At runtime, the reference is stored as a discrete time-series $\{x_{\rm ref,k}, u_{\rm ref,k}\}_{k=0}^{K}$ indexed by time, and interpolated as needed. The guidance task is then to steer the vehicle from the uncertain, perturbed initial condition back to—and along—this reference trajectory, subject to the actuator limits.

---

## III. Baseline Model Predictive Guidance (MPG)

This section reviews the MPG law of [Ref. 1] in the notation used throughout the paper.

### III.A Error-State Linearization

At each guidance update time $t_0$, let $\delta x_0 = x(t_0) - x_{\rm ref}(t_0)$ be the current state error. The nonlinear dynamics $\dot{x} = f(x, u)$ are linearized about the reference pair at each node $k$ along the horizon:

$$
\dot{\delta x}(t) \approx A_k \,\delta x(t) + B_k\, \Delta u(t), \qquad t \in [t_k, t_{k+1}],
$$

where

$$
A_k = \left.\frac{\partial f}{\partial x}\right|_{(x_{{\rm ref},k},\, u_{{\rm ref},k})} \in \mathbb{R}^{6\times 6}, \qquad B_k = \left.\frac{\partial f}{\partial u}\right|_{(x_{{\rm ref},k},\, u_{{\rm ref},k})} \in \mathbb{R}^{6\times 2},
$$

computed via automatic differentiation (ForwardDiff). The control deviation at node $k$ is $\Delta u_k = u_k - u_{{\rm ref},k}$.

### III.B Heun / RK2 Transcription

Rather than a zero-order-hold discretization, the prediction model uses a Heun (second-order Runge-Kutta) transcription over each interval of length $h_k = t_{k+1} - t_k$:

$$
S_k = I + \frac{h_k}{2}(A_k + A_{k+1}) + \frac{h_k^2}{2} A_{k+1} A_k,
$$

$$
P_k = \frac{h_k}{2}(I + h_k A_{k+1}) B_k, \qquad Q_k = \frac{h_k}{2} B_{k+1}.
$$

The discrete error-propagation relation is

$$
\delta x_{k+1} = S_k \delta x_k + P_k \Delta u_k + Q_k \Delta u_{k+1}.
$$

Because both $\Delta u_k$ and $\Delta u_{k+1}$ appear, the optimization variable contains a control correction at every node $k = 0, \ldots, N$:

$$
z = \begin{bmatrix} \Delta u_0^\top & \Delta u_1^\top & \cdots & \Delta u_N^\top \end{bmatrix}^\top \in \mathbb{R}^{2(N+1)}.
$$

### III.C Global Prediction Matrices

Recursively building from $\Phi_0 = I$ and $\Psi_0 = 0$:

$$
\Phi_k = S_{k-1} \Phi_{k-1}, \qquad k = 1, \ldots, N,
$$

$$
\Psi_k = S_{k-1}\Psi_{k-1} + [\underbrace{0 \;\cdots\; 0}_{k-1} \;\; P_{k-1} \;\; Q_{k-1} \;\; \underbrace{0 \;\cdots\; 0}_{N-k}],
$$

the predicted error at node $k$ is expressed as the affine map

$$
\delta x_k = \Phi_k \delta x_0 + \Psi_k z, \qquad k = 0, 1, \ldots, N.
$$

### III.D Trapezoidal Quadrature Weights

The stage cost is integrated using node quadrature weights consistent with the trapezoidal rule over a non-uniform grid:

$$
w_0 = \frac{h_0}{4}, \quad w_k = \frac{h_{k-1} + h_k}{4}\ \text{for } 1 \leq k \leq N-1, \quad w_N = \frac{h_{N-1}}{4}.
$$

### III.E Baseline MPG Objective

Define the diagonal scaling matrix

$$
G = \mathrm{diag}(10^{-5},\ 1,\ 1,\ 10^{-4},\ 1,\ 1),
$$

with corresponding normalized weight matrices

$$
Q_n = \mathrm{diag}(1200,\ 3500,\ 3500,\ 1200,\ 700,\ 1200),
$$

$$
F_n = \mathrm{diag}(2500,\ 180000,\ 180000,\ 4500,\ 2500,\ 4000),
$$

$$
R = \mathrm{diag}(1,\ 1).
$$

The actual (unnormalized) cost matrices are $Q = G^\top Q_n G$ and $F = G^\top F_n G$. The baseline MPG objective is

$$
\boxed{J_{\rm MPG}(z) = \frac{1}{2}\sum_{k=0}^{N} w_k \left(\delta x_k^\top Q\,\delta x_k + \Delta u_k^\top R\, \Delta u_k\right) + \frac{1}{2}\,\delta x_N^\top F\, \delta x_N.}
$$

Substituting $\delta x_k = \Phi_k \delta x_0 + \Psi_k z$, this is quadratic in $z$ and assembles into the standard OSQP form

$$
\min_z \quad \frac{1}{2} z^\top P_{\rm qp} z + q_{\rm qp}^\top z,
$$

with

$$
P_{\rm qp} = \sum_{k=0}^{N} w_k \left(\Psi_k^\top Q \Psi_k + E_k^\top R E_k\right) + \Psi_N^\top F \Psi_N,
$$

$$
q_{\rm qp} = \sum_{k=0}^{N} w_k \Psi_k^\top Q \Phi_k \delta x_0 + \Psi_N^\top F \Phi_N \delta x_0,
$$

where $E_k \in \mathbb{R}^{2 \times 2(N+1)}$ is the selector matrix extracting $\Delta u_k$ from $z$.

### III.F Control Constraints

**Amplitude limits** at each node:

$$
u_{\min} \leq u_{{\rm ref},k} + \Delta u_k \leq u_{\max}, \qquad k = 0, \ldots, N.
$$

**First-step rate limit** (relative to the previously applied control $u_{\rm prev}$):

$$
-\dot{u}_{\max} \Delta t_{\rm step} \leq (u_{{\rm ref},0} + \Delta u_0) - u_{\rm prev} \leq \dot{u}_{\max} \Delta t_{\rm step}.
$$

**Inter-node rate limits** for $k = 1, \ldots, N$:

$$
-\dot{u}_{\max} h_{k-1} - \Delta u_{{\rm ref},k} \leq \Delta u_k - \Delta u_{k-1} \leq \dot{u}_{\max} h_{k-1} - \Delta u_{{\rm ref},k},
$$

where $\Delta u_{{\rm ref},k} = u_{{\rm ref},k} - u_{{\rm ref},k-1}$. All constraint rows are collected into a single linear inequality system $A_c z \in [l_c, u_c]$ imposed on OSQP.

### III.G Shrinking Horizon and Receding-Horizon Execution

The prediction horizon shrinks as the vehicle approaches the end of the reference trajectory. If $t_f$ is the final reference time and $N_{\rm max}$ is the maximum horizon length, the active horizon at time $t_0$ is

$$
N = \min\!\left(N_{\rm max},\left\lfloor \frac{t_f - t_0}{\Delta t} \right\rfloor\right).
$$

At each guidance cycle, the QP is solved (warm-started from the previous shifted solution), and the first correction $\Delta u_0^\star$ is applied: $u_{\rm cmd} = u_{{\rm ref},0} + \Delta u_0^\star$, clamped to actuator limits.

---

## IV. MPG with Integral Action (MPG-IA)

### IV.A Motivation

When a persistent disturbance—such as a constant atmospheric density bias—acts on the vehicle, the baseline MPG converges to a nonzero steady-state tracking error whose magnitude depends on the disturbance magnitude and the prediction horizon length. This is the standard proportional-only behavior of finite-horizon receding-horizon control. Adding a penalization of the accumulated (integral) tracking error is the natural extension to achieve zero steady-state offset, analogous to the PI/I augmentation in classical linear control.

### IV.B Integral State Propagation

Let $i_0 \in \mathbb{R}^6$ denote the integral-of-error state accumulated up to time $t_0$:

$$
i(t) = i_0 + \int_{t_0}^{t} \delta x(\tau)\, d\tau.
$$

Using the linear error propagation $\delta x_k = \Phi_k \delta x_0 + \Psi_k z$, a left-endpoint rectangular approximation yields

$$
i_k \approx i_0 + \Phi^i_k \delta x_0 + \Psi^i_k z,
$$

with integral propagation matrices initialized as $\Phi^i_0 = 0$, $\Psi^i_0 = 0$ and updated recursively:

$$
\Phi^i_k = \Phi^i_{k-1} + h_{k-1} \Phi_{k-1}, \qquad \Psi^i_k = \Psi^i_{k-1} + h_{k-1} \Psi_{k-1}.
$$

### IV.C Integral Cost Term

Let the time-horizon-normalized integral weight be

$$
G_i = \mathrm{diag}\!\left(\frac{1}{T_H s_1},\ldots,\frac{1}{T_H s_6}\right), \qquad T_H = \max\!\left(\sum_{k=0}^{N-1} h_k,\,1\right),
$$

where $s_j$ are the same state scales as in $G$, and $T_H$ is the current horizon duration in seconds. With diagonal weight $Q_{i,n} = \mathrm{diag}(800, 1500, 1500, 800, 300, 800)$, the integral cost is

$$
\boxed{J_{\rm int}(z) = \frac{1}{2}\sum_{k=0}^{N} w_k\, i_k^\top Q_i\, i_k, \qquad Q_i = G_i^\top Q_{i,n} G_i.}
$$

Since $i_k$ is affine in $z$, this adds a positive-semidefinite block to $P_{\rm qp}$ and a linear term to $q_{\rm qp}$, with no change to the OSQP problem structure.

### IV.D Actual Control Increment Penalty

To additionally discourage abrupt deviations in the commanded control history (as opposed to deviations from the reference increment), MPG-IA penalizes

$$
\boxed{J_{\Delta u,{\rm act}}(z) = \left\|u_0 - u_{\rm prev}\right\|_{R_{\Delta a}}^2 + \sum_{k=1}^{N} \left\|u_k - u_{k-1}\right\|_{R_{\Delta a}}^2,}
$$

with $R_{\Delta a} = \mathrm{diag}(1, 1)$ scaled by normalized interval lengths. Because $u_k = u_{{\rm ref},k} + \Delta u_k$, this term equals $\left\|\Delta u_k - \Delta u_{k-1} + \Delta u_{{\rm ref},k}\right\|^2_{R_{\Delta a}}$, which differs from the inter-node rate constraint term in the constraint set: here it is a soft penalty in the objective, not a hard inequality.

### IV.E Total Objective and Integral State Update

$$
J_{\rm MPG\text{-}IA}(z) = J_{\rm MPG}(z) + J_{\rm int}(z) + J_{\Delta u,{\rm act}}(z).
$$

After each guidance cycle, the integral state is updated using a forward-Euler step:

$$
i_0^{+} = i_0 + \Delta t_{\rm update}\, \delta x_0,
$$

where $\Delta t_{\rm update}$ is the guidance period. This update runs independently of the QP horizon and accumulates disturbance history across guidance cycles.

### IV.F QP Dimension Analysis

The addition of $J_{\rm int}$ introduces no new decision variables; the Hessian $P_{\rm qp}$ grows by $\sum_k w_k (\Psi^i_k)^\top Q_i \Psi^i_k$ and $q_{\rm qp}$ acquires an additional term $\sum_k w_k (\Psi^i_k)^\top Q_i (i_0 + \Phi^i_k \delta x_0)$. Both are computed in $O(n^2(N+1)^2 m)$ operations, where $m = 2$ (control dimension), identical in order to the baseline MPG assembly.

---

## V. Sliding-Mode Model Predictive Guidance

Two variants of the SM-MPG concept are presented, differing in how the "sliding" structure enters the cost function.

### V.A Motivation: Convergence Rate in Receding-Horizon Guidance

The baseline MPG provides asymptotically stable tracking, but its convergence rate to the reference is governed entirely by the choice of stage and terminal weights $Q$, $F$ and the horizon length $N$. For missions that begin far off the reference (e.g., due to a large dispersed entry interface condition), there is practical value in a guidance law that explicitly prescribes a minimum convergence rate. Sliding-mode control achieves prescribed convergence rates by driving the system state onto an invariant manifold $\mathcal{S} = \{x : \sigma(x) = 0\}$ and then maintaining it there. The classical approach is, however, discontinuous and produces high-frequency control chattering that is incompatible with the slew-rate limits of real actuators. The SM-MPG approach replaces the discontinuous switching law with a quadratic penalty on the sliding variable, embedding its minimization in the convex QP.

### V.B SM-MPG-State: Augmented Nodal Tracking Penalty

`sm_mpg.jl` adds an extra nodal state-tracking penalty beyond the baseline, with a separate weight matrix tuned for aggressiveness:

$$
Q_{\rm slide} = G_t^\top \,\mathrm{diag}(3000, 3000, 5000, 100, 10, 100)\, G_t,
$$

$$
F_{\rm slide} = G_t^\top \,\mathrm{diag}(3000, 3000, 3000, 100, 10, 100)\, G_t,
$$

where $G_t = G$ (same scales). The added cost is

$$
\boxed{J_{{\rm slide,state}}(z) = \frac{1}{2}\sum_{k=1}^{N} \omega_k\, \delta x_k^\top Q_{\rm slide}\, \delta x_k + \frac{1}{2}\, \delta x_N^\top F_{\rm slide}\, \delta x_N,}
$$

with normalized interval weights $\omega_k$.

Two additional regularization terms penalize the control deviation $v_k = \Delta u_k$ and its increments:

$$
\boxed{J_{\rm dev}(z) = \frac{1}{2}\sum_{k=0}^{N} w_k\, v_k^\top R_v\, v_k, \qquad R_v = \mathrm{diag}(10^{-2}, 10^{-1}),}
$$

$$
\boxed{J_{\Delta v}(z) = \left\|v_0 - v_{-1}\right\|_{R_{\Delta v}}^2 + \sum_{k=1}^{N}\left\|v_k - v_{k-1}\right\|_{R_{\Delta v}}^2, \qquad R_{\Delta v} = \mathrm{diag}(0.5, 0.5),}
$$

where $v_{-1} = u_{\rm prev} - u_{{\rm ref,prev}}$ is the previous control deviation. The total SM-MPG-State objective is

$$
J_{\rm SM\text{-}State}(z) = J_{\rm MPG}(z) + J_{\rm slide,state}(z) + J_{\rm dev}(z) + J_{\Delta v}(z).
$$

### V.C SM-MPG-Q4: True Inter-Node Sliding Variable

`sm_mpg_q4_tracking.jl` introduces a genuinely inter-node sliding variable. For $k = 0, \ldots, N-1$, define

$$
s_k = C_s\!\left(\delta x_{k+1} - \lambda\, \delta x_k\right),
$$

where $C_s = G_t$ (same scales as the tracking formulation) and $\lambda \in [0, 1]$ is the prescribed sliding-manifold decay rate. Substituting the prediction model:

$$
s_k = C_s\left[(\Phi_{k+1} - \lambda \Phi_k)\,\delta x_0 + (\Psi_{k+1} - \lambda \Psi_k)\, z\right].
$$

Define $\tilde{\Phi}_k = C_s(\Phi_{k+1} - \lambda \Phi_k)$ and $\tilde{\Psi}_k = C_s(\Psi_{k+1} - \lambda \Psi_k)$. The sliding-variable cost is

$$
\boxed{J_{Q4}(z) = \frac{1}{2}\sum_{k=0}^{N-1} \omega_k\, s_k^\top W_s\, s_k, \qquad W_s = \mathrm{diag}(1200, 3500, 3500, 1200, 700, 1200),}
$$

expanding to

$$
J_{Q4}(z) = \frac{1}{2} z^\top \!\left(\sum_{k=0}^{N-1} \omega_k \tilde{\Psi}_k^\top W_s \tilde{\Psi}_k\right) z + z^\top \!\left(\sum_{k=0}^{N-1} \omega_k \tilde{\Psi}_k^\top W_s \tilde{\Phi}_k \delta x_0\right) + \text{const}.
$$

The total SM-MPG-Q4 objective is

$$
J_{\rm SM\text{-}Q4}(z) = J_{\rm MPG}(z) + J_{Q4}(z) + J_{\Delta u,{\rm act}}(z).
$$

**Interpretation.** When $\lambda = 0$, $s_k = C_s \delta x_{k+1}$ and the term collapses to a standard nodal tracking penalty. When $\lambda = 1$, $s_k = C_s(\delta x_{k+1} - \delta x_k)$ penalizes the increment of the tracking error—analogous to a derivative action. For intermediate $\lambda$, the term penalizes linear combinations of error and error increment that correspond to placing the closed-loop poles at $\lambda$ in the $z$-domain along the sliding direction. Thus, tuning $\lambda$ directly sets the desired error convergence rate in the prediction model.

### V.D SM-MPG with Integral Action (SM-MPG-IA)

`sm_mpg_integral_tracking.jl` combines the SM-MPG-State terms, the integral error cost, and the actual-control increment penalty:

$$
J_{\rm SM\text{-}IA}(z) = J_{\rm MPG}(z) + J_{\rm slide,state}(z) + J_{\rm dev}(z) + J_{\Delta v}(z) + J_{\rm int}(z) + J_{\Delta u,{\rm act}}(z).
$$

This is the most heavily regularized formulation in the family, targeting simultaneously: (a) trajectory tracking through $J_{\rm MPG}$, (b) increased aggressiveness through $J_{\rm slide,state}$, (c) smooth control deviation through $J_{\rm dev}$ and $J_{\Delta v}$, (d) steady-state offset rejection through $J_{\rm int}$, and (e) smooth actual-control history through $J_{\Delta u,{\rm act}}$.

---

## VI. Chance-Constrained MPG (CC-MPG)

### VI.A Motivation: Probabilistic Constraint Satisfaction

The formulations in Sections III–V minimize expected tracking error but provide no formal guarantee that state constraints are satisfied with a prescribed probability. In practice, a guidance law that produces excellent mean terminal accuracy may still yield an unacceptably large fraction of Monte Carlo runs where, for example, the predicted heating corridor is violated or the vehicle exits a safe altitude band. Chance-constrained MPC (CC-MPC) addresses this by reformulating the path constraints as probabilistic tube conditions:

$$
\mathrm{Pr}\!\left(\left|[\delta x_k]_i\right| \leq \delta_i\right) \geq p_i, \qquad i = 1, \ldots, 6,\ k = 1, \ldots, N,
$$

where $\delta_i$ is the tube half-width for state $i$ and $p_i \in (0, 1)$ is the desired satisfaction probability.

### VI.B Pre-Computed Uncertainty Propagation

A Monte Carlo campaign of $M_\sigma$ runs (e.g., $M_\sigma = 200$) is conducted offline under the open-loop (nominal) control policy with GRAM stochastic atmosphere realizations. At each time node $t_k$ along the reference trajectory, the empirical standard deviation of each state deviation is recorded:

$$
\sigma_i(t_k) = \sqrt{\frac{1}{M_\sigma - 1}\sum_{j=1}^{M_\sigma}\left([x_j(t_k)]_i - [x_{{\rm ref}}(t_k)]_i\right)^2}.
$$

The resulting schedule $\sigma_i(t)$ is stored in `mc_sigma_schedule.csv` and interpolated at runtime. This offline sigma schedule characterizes the free-flying (uncontrolled) uncertainty growth and serves as a conservative upper bound on the uncertainty under any stabilizing guidance law.

### VI.C Tightened State Tube Constraints

Under a Gaussian assumption (or by invoking a Chebyshev-type bound), the $p_i$-probability satisfaction of $|[\delta x_k]_i| \leq \delta_i$ is guaranteed if

$$
|\mathbb{E}[[\delta x_k]_i]| \leq \delta_i - z_{p_i}\,\sigma_i(t_k),
$$

where $z_{p_i}$ is the $p_i$-quantile of the standard normal distribution ($z_{0.95} = 1.645$, $z_{0.975} = 1.96$). The mean predicted state deviation under the MPG law is $\Phi_k \delta x_0 + \Psi_k z$ (since the QP is deterministic given $\delta x_0$). The tightened tube constraint on the prediction is therefore

$$
\left|[\Phi_k \delta x_0 + \Psi_k z]_i\right| \leq \max\!\left(0,\, \delta_i - z_p\,\sigma_i(t_k)\right) + s_{k,i}, \qquad s_{k,i} \geq 0,
$$

where $s_{k,i} \geq 0$ is a slack variable introduced for recursive feasibility: when the tube is so tight that no feasible $z$ exists without violating it, the slack allows constraint softening at the cost of a large penalty in the objective. The nominal tube half-widths used here are

$$
\delta = [2000\ {\rm m},\ 0.01\ {\rm rad},\ 0.01\ {\rm rad},\ 200\ {\rm m/s},\ 0.05\ {\rm rad},\ 0.05\ {\rm rad}]^\top.
$$

### VI.D CC-MPG Objective and QP Formulation

The CC-MPG objective augments the baseline MPG tracking cost with a slack-penalty term:

$$
\boxed{J_{\rm CC}(z, s) = J_{\rm MPG}(z) + \rho \sum_{k=1}^{N}\sum_{i=1}^{6} s_{k,i}^2, \qquad \rho = 10^6.}
$$

The augmented decision vector is $\tilde{z} = [z^\top,\, s^\top]^\top \in \mathbb{R}^{2(N+1) + 6N}$. The OSQP problem then includes

- the baseline cost Hessian in the $z$-block;
- $\rho I$ in the slack-variable diagonal;
- zero coupling between $z$ and $s$ in the Hessian (slacks enter only in the cost diagonal);
- additional linear inequality rows (upper tube, lower tube, slack non-negativity) in the constraint matrix $A_c$.

The constraint rows for the tube are (normalized by $\delta_i$ for numerical conditioning):

$$
-\left(\delta_i - z_p\sigma_i(t_k)\right)/\delta_i - s_{k,i}/\delta_i \leq \left[\Phi_k \delta x_0 + \Psi_k z\right]_i / \delta_i \leq \left(\delta_i - z_p\sigma_i(t_k)\right)/\delta_i + s_{k,i}/\delta_i.
$$

### VI.E Relationship to Tube MPC

The CC-MPG formulation is philosophically related to Tube MPC [Ref. X], in which a nominal trajectory is planned over the mean-equivalent dynamics and a separate feedback law confines the actual trajectory within an invariant tube. Here, rather than a separate ancillary controller, the tube constraint is embedded directly in the tracking QP, and the sigma schedule replaces an analytically computed invariant set. This is a computationally tractable approximation: the tube tightening is based on the open-loop uncertainty growth (conservative), while the actual closed-loop uncertainty is smaller due to the MPG feedback. The conservatism is quantified in the Monte Carlo analysis of Section VIII.

---

## VII. Side-by-Side Summary of All Formulations

| Formulation | File | Additional Cost Terms | Decision Variables | QP Size (approx.) |
|---|---|---|---|---|
| MPG (baseline) | `model_predictive_guidance.jl` | — | $z \in \mathbb{R}^{2(N+1)}$ | 80–200 |
| MPG-IA | `mpg_integral_tracking.jl` | $J_{\rm int} + J_{\Delta u,{\rm act}}$ | $z \in \mathbb{R}^{2(N+1)}$ | 80–200 |
| SM-MPG-State | `sm_mpg.jl` | $J_{\rm slide,state} + J_{\rm dev} + J_{\Delta v}$ | $z \in \mathbb{R}^{2(N+1)}$ | 80–200 |
| SM-MPG-Q4 | `sm_mpg_q4_tracking.jl` | $J_{Q4} + J_{\Delta u,{\rm act}}$ | $z \in \mathbb{R}^{2(N+1)}$ | 80–200 |
| SM-MPG-IA | `sm_mpg_integral_tracking.jl` | $J_{\rm slide,state} + J_{\rm dev} + J_{\Delta v} + J_{\rm int} + J_{\Delta u,{\rm act}}$ | $z \in \mathbb{R}^{2(N+1)}$ | 80–200 |
| CC-MPG | `cc_mpg.jl` | $\rho\|s\|^2$ with tube constraints | $[z;\, s] \in \mathbb{R}^{2(N+1)+6N}$ | 500–1200 |

All formulations share: the same six-state entry dynamics; the same Heun/RK2 prediction matrices $(\Phi_k, \Psi_k)$; the same amplitude and slew-rate constraints; the same warm-start receding-horizon shift; and the same OSQP backend.

---

## VIII. Monte Carlo Simulation Setup

### VIII.A Simulator and Uncertainty Model

The high-fidelity closed-loop simulator integrates the full nonlinear equations of motion (Section II.A) using a variable-step ODE solver. At each guidance cycle (period $\approx 0.6$ s), the guidance law queries the current state from the integrator and returns an updated control command, clamped to actuator limits.

Atmospheric uncertainty is modeled using the Global Reference Atmosphere Model (GRAM), which generates correlated stochastic realizations of density, temperature, and wind profiles. Each of the $M = 100$ Monte Carlo runs uses a distinct GRAM random seed, producing a different density and wind realization over the full entry trajectory.

### VIII.B Controllers Compared

The following guidance policies are evaluated:

1. **Open Loop:** Replay of the nominal control $u_{\rm ref}(t)$ with no feedback—a pure baseline.
2. **MPG:** Baseline law of Section III.
3. **MPG-IA:** Integral action variant (Section IV).
4. **SM-MPG-State:** Sliding-mode state variant (Section V.B).
5. **SM-MPG-Q4:** True sliding variable variant (Section V.C).

### VIII.C Performance Metrics

The following terminal metrics are recorded for each run $j = 1, \ldots, M$:

- **Final position error norm:** $\left\|\begin{bmatrix}x_j - x_{\rm tgt} \\ y_j - y_{\rm tgt} \\ z_j - z_{\rm tgt}\end{bmatrix}\right\|$ (km), where positions are in Earth-centered Cartesian coordinates.

- **Final velocity error norm:** $\left\|v_{{\rm cart},j} - v_{\rm tgt}\right\|$ (m/s).

- **Landing location:** Final latitude/longitude scatter (deg).

- **Altitude error:** $|h_j - h_{\rm tgt}|$ (m).

- **Solver timing statistics:** Mean and max OSQP solve time per guidance cycle (ms).

Aggregate statistics reported: mean, median, 1-σ standard deviation, 90th/95th percentile, and maximum across all $M$ runs.

---

## IX. Results

*[PLACEHOLDER — Results will be filled in from Monte Carlo runs]*

### IX.A Terminal Position Error

*[Table and box plots comparing position error distributions across all controllers. Expected finding: MPG-IA and SM-MPG-Q4 show reduced mean and variance relative to baseline MPG and open loop.]*

### IX.B Terminal Velocity Error

*[Table and scatter plots of final velocity magnitude error. Expected finding: sliding-mode variants converge faster under large initial deviations, reducing velocity error tail risk.]*

### IX.C Landing Location Scatter

*[Lat/lon scatter plots showing landing ellipse area and centroid offset for each controller. Expected finding: MPG family dramatically smaller than open loop; SM/IA variants show incremental improvement.]*

### IX.D Solver Timing

*[Bar chart or table of mean/max OSQP solve times. Expected finding: MPG, MPG-IA, SM-MPG all comparable (~5–15 ms per guidance cycle); CC-MPG higher (~30–80 ms) due to larger QP.]*

### IX.E Robustness vs. Conservatism Trade-off in CC-MPG

*[Comparison of actual MC constraint violation rate versus the prescribed confidence level $p = 0.975$. Quantification of conservatism introduced by the open-loop sigma schedule.]*

---

## X. Conclusions

This paper presented and comparatively evaluated a family of Model Predictive Tracking Guidance laws for atmospheric entry, building upon the baseline MPG formulation of Davami, Lu, and Rosengren. All variants operate on the same linearized error-propagation model, respect the same actuator constraints, and are solved with the same convex QP solver—differing only in the structure of their objective function. The key conclusions are:

1. **Integral action (MPG-IA)** provides a systematic mechanism to reject persistent, slowly varying disturbances (e.g., mean density bias) that the baseline MPG cannot suppress, at negligible additional computational cost. The integral state update is compatible with the receding-horizon framework and adds no new decision variables.

2. **Sliding-mode cost shaping (SM-MPG-State)** offers a heuristic but effective way to increase aggressiveness of convergence by independently weighting the nodal tracking errors. Its contribution relative to baseline MPG is primarily to increase the effective closed-loop bandwidth.

3. **The inter-node sliding variable formulation (SM-MPG-Q4)** is the most principled of the SM variants: the parameter $\lambda$ directly governs the desired convergence rate in the prediction model, analogous to pole placement on the sliding manifold. This provides a theoretically grounded tuning knob for performance-robustness trade-offs.

4. **Combining SM and integral action (SM-MPG-IA)** addresses both convergence rate and steady-state accuracy simultaneously. The penalty is a larger Hessian assembly cost, though the QP dimension remains unchanged.

5. **Chance-constrained MPG** provides a probabilistic safety layer absent from all other variants. The offline sigma schedule from Monte Carlo pre-runs is an operationally tractable approximation to full uncertainty propagation, and the slack-penalty formulation ensures recursive feasibility.

Future work includes: (i) adaptive integral gain scheduling to avoid wind-up during large transients; (ii) co-design of the nominal trajectory and the sigma schedule in an iterative loop to reduce CC-MPG conservatism; (iii) online estimation of the disturbance process to inform the chance-constraint tightening in real time; and (iv) extension to the powered descent phase, where thrust constraints add a third control channel.

---

## References

[1] Davami, C., Lu, P., and Rosengren, A. J., "Model Predictive Tracking Guidance Applied to Planetary Entry and Powered Descent," *AIAA SciTech Forum 2025*, Paper AIAA-2025-2599.

[2] Lu, P., "Entry Guidance: A Unified Method," *Journal of Guidance, Control, and Dynamics*, Vol. 37, No. 3, 2014, pp. 713–728.

[3] Mayne, D. Q., Rawlings, J. B., Rao, C. V., and Scokaert, P. O. M., "Constrained Model Predictive Control: Stability and Optimality," *Automatica*, Vol. 36, No. 6, 2000, pp. 789–814.

[4] Blackmore, L., Ono, M., and Williams, B. C., "Chance-Constrained Optimal Path Planning with Obstacles," *IEEE Transactions on Robotics*, Vol. 27, No. 6, 2011, pp. 1080–1094.

[5] Mayne, D. Q., Seron, M. M., and Raković, S. V., "Robust Model Predictive Control of Constrained Linear Systems with Bounded Disturbances," *Automatica*, Vol. 41, No. 2, 2005, pp. 219–224.

[6] Szmuk, M., and Açikmeşe, B., "Successive Convexification for 6-DoF Mars Rocket Powered Landing with Free-Final-Time," *AIAA SciTech Forum 2018*, Paper AIAA-2018-0617.

[7] Stellato, B., Banjac, G., Goulart, P., Bemporad, A., and Boyd, S., "OSQP: An Operator Splitting Solver for Quadratic Programs," *Mathematical Programming Computation*, Vol. 12, No. 4, 2020, pp. 637–672.

[8] Slotine, J.-J. E., and Li, W., *Applied Nonlinear Control*, Prentice Hall, Englewood Cliffs, NJ, 1991.

[9] Camacho, E. F., and Bordons, C., *Model Predictive Control*, 2nd ed., Springer-Verlag, London, 2004.

[10] Justus, C. G., and Braun, R. D., "Atmospheric Environments for Entry, Descent and Landing (EDL)," *5th International Planetary Probe Workshop*, Bordeaux, France, 2007.

---

*End of report outline.*
