# Model Predictive Guidance (MPG) Algorithm — Implementation Specification

**Source:** Davami, C., Lu, P., and Rosengren, A. J., "Model Predictive Tracking Guidance Applied to Planetary Entry and Powered Descent," AIAA SciTech 2025 Forum, Paper 2025-2599.

---

## Overview

MPG is a receding-horizon trajectory tracking algorithm. The MPC problem is linearized about a pre-generated reference trajectory, discretized using Heun's second-order RK scheme, and cast as a box-constrained quadratic program (QP). At each guidance cycle only the first element of the horizon solution is applied (standard receding-horizon). The algorithm runs in tandem with an offline trajectory planner that supplies the reference.

This implementation uses **OSQP** (via `OSQP.jl`) as the QP solver. OSQP accepts the standard form:

```
minimize    (1/2) z' P z + q' z
subject to  lb ≤ A_con z ≤ ub
```

where `z = du_tilde` is the augmented optimization variable defined below.

**Required Julia packages:**
```julia
using LinearAlgebra
using OSQP
using SparseArrays   # OSQP requires sparse P and A_con
using ForwardDiff    # for Jacobian computation
```

---

## Data Structures & Notation

| Symbol | Julia type | Description |
|---|---|---|
| `x` | `Vector{Float64}` length `n` | State vector |
| `u` | `Vector{Float64}` length `m` | Control vector |
| `x_ref[:, k]` | column of `Matrix{Float64}` | Reference state at node k (1-indexed) |
| `u_ref[:, k]` | column of `Matrix{Float64}` | Reference control at node k |
| `dx` | `Vector{Float64}` length `n` | State deviation: `x - x_ref` |
| `du` | `Vector{Float64}` length `m` | Control excursion: `u - u_ref` |
| `A[k]` | `Matrix{Float64}` `(n,n)` | Linearized dynamics Jacobian w.r.t. state |
| `B[k]` | `Matrix{Float64}` `(n,m)` | Linearized dynamics Jacobian w.r.t. control |
| `H[k]` | `Matrix{Float64}` `(q,n)` | Constraint Jacobian w.r.t. state |
| `Uc[k]` | `Matrix{Float64}` `(q,m)` | Constraint Jacobian w.r.t. control |
| `d[k]` | `Vector{Float64}` length `q` | Constraint residual at reference |
| `N` | `Int` | Number of nodes in prediction horizon (`N > 2`) |
| `Γ` | `Float64` | Horizon length (time or energy units) |
| `h` | `Float64` | Discrete step: `h = Γ / N` |
| `du_hat` | `Vector{Float64}` length `m*(N+1)` | Stacked control excursions over horizon |
| `ζ[k]` | `Vector{Float64}` length `q` | Virtual control (constraint slack) at node k |
| `du_tilde` | `Vector{Float64}` length `(m+q)*(N+1)` | Full optimization variable: `[du_hat; ζ_1; ...; ζ_{N+1}]` |

> **Julia indexing:** All arrays use 1-based indexing. Horizon nodes run from `1` to `N+1`, where node `1` = current time.

---

## Step 1 — Precompute Reference Trajectory Matrices

Given a pre-generated reference `{x_ref(τ), u_ref(τ)}` over the full flight, evaluate and store the following at each node. This is done once offline before guidance begins.

### Dynamics Jacobians

```julia
# For each node k in the full reference grid:
A[k] = ∂f/∂x  evaluated at (x_ref[:, k], u_ref[:, k])   # (n × n)
B[k] = ∂f/∂u  evaluated at (x_ref[:, k], u_ref[:, k])   # (n × m)
```

### Constraint Jacobians (if path constraints η(x,u) ≤ 0 exist)

```julia
H[k]  = ∂η/∂x  evaluated at (x_ref[:, k], u_ref[:, k])  # (q × n)
Uc[k] = ∂η/∂u  evaluated at (x_ref[:, k], u_ref[:, k])  # (q × m)
d[k]  = η(x_ref[:, k], u_ref[:, k])                       # (q,)
```

### Output Matrix (output-tracking variant)

If tracking an output `y = c(x)` rather than the full state:

```julia
C[k] = ∂c/∂x  evaluated at x_ref[:, k]    # (l × n)
# Effective running state weight:
Q_run[k] = C[k]' * Q_y * C[k]             # (n × n)
```

If tracking the full state directly, set `Q_run[k] = Q` for all `k`.

---

## Step 2 — Heun's RK Discretization (Second-Order)

At each guidance cycle, operate over nodes `k = 1, ..., N+1` (node 1 = current time `t`).

### Discrete LTV step matrices

For each interval `i = 1, ..., N` (from node `i` to node `i+1`):

```julia
Iₙ = Matrix{Float64}(I, n, n)

S[i]    = Iₙ + (h/2)*(A[i] + A[i+1]) + (h^2/2)*(A[i+1] * A[i])
P_rk[i] = (h/2) * (Iₙ + h*A[i+1]) * B[i]
Q_rk[i] = (h/2) * B[i+1]
```

These implement: `δx[i+1] = S[i]*δx[i] + P_rk[i]*δu[i] + Q_rk[i]*δu[i+1]`

### State transition Φ and control influence Ψ

`Φ[k]` is `(n × n)`, `Ψ[k]` is `(n × m*(N+1))`. They satisfy:

```
δx[k] = Φ[k] * δx₀ + Ψ[k] * du_hat    for k = 1, ..., N+1
```

Note: `Φ[1] = I` and `Ψ[1] = 0` (node 1 is the current state, no propagation yet).

```julia
Φ = Vector{Matrix{Float64}}(undef, N+1)
Φ[1] = Matrix{Float64}(I, n, n)
for k in 2:N+1
    Φ[k] = S[k-1] * Φ[k-1]
end

Ψ = [zeros(n, m*(N+1)) for _ in 1:N+1]
# Ψ[1] stays zero
for k in 2:N+1
    i = k - 1   # interval index (1-based)

    # Propagate old columns through S[i]
    if k > 2
        Ψ[k][:, 1:m*(i-1)] = S[i] * Ψ[k-1][:, 1:m*(i-1)]
    end

    # P_rk[i] contribution: columns m*(i-1)+1 : m*i
    col_P = (m*(i-1)+1):(m*i)
    prev_col_P = (k > 2) ? S[i] * Ψ[k-1][:, col_P] : zeros(n, m)
    Ψ[k][:, col_P] = prev_col_P + P_rk[i]

    # Q_rk[i] contribution: columns m*i+1 : m*(i+1)
    col_Q = (m*i+1):(m*(i+1))
    Ψ[k][:, col_Q] = Q_rk[i]

    # Columns beyond m*(i+1) remain zero (future controls do not affect δx[k])
end
```

---

## Step 3 — Constraint Discretization

The linearized path constraint `H[k]*δx[k] + Uc[k]*δu[k] + d[k] ≤ 0` is expressed purely in terms of `du_hat` by substituting `δx[k] = Φ[k]*δx₀ + Ψ[k]*du_hat`.

```julia
E  = Vector{Matrix{Float64}}(undef, N+1)
Fc = Vector{Matrix{Float64}}(undef, N+1)

# Node k=1: δx[1] = δx₀, no Ψ contribution
E[1]  = [Uc[1]  zeros(q, m*N)]   # (q × m*(N+1)), Uc[1] in first m columns
Fc[1] = H[1]                      # (q × n)

for k in 2:N+1
    # Uc[k] occupies column block k in du_hat (columns m*(k-1)+1 : m*k)
    Uc_block = zeros(q, m*(N+1))
    Uc_block[:, (m*(k-1)+1):(m*k)] = Uc[k]

    E[k]  = H[k] * Ψ[k] + Uc_block   # (q × m*(N+1))
    Fc[k] = H[k] * Φ[k]               # (q × n)
end
```

The constraint at node k with slack `ζ[k] ≥ 0` is:

```
E[k] * du_hat + Fc[k] * δx₀ + d[k] + ζ[k] = 0,    0 ≤ ζ[k] ≤ ζ_max
```

---

## Step 4 — Assemble the QP

### Augmented optimization variable layout

```julia
nz = (m + q) * (N + 1)   # total decision variable count

# Index helpers (1-based):
du_hat_range(k) = (m*(k-1)+1):(m*k)             # δu at node k
zeta_range(k)   = m*(N+1) + (q*(k-1)+1) : m*(N+1) + q*k   # ζ at node k
```

Layout: `du_tilde = [δu_1; δu_2; ...; δu_{N+1}; ζ_1; ζ_2; ...; ζ_{N+1}]`

### Trapezoidal quadrature weights

```julia
w = fill(h/2, N+1)
w[1]   = h/4   # endpoint half-weight
w[N+1] = h/4
```

### Hessian P_qp (does not depend on δx₀ — precompute offline)

```julia
P_qp = zeros(nz, nz)

# Terminal cost: (kF/2) * δx_{N+1}' * F_mat * δx_{N+1}
# δx_{N+1} = Φ[N+1]*δx₀ + Ψ[N+1]*du_hat  →  contributes Ψ[N+1]'*F_mat*Ψ[N+1] to P
P_qp[1:m*(N+1), 1:m*(N+1)] += kF * (Ψ[N+1]' * F_mat * Ψ[N+1])

for k in 1:N+1
    Ψk    = Ψ[k]   # Ψ[1] = 0 so k=1 contributes nothing to state cost
    du_rng = du_hat_range(k)
    ζ_rng  = zeta_range(k)

    # State tracking cost: w[k] * Ψk' * Q_run[k] * Ψk
    P_qp[1:m*(N+1), 1:m*(N+1)] += w[k] * (Ψk' * Q_run[k] * Ψk)

    # Control regularization: w[k] * kR * R_mat  (block diagonal)
    P_qp[du_rng, du_rng] += w[k] * kR * R_mat

    # Constraint penalty J₂: w[k] * ‖E[k]*du_hat + ζ[k] + Fc[k]*δx₀ + d[k]‖²_{Q_ζ}
    # Expands to three block contributions: du_hat×du_hat, ζ×ζ, and cross terms
    P_qp[1:m*(N+1), 1:m*(N+1)] += w[k] * (E[k]' * Q_zeta * E[k])
    P_qp[ζ_rng, ζ_rng]          += w[k] * Q_zeta
    cross = w[k] * E[k]' * Q_zeta             # (m*(N+1) × q)
    P_qp[1:m*(N+1), ζ_rng]      += cross
    P_qp[ζ_rng, 1:m*(N+1)]      += cross'
end

P_qp = (P_qp + P_qp') / 2   # symmetrize to remove floating-point asymmetry
```

### Linear term q_qp (depends on δx₀ — recompute each cycle)

```julia
q_qp = zeros(nz)

# Terminal cost linear term: kF * Ψ[N+1]' * F_mat * Φ[N+1] * δx₀
q_qp[1:m*(N+1)] += kF * (Ψ[N+1]' * (F_mat * (Φ[N+1] * dx0)))

for k in 1:N+1
    Ψk    = Ψ[k]
    Φkdx0 = Φ[k] * dx0   # Φ[1] = I, so Φ[1]*dx0 = dx0
    ζ_rng  = zeta_range(k)
    rk     = Fc[k] * dx0 + d[k]   # (q,) constant offset for this cycle

    # State tracking linear term
    q_qp[1:m*(N+1)] += w[k] * (Ψk' * (Q_run[k] * Φkdx0))

    # Constraint penalty linear terms
    q_qp[1:m*(N+1)] += w[k] * (E[k]' * (Q_zeta * rk))
    q_qp[ζ_rng]     += w[k] * (Q_zeta * rk)
end
```

### Bounds (box constraints encoded as `lb ≤ I*z ≤ ub`)

```julia
lb = Vector{Float64}(undef, nz)
ub = Vector{Float64}(undef, nz)

# Control excursion bounds (update du_min/du_max each cycle)
du_min = u_min - u_ref_current   # (m,)
du_max = u_max - u_ref_current   # (m,)

for k in 1:N+1
    lb[du_hat_range(k)] .= du_min
    ub[du_hat_range(k)] .= du_max
    lb[zeta_range(k)]   .= 0.0
    ub[zeta_range(k)]   .= ζ_max   # (q,) user-defined large upper bound
end

A_con = sparse(I, nz, nz)   # identity: OSQP interprets lb ≤ A_con*z ≤ ub as box
```

---

## Step 5 — Solve with OSQP

```julia
using OSQP, SparseArrays

# Build model once (outside guidance loop), update q and bounds each cycle
function build_osqp_model(P_qp, nz)
    P_sparse = sparse(triu(P_qp))          # OSQP takes upper triangular of P
    A_con    = sparse(I, nz, nz)
    model    = OSQP.Model()
    # Placeholder q, lb, ub — will be updated each cycle via OSQP.update!
    OSQP.setup!(model, P_sparse, zeros(nz), A_con,
                fill(-Inf, nz), fill(Inf, nz);
                warm_starting = true,
                eps_abs       = 1e-6,
                eps_rel       = 1e-6,
                max_iter      = 4000,
                verbose       = false,
                polish        = true)
    return model
end

# Each guidance cycle: update and solve
function solve_cycle!(model, q_qp, lb, ub; warm_x=nothing)
    OSQP.update!(model; q=q_qp, l=lb, u=ub)
    if warm_x !== nothing
        OSQP.warm_start!(model; x=warm_x)
    end
    result = OSQP.solve!(model)
    return result.x, result.info.status
end
```

> **Solver settings:** `polish=true` activates a post-solve refinement step that substantially improves accuracy for the small dense QPs typical here (problem size `nz ≈ 50–200`). `eps_abs = eps_rel = 1e-6` matches the paper's convergence target. Using `OSQP.update!` instead of `OSQP.setup!` at every cycle avoids re-factoring the KKT system and is strongly recommended.

---

## Step 6 — Extract and Apply Control

```julia
du_tilde_star, status = solve_cycle!(model, q_qp, lb, ub; warm_x=du_tilde_prev)

# First m elements of du_tilde are δu at the current node
du_star   = du_tilde_star[1:m]
u_applied = clamp.(u_ref_current + du_star, u_min, u_max)

# Warm-start for next cycle: shift by one node, append zeros
du_tilde_prev = [du_tilde_star[(m+q)+1:end]; zeros(m+q)]
```

---

## Full Guidance Cycle (Reference Implementation)

```julia
"""
    mpg_cycle!(model, x_current, k0, ref, params, du_tilde_prev)

One MPG guidance cycle using a pre-built OSQP model.
Returns (u_applied, du_tilde_shifted).

`ref` is a NamedTuple with fields:
    x_ref :: Matrix{Float64}    (n × T_total)
    u_ref :: Matrix{Float64}    (m × T_total)
    A, B  :: Vector of matrices  precomputed Jacobians
    H, Uc, d :: Vector of matrices/vectors  constraint Jacobians
    C     :: Vector of matrices  output Jacobians

`params` is a NamedTuple with fields:
    n, m, q, l, N, Γ, h,
    kF, kR, F_mat, Q_y, R_mat, Q_zeta, ζ_max,
    u_min, u_max
"""
function mpg_cycle!(model, x_current, k0, ref, params, du_tilde_prev)
    (; n, m, q, N, h, kF, kR, F_mat, Q_y, R_mat, Q_zeta, ζ_max, u_min, u_max) = params

    dx0            = x_current - ref.x_ref[:, k0]
    u_ref_current  = ref.u_ref[:, k0]

    # Extract horizon window (nodes k0 to k0+N, 1-indexed)
    idx = k0:(k0 + N)
    A    = ref.A[idx];  B  = ref.B[idx]
    H    = ref.H[idx];  Uc = ref.Uc[idx];  d = ref.d[idx]
    C    = ref.C[idx]

    # Output-tracking state weights
    Q_run = [C[k]' * Q_y * C[k] for k in 1:N+1]

    # Heun's RK step matrices (intervals 1..N)
    Iₙ    = Matrix{Float64}(I, n, n)
    S     = [Iₙ + (h/2)*(A[i]+A[i+1]) + (h^2/2)*(A[i+1]*A[i])  for i in 1:N]
    P_rk  = [(h/2)*(Iₙ + h*A[i+1])*B[i]                          for i in 1:N]
    Q_rk  = [(h/2)*B[i+1]                                          for i in 1:N]

    # State transition Φ and control influence Ψ
    Φ = Vector{Matrix{Float64}}(undef, N+1)
    Φ[1] = Iₙ
    for k in 2:N+1; Φ[k] = S[k-1] * Φ[k-1]; end

    Ψ = [zeros(n, m*(N+1)) for _ in 1:N+1]
    for k in 2:N+1
        i = k - 1
        if k > 2
            Ψ[k][:, 1:m*(i-1)] = S[i] * Ψ[k-1][:, 1:m*(i-1)]
        end
        col_P = (m*(i-1)+1):(m*i)
        Ψ[k][:, col_P] = (k > 2 ? S[i]*Ψ[k-1][:, col_P] : zeros(n, m)) + P_rk[i]
        Ψ[k][:, (m*i+1):(m*(i+1))] = Q_rk[i]
    end

    # Constraint matrices E, Fc
    E  = Vector{Matrix{Float64}}(undef, N+1)
    Fc = Vector{Matrix{Float64}}(undef, N+1)
    E[1]  = [Uc[1] zeros(q, m*N)];  Fc[1] = H[1]
    for k in 2:N+1
        blk = zeros(q, m*(N+1))
        blk[:, (m*(k-1)+1):(m*k)] = Uc[k]
        E[k]  = H[k] * Ψ[k] + blk
        Fc[k] = H[k] * Φ[k]
    end

    # Assemble q_qp (P_qp precomputed; only q_qp changes each cycle)
    nz   = (m + q) * (N + 1)
    q_qp = zeros(nz)
    w    = fill(h/2, N+1); w[1] = h/4; w[N+1] = h/4
    q_qp[1:m*(N+1)] += kF * (Ψ[N+1]' * (F_mat * (Φ[N+1] * dx0)))
    for k in 1:N+1
        Φkdx0 = Φ[k] * dx0
        rk    = Fc[k] * dx0 + d[k]
        ζ_rng = m*(N+1) + (q*(k-1)+1) : m*(N+1) + q*k
        q_qp[1:m*(N+1)] += w[k] * (Ψ[k]' * (Q_run[k] * Φkdx0)
                                  + E[k]' * (Q_zeta  * rk))
        q_qp[ζ_rng]     += w[k] * (Q_zeta * rk)
    end

    # Update bounds
    du_min = u_min - u_ref_current
    du_max = u_max - u_ref_current
    lb = Vector{Float64}(undef, nz)
    ub = Vector{Float64}(undef, nz)
    for k in 1:N+1
        dh_rng = (m*(k-1)+1):(m*k)
        ζ_rng  = m*(N+1) + (q*(k-1)+1) : m*(N+1) + q*k
        lb[dh_rng] .= du_min;  ub[dh_rng] .= du_max
        lb[ζ_rng]  .= 0.0;    ub[ζ_rng]  .= ζ_max
    end

    # Solve
    du_tilde_star, _ = solve_cycle!(model, q_qp, lb, ub; warm_x=du_tilde_prev)

    # Extract control
    du_star          = du_tilde_star[1:m]
    u_applied        = clamp.(u_ref_current + du_star, u_min, u_max)
    du_tilde_shifted = [du_tilde_star[(m+q)+1:end]; zeros(m+q)]

    return u_applied, du_tilde_shifted
end
```

---

## Tuning Parameters

| Parameter | Symbol | Role | Guidance |
|---|---|---|---|
| Horizon length | `Γ` | Prediction horizon duration | Longer → more stable, larger QP; Lunar: 8 s, Mars PD: 2 s |
| Horizon nodes | `N` | Discretization resolution | `N > 2`; QP size grows as `(m+q)²(N+1)²` |
| Terminal cost gain | `kF` | Endpoint state penalty scalar | Large `kF` can substitute for large `Γ` |
| State weight | `Q_y` | Running output penalty (`l×l` diagonal) | Tune per tracked output component |
| Control weight | `kR`, `R_mat` | Running control effort penalty | Larger → smoother control excursions |
| Constraint penalty | `Q_zeta` | Penalizes constraint violation | Large enough to drive `ζ → 0` when feasible |
| Virtual control bound | `ζ_max` | Upper bound on constraint slack | Large enough to never bind; e.g. `1e4 * ones(q)` |
| OSQP polish | `polish=true` | Post-solve refinement | Always enable for small problems |

---

## Application: Powered Descent (3-DOF)

### State and control vectors

```julia
# State: position + velocity in topocentric NUE frame (n = 6)
x = [r_N, r_U, r_E, v_N, v_U, v_E]

# Control: throttle, azimuth, elevation (m = 3)
u = [T, θ, ϕ]
# Mass is propagated externally and not included in the tracking state
```

### Equations of motion

```julia
function powered_descent_eom(x, u, mass, g_vec, Ve)
    r, v = x[1:3], x[4:6]
    T, θ, ϕ = u
    thrust_dir = [cos(ϕ)*cos(θ), sin(ϕ), cos(ϕ)*sin(θ)]
    r_dot  = v
    v_dot  = (T / mass) .* thrust_dir + g_vec
    ṁ      = -T / Ve
    return [r_dot; v_dot], ṁ
end
```

### Linearized dynamics Jacobians

```julia
function powered_descent_jacobians(x_ref, u_ref, mass)
    T, θ, ϕ = u_ref
    A = zeros(6, 6)
    A[1:3, 4:6] = I(3)                          # ṙ = v

    B = zeros(6, 3)
    B[4:6, 1] = [cos(ϕ)*cos(θ), sin(ϕ), cos(ϕ)*sin(θ)] ./ mass     # ∂v̇/∂T
    B[4:6, 2] = (T/mass) .* [-cos(ϕ)*sin(θ), 0.0,  cos(ϕ)*cos(θ)]  # ∂v̇/∂θ
    B[4:6, 3] = (T/mass) .* [-sin(ϕ)*cos(θ), cos(ϕ), -sin(ϕ)*sin(θ)] # ∂v̇/∂ϕ
    return A, B
end
```

### Control bounds

```julia
u_min = [T_min, -θ_max, -ϕ_max]
u_max = [T_max,  θ_max,  ϕ_max]
# Per-cycle excursion bounds (recomputed each cycle):
du_min = u_min .- u_ref_current
du_max = u_max .- u_ref_current
```

### Path constraints

**Glide-slope** (pure state, `q = 1`):

```julia
function glide_slope_jacobians(x_ref, Θ_G, ê_up)
    r_vec = x_ref[1:3]
    r     = norm(r_vec)
    η     = r * cos(Θ_G) - dot(r_vec, ê_up)
    H_row = (cos(Θ_G) .* r_vec ./ r .- ê_up)'   # (1 × 3) position part
    H     = [H_row  zeros(1, 3)]                  # (1 × 6) full state
    Uc    = zeros(1, 3)
    return η, H, Uc
end
```

**Pointing constraint** (pure control, `q = 1`):

```julia
function pointing_jacobians(u_ref, Θ_P, ê_up)
    # Use ForwardDiff for Uc to avoid transcription errors
    η_fn = u -> cos(Θ_P) - dot([cos(u[3])*cos(u[2]), sin(u[3]), cos(u[3])*sin(u[2])], ê_up)
    η    = η_fn(u_ref)
    Uc   = ForwardDiff.gradient(η_fn, u_ref)'   # (1 × 3)
    H    = zeros(1, 6)
    return η, H, Uc
end
```

---

## Application: Atmospheric Entry (3-DOF, Bank-Angle Steering)

### Independent variable

Use **specific energy** `e = 1/r - V²/2` (monotonically increasing during entry). All primes (`'`) denote `d/de`.

```julia
# Nondimensionalization constants (Mars example):
const R0      = 3_396_200.0   # equatorial radius [m]
const g0      = 3.71          # surface gravity [m/s²]
const t_scale = sqrt(R0/g0)
const V_scale = sqrt(g0*R0)
# Derived velocity from energy:  V = sqrt(2*(1/r - e))
```

### State vector

```julia
x = [r, θ, ϕ, γ, ψ]   # n = 5
# r: nondim radial distance from planet center
# θ, ϕ: longitude, latitude
# γ: flight-path angle (positive above local horizontal)
# ψ: heading angle (clockwise from north)
```

### Equations of motion w.r.t. energy `e`

```julia
function entry_eom(x, σ, e, aero)
    r, θ, ϕ, γ, ψ = x
    V = sqrt(max(2*(1/r - e), 0.0))
    L, D = aero_forces(r, V, aero)
    return [
        sin(γ) / D,
        cos(γ)*sin(ψ) / (r*D*cos(ϕ)),
        cos(γ)*cos(ψ) / (r*D),
        (L*cos(σ) + (V^2 - 1/r)*cos(γ)/r) / (D*V^2),
        (L*sin(σ)/cos(γ) + (V^2/r)*cos(γ)*sin(ψ)*tan(ϕ)) / (D*V^2)
    ]
end

function aero_forces(r, V, aero)
    ρ = aero.density_model(r)   # user-supplied ρ(r), dimensional [kg/m³]
    q_dyn = ρ * V^2             # nondim dynamic pressure factor
    L = q_dyn * aero.Sref * R0 * aero.CL / (2 * aero.m0)
    D = q_dyn * aero.Sref * R0 * aero.CD / (2 * aero.m0)
    return L, D
end
```

### Jacobians via ForwardDiff

```julia
function entry_jacobians(x_ref, σ_ref, e, aero)
    A = ForwardDiff.jacobian(x -> entry_eom(x, σ_ref, e, aero), x_ref)  # (5×5)
    b = ForwardDiff.derivative(σ -> entry_eom(x_ref, σ, e, aero), σ_ref) # (5,)
    return A, reshape(b, 5, 1)   # B is (5×1) for single bank-angle control
end
```

> Use `ForwardDiff.jl` for all entry Jacobians. The equations are complex enough that manual derivation errors are likely. Verify against finite differences: `(f(x+ε*eᵢ) - f(x-ε*eᵢ)) / (2ε)` with `ε = 1e-6`.

### Two-Phase Sequential Tracking Strategy

**Phase 1 — Longitudinal tracking** (from entry interface to `e_switch = 0.8`):

```julia
# Reduced state (n_lon = 3), single bank-angle control (m = 1)
# s = great-circle range-to-go [radians]
function entry_lon_eom(x_lon, σ, e, aero)
    r, s, γ = x_lon
    V = sqrt(max(2*(1/r - e), 0.0))
    L, D = aero_forces(r, V, aero)
    return [
        sin(γ) / D,
        -cos(γ) / (r*D),
        (L*cos(σ) + (V^2 - 1/r)*cos(γ)/r) / (D*V^2)
    ]
end

# Output: y = [r, s],  y* = [r*(e), s*(e)]
# C_lon = [1 0 0; 0 1 0]  →  Q_run = C_lon' * Q_y_lon * C_lon  (Q_y_lon is 2×2)
```

**Phase 2 — Longitudinal + lateral tracking** (from `e_switch` to `e_f`):

```julia
# Full 5-state dynamics, two controls: u = [σ, α]  (m = 2)
function entry_2ctrl_jacobians(x_ref, u_ref, e, aero)
    σ_ref, α_ref = u_ref
    # Compute B for α via finite difference on CL, CD
    A = ForwardDiff.jacobian(x -> entry_2ctrl_eom(x, u_ref, e, aero), x_ref)
    B = ForwardDiff.jacobian(u -> entry_2ctrl_eom(x_ref, u, e, aero), u_ref)
    return A, B   # (5×5), (5×2)
end

# Aerodynamic coefficient linearization around α*:
function aero_alpha_slope(CL_fn, CD_fn, α_ref, α_min, α_max)
    dCL_dα = 0.5 * ((CL_fn(α_max) - CL_fn(α_ref))/(α_max - α_ref) +
                     (CL_fn(α_min) - CL_fn(α_ref))/(α_min - α_ref))
    dCD_dα = 0.5 * ((CD_fn(α_max) - CD_fn(α_ref))/(α_max - α_ref) +
                     (CD_fn(α_min) - CD_fn(α_ref))/(α_min - α_ref))
    return dCL_dα, dCD_dα
end

# Great-circle azimuth and range to landing site:
function great_circle_azimuth_range(θ, ϕ, Θ_tgt, Φ_tgt)
    s = acos(clamp(sin(ϕ)*sin(Φ_tgt) + cos(ϕ)*cos(Φ_tgt)*cos(Θ_tgt-θ), -1.0, 1.0))
    Ψ = asin(clamp(sin(Θ_tgt-θ)*cos(Φ_tgt)/sin(s), -1.0, 1.0))
    return Ψ, s
end

# Output Jacobian C (3×5) for Phase 2:
function entry_output_jacobian(x_ref, Θ_tgt, Φ_tgt)
    θ, ϕ = x_ref[2], x_ref[3]
    # Use ForwardDiff on the scalar functions s(θ,ϕ) and Ψ(θ,ϕ)
    s_fn = v -> acos(clamp(sin(v[2])*sin(Φ_tgt)+cos(v[2])*cos(Φ_tgt)*cos(Θ_tgt-v[1]),-1,1))
    Ψ_fn = v -> begin
        sv = s_fn(v)
        asin(clamp(sin(Θ_tgt-v[1])*cos(Φ_tgt)/sin(sv), -1, 1))
    end
    ∂s = ForwardDiff.gradient(s_fn, [θ, ϕ])   # [∂s/∂θ, ∂s/∂ϕ]
    ∂Ψ = ForwardDiff.gradient(Ψ_fn, [θ, ϕ])   # [∂Ψ/∂θ, ∂Ψ/∂ϕ]
    C = [1.0    0.0      0.0      0.0  0.0;
         0.0    ∂s[1]    ∂s[2]    0.0  0.0;
         0.0   -∂Ψ[1]  -∂Ψ[2]    0.0  1.0]
    return C   # (3×5)
end
```

**Phase switching logic:**

```julia
# Switch from Phase 1 to Phase 2 at e_switch = 0.8 (dimensionless)
function entry_phase(e_current; e_switch=0.8)
    return e_current < e_switch ? :longitudinal : :full
end
```

---

## Numerical Implementation Notes

1. **Precompute `P_qp` offline.** `P_qp` does not depend on `dx0`. If reference Jacobians are stored for the full flight, build `P_qp` once before simulation begins. Only `q_qp` and bounds change each cycle and require re-assembly.

2. **Use `OSQP.update!` not `OSQP.setup!` in the guidance loop.** Calling `setup!` repeats the KKT factorization at every cycle. Use `update!` to modify `q`, `l`, and `u` only, keeping the factorization intact:
   ```julia
   OSQP.update!(model; q=q_qp, l=lb, u=ub)
   result = OSQP.solve!(model)
   ```

3. **QP size.** `nz = (m+q)*(N+1)`. For `m=3, q=2, N=10`: `nz=55`. Even `N=50` gives `nz=255` — trivially small for OSQP. The bottleneck will be the matrix multiplications in assembling `q_qp`, not the solve.

4. **Warm starting.** After each solve, shift the solution vector by `m+q` elements (one full node) and append zeros:
   ```julia
   du_tilde_prev = [du_tilde_star[(m+q)+1:end]; zeros(m+q)]
   ```
   Pass this as the warm start to `OSQP.warm_start!` at the next cycle.

5. **`q_qp` assembly cost.** The dominant operation is the `Ψ[k]' * Q_run[k] * Φ[k] * dx0` term, which is `O(m*(N+1) * n)` per node. For `N=10, n=6, m=3` this is ~200 multiplications — negligible. No optimization needed at this scale.

6. **Sparse matrices.** Pass `sparse(triu(P_qp))` to OSQP (upper triangular of symmetric `P_qp` only). The constraint matrix `A_con = sparse(I, nz, nz)` is trivially sparse. Constructing `P_sparse` once offline and reusing it is sufficient since `P_qp` does not change.

7. **Horizon window bounds check.** At each cycle, verify `k0 + N ≤ T_total` before indexing. Near the end of the trajectory, clamp the horizon or use the last available reference node for the tail:
   ```julia
   k0 = min(k0, T_total - N)
   ```

8. **Dimensionless entry units.** All entry quantities must be normalized before passing to the QP. States `r`, `V`, and `e` use the nondimensionalization above. Angles are already dimensionless (radians). Verify by checking that `e` increases monotonically and stays in the range `[0.0, ~0.5]` for typical Mars entry.

9. **`drho/dr` in linearization.** The `dL/dr` and `dD/dr` entries in `A` have two contributions: a velocity term `(2L/V)*dV/dr` and a density gradient term `(L/ρ)*dρ/dr`. The velocity term dominates at hypersonic speeds. Verify the ratio `|(L/ρ)*dρ/dr| / |(2L/V)*dV/dr| ≪ 1` for your atmospheric model; if true, omit the density gradient term from the analytical Jacobian. `ForwardDiff` will include it automatically.

10. **`ForwardDiff` for Jacobians.** Use `ForwardDiff.jacobian` during development for all dynamics. This is correct-by-construction and avoids transcription errors in complex equations (especially entry). Switch to hand-coded analytic Jacobians only if profiling identifies them as a bottleneck — unlikely at `n ≤ 6` dimensions.

---

## Validation Tests

```julia
# 1. KKT stationarity (for active-set interior):
#    P_qp * du_star + q_qp should be near zero at unconstrained optimum
residual = norm(Matrix(P_qp) * du_tilde_star + q_qp)
@test residual < 1e-4

# 2. OSQP status check at every guidance cycle:
@test status ∈ (:Solved, :Solved_inaccurate)

# 3. Constraint satisfaction after initial transient (~10–40 s):
η_val = η_fn(x_sim, u_applied)
@test all(η_val .≤ 1e-3)   # small tolerance for linearization error

# 4. State deviation convergence (powered descent):
@test norm(x_sim - x_ref[:, end]) < 50.0   # meters at terminal node

# 5. Monte Carlo targets from paper (1000 cases, 3-sigma IC dispersions):
#    Lunar PD:        99.8% within 20 m miss distance, all within 40 m
#    Mars end-to-end: 99.7% within 5 m, all within 36 m
```
