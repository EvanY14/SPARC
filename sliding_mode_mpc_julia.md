# Sliding-Mode MPC Without Integrator (LTV) — Julia / JuMP Implementation Notes

This note gives a solver-ready formulation of **tracking sliding-mode MPC without integral action** for a **discrete-time linear time-varying (LTV)** system, in a form that can be implemented in **Julia** using **JuMP** and a QP solver such as **OSQP** or **Clarabel**.

The formulation below uses:

- a tracking-error state,
- a linear sliding variable,
- input increments as the optimization variable,
- a condensed QP form,
- linear inequality constraints.

It is written so a code-generation tool can translate it directly into Julia.

---

## 1. Problem setup

Plant:
\[
x_{k+1} = A_k x_k + B_k u_k,\qquad y_k = C_k x_k
\]

Reference trajectory:
\[
x^r_{k+1} = A_k x^r_k + B_k u^r_k,\qquad y^r_k = C_k x^r_k
\]

Tracking errors:
\[
e_k := x_k - x^r_k,\qquad v_k := u_k - u^r_k
\]

Then the error dynamics are:
\[
e_{k+1} = A_k e_k + B_k v_k
\]

Since there is **no integrator**, the tracking state is just:
\[
z_k := e_k
\]

---

## 2. Sliding variable

Choose a linear sliding variable:
\[
s_k = G_k e_k
\]

Dimensions:

- \(e_k \in \mathbb{R}^{n_x}\)
- \(s_k \in \mathbb{R}^{n_s}\)
- \(G_k \in \mathbb{R}^{n_s \times n_x}\)

`G_k` is a design matrix. In many cases it is constant, so \(G_k = G\).

Example for a 2-state SISO tracking problem:
\[
e_k = \begin{bmatrix} e_{p,k} \\ e_{v,k} \end{bmatrix},\qquad
s_k = \begin{bmatrix}\lambda & 1\end{bmatrix} e_k
\]

---

## 3. Input increment augmentation

Use input increments as the optimization variables:
\[
\Delta v_k := v_k - v_{k-1}
\]

Augment the state with the previous input error:
\[
\chi_k :=
\begin{bmatrix}
e_k \\
v_{k-1}
\end{bmatrix}
\in \mathbb{R}^{n_x + n_u}
\]

Then:
\[
\chi_{k+1} = A_{\chi,k}\chi_k + B_{\chi,k}\Delta v_k
\]

with
\[
A_{\chi,k} =
\begin{bmatrix}
A_k & B_k \\
0 & I
\end{bmatrix},
\qquad
B_{\chi,k} =
\begin{bmatrix}
B_k \\
I
\end{bmatrix}
\]

This is the key augmentation that makes:

- input rate constraints linear,
- input magnitude constraints linear,
- control smoothness penalties quadratic.

---

## 4. Horizon stacking

Let the horizon length be \(N\).

Decision vector:
\[
\Delta U_k :=
\begin{bmatrix}
\Delta v_{k|k} \\
\Delta v_{k+1|k} \\
\vdots \\
\Delta v_{k+N-1|k}
\end{bmatrix}
\in \mathbb{R}^{N n_u}
\]

Predicted augmented state stack:
\[
\mathcal{X}_k :=
\begin{bmatrix}
\chi_{k+1|k} \\
\chi_{k+2|k} \\
\vdots \\
\chi_{k+N|k}
\end{bmatrix}
\]

For the LTV system:
\[
\mathcal{X}_k = \Phi_k \chi_k + \Gamma_k \Delta U_k
\]

### 4.1 Transition product

Define the LTV transition matrix:
\[
\Phi_\chi(\ell,k) :=
\begin{cases}
A_{\chi,\ell-1}A_{\chi,\ell-2}\cdots A_{\chi,k}, & \ell > k\\
I, & \ell = k
\end{cases}
\]

Then:
\[
\chi_{k+i|k}
=
\Phi_\chi(k+i,k)\chi_k
+
\sum_{j=0}^{i-1}
\Phi_\chi(k+i,k+j+1)\,B_{\chi,k+j}\,\Delta v_{k+j|k}
\]

for \(i = 1,\dots,N\).

### 4.2 Block matrices

Define \(\Phi_k\) and \(\Gamma_k\) so that:
\[
\mathcal{X}_k = \Phi_k \chi_k + \Gamma_k \Delta U_k
\]

where:

\[
\Phi_k =
\begin{bmatrix}
\Phi_\chi(k+1,k) \\
\Phi_\chi(k+2,k) \\
\vdots \\
\Phi_\chi(k+N,k)
\end{bmatrix}
\]

and \(\Gamma_k\) is block lower triangular with block \((i,j)\):
\[
[\Gamma_k]_{ij} =
\begin{cases}
\Phi_\chi(k+i,k+j)\,B_{\chi,k+j-1}, & j \le i \\
0, & j > i
\end{cases}
\]

for \(i,j = 1,\dots,N\).

---

## 5. Extract predicted errors, inputs, and sliding variables

Define selection matrices:

\[
M_e = \begin{bmatrix} I_{n_x} & 0 \end{bmatrix}
\]

\[
M_v = \begin{bmatrix} 0 & I_{n_u} \end{bmatrix}
\]

where \(M_e\) extracts the tracking error \(e\) from \(\chi\), and \(M_v\) extracts the input error \(v\).

### 5.1 Predicted error stack

\[
\mathcal{E}_k :=
\begin{bmatrix}
e_{k+1|k} \\
e_{k+2|k} \\
\vdots \\
e_{k+N|k}
\end{bmatrix}
=
\Phi_{e,k}\chi_k + \Gamma_{e,k}\Delta U_k
\]

with
\[
\Phi_{e,k} = (I_N \otimes M_e)\Phi_k,\qquad
\Gamma_{e,k} = (I_N \otimes M_e)\Gamma_k
\]

### 5.2 Predicted input-error stack

\[
\mathcal{V}_k :=
\begin{bmatrix}
v_{k|k} \\
v_{k+1|k} \\
\vdots \\
v_{k+N-1|k}
\end{bmatrix}
=
\Phi_{v,k}\chi_k + \Gamma_{v,k}\Delta U_k
\]

with
\[
\Phi_{v,k} = (I_N \otimes M_v)\Phi_k,\qquad
\Gamma_{v,k} = (I_N \otimes M_v)\Gamma_k
\]

### 5.3 Predicted sliding-variable stack

Define
\[
\bar G_k := \operatorname{blkdiag}(G_{k+1},G_{k+2},\dots,G_{k+N})
\]

Then
\[
\mathcal{S}_k :=
\begin{bmatrix}
s_{k+1|k} \\
s_{k+2|k} \\
\vdots \\
s_{k+N|k}
\end{bmatrix}
=
\bar G_k \mathcal{E}_k
=
\Phi_{s,k}\chi_k + \Gamma_{s,k}\Delta U_k
\]

with
\[
\Phi_{s,k} = \bar G_k \Phi_{e,k},\qquad
\Gamma_{s,k} = \bar G_k \Gamma_{e,k}
\]

### 5.4 Predicted output-error stack

Define
\[
\bar C_k := \operatorname{blkdiag}(C_{k+1},C_{k+2},\dots,C_{k+N})
\]

Then
\[
\mathcal{Y}^{err}_k
=
\bar C_k \mathcal{E}_k
=
\Phi_{y,k}\chi_k + \Gamma_{y,k}\Delta U_k
\]

with
\[
\Phi_{y,k} = \bar C_k \Phi_{e,k},\qquad
\Gamma_{y,k} = \bar C_k \Gamma_{e,k}
\]

---

## 6. Actual predicted inputs, states, outputs

Reference stacks:
\[
\mathcal{X}^r_k =
\begin{bmatrix}
x^r_{k+1} \\
x^r_{k+2} \\
\vdots \\
x^r_{k+N}
\end{bmatrix},\qquad
\mathcal{U}^r_k =
\begin{bmatrix}
u^r_k \\
u^r_{k+1} \\
\vdots \\
u^r_{k+N-1}
\end{bmatrix}
\]

\[
\mathcal{Y}^r_k =
\begin{bmatrix}
y^r_{k+1} \\
y^r_{k+2} \\
\vdots \\
y^r_{k+N}
\end{bmatrix}
\]

Then:
\[
\mathcal{X}^{act}_k = \mathcal{X}^r_k + \mathcal{E}_k
\]
\[
\mathcal{U}^{act}_k = \mathcal{U}^r_k + \mathcal{V}_k
\]
\[
\mathcal{Y}^{act}_k = \mathcal{Y}^r_k + \mathcal{Y}^{err}_k
\]

---

## 7. Cost function

A practical cost is:
\[
J_k
=
\sum_{j=1}^{N}
s_{k+j|k}^\top Q_{s,k+j} s_{k+j|k}
+
\sum_{j=0}^{N-1}
v_{k+j|k}^\top R_{v,k+j} v_{k+j|k}
+
\sum_{j=0}^{N-1}
\Delta v_{k+j|k}^\top R_{\Delta,k+j}\Delta v_{k+j|k}
+
e_{k+N|k}^\top P_{k+N} e_{k+N|k}
\]

where:

- \(Q_{s,k+j} \succeq 0\)
- \(R_{v,k+j} \succeq 0\)
- \(R_{\Delta,k+j} \succ 0\)
- \(P_{k+N} \succeq 0\)

Define block-diagonal weights:
\[
\bar Q_{s,k} = \operatorname{blkdiag}(Q_{s,k+1},\dots,Q_{s,k+N})
\]
\[
\bar R_{v,k} = \operatorname{blkdiag}(R_{v,k},\dots,R_{v,k+N-1})
\]
\[
\bar R_{\Delta,k} = \operatorname{blkdiag}(R_{\Delta,k},\dots,R_{\Delta,k+N-1})
\]

Define a selector \(E_N\) that extracts the last block of \(\mathcal{X}_k\). Then:
\[
e_{k+N|k} = \Phi_{N,k}\chi_k + \Gamma_{N,k}\Delta U_k
\]
with
\[
\Phi_{N,k} = M_e E_N \Phi_k,\qquad
\Gamma_{N,k} = M_e E_N \Gamma_k
\]

Substitute into the cost:
\[
J_k = \frac12 \Delta U_k^\top H_k \Delta U_k + h_k^\top \Delta U_k + c_k
\]

with
\[
H_k =
2\left(
\Gamma_{s,k}^\top \bar Q_{s,k}\Gamma_{s,k}
+
\Gamma_{v,k}^\top \bar R_{v,k}\Gamma_{v,k}
+
\bar R_{\Delta,k}
+
\Gamma_{N,k}^\top P_{k+N}\Gamma_{N,k}
\right)
\]

\[
h_k =
2\left(
\Gamma_{s,k}^\top \bar Q_{s,k}\Phi_{s,k}
+
\Gamma_{v,k}^\top \bar R_{v,k}\Phi_{v,k}
+
\Gamma_{N,k}^\top P_{k+N}\Phi_{N,k}
\right)\chi_k
\]

The constant term \(c_k\) can be ignored.

---

## 8. Constraints

All constraints are linear in \(\Delta U_k\).

### 8.1 Input increment bounds

\[
\Delta v_{\min,k+j} \le \Delta v_{k+j|k} \le \Delta v_{\max,k+j}
\]

Stack:
\[
\underline{\Delta V}_k \le \Delta U_k \le \overline{\Delta V}_k
\]

Equivalent inequality form:
\[
\begin{bmatrix}
I\\
-I
\end{bmatrix}\Delta U_k
\le
\begin{bmatrix}
\overline{\Delta V}_k\\
-\underline{\Delta V}_k
\end{bmatrix}
\]

### 8.2 Input magnitude bounds

\[
\underline U_k \le \mathcal{U}^{act}_k \le \overline U_k
\]

Since
\[
\mathcal{U}^{act}_k = \mathcal{U}^r_k + \Phi_{v,k}\chi_k + \Gamma_{v,k}\Delta U_k
\]

we get
\[
\begin{bmatrix}
\Gamma_{v,k}\\
-\Gamma_{v,k}
\end{bmatrix}\Delta U_k
\le
\begin{bmatrix}
\overline U_k - \mathcal{U}^r_k - \Phi_{v,k}\chi_k\\
-\underline U_k + \mathcal{U}^r_k + \Phi_{v,k}\chi_k
\end{bmatrix}
\]

### 8.3 State bounds

\[
\underline X_k \le \mathcal{X}^{act}_k \le \overline X_k
\]

Since
\[
\mathcal{X}^{act}_k = \mathcal{X}^r_k + \Phi_{e,k}\chi_k + \Gamma_{e,k}\Delta U_k
\]

we get
\[
\begin{bmatrix}
\Gamma_{e,k}\\
-\Gamma_{e,k}
\end{bmatrix}\Delta U_k
\le
\begin{bmatrix}
\overline X_k - \mathcal{X}^r_k - \Phi_{e,k}\chi_k\\
-\underline X_k + \mathcal{X}^r_k + \Phi_{e,k}\chi_k
\end{bmatrix}
\]

### 8.4 Output bounds

\[
\underline Y_k \le \mathcal{Y}^{act}_k \le \overline Y_k
\]

Since
\[
\mathcal{Y}^{act}_k = \mathcal{Y}^r_k + \Phi_{y,k}\chi_k + \Gamma_{y,k}\Delta U_k
\]

we get
\[
\begin{bmatrix}
\Gamma_{y,k}\\
-\Gamma_{y,k}
\end{bmatrix}
\Delta U_k
\le
\begin{bmatrix}
\overline Y_k - \mathcal{Y}^r_k - \Phi_{y,k}\chi_k\\
-\underline Y_k + \mathcal{Y}^r_k + \Phi_{y,k}\chi_k
\end{bmatrix}
\]

---

## 9. Final condensed QP

At each time step \(k\), solve:
\[
\min_{\Delta U_k}\quad \frac12 \Delta U_k^\top H_k \Delta U_k + h_k^\top \Delta U_k
\]
subject to
\[
A_{qp,k}\Delta U_k \le b_{qp,k}
\]

where \(A_{qp,k}\) and \(b_{qp,k}\) are formed by stacking the constraint blocks above.

After solving, apply only the first control move:
\[
v_k^\star = v_{k-1} + \Delta v_{k|k}^\star
\]
\[
u_k^\star = u_k^r + v_k^\star
\]

Then shift the horizon and repeat.

---

## 10. Julia implementation plan

A clean implementation can be organized into these functions:

1. `build_augmented_matrices(Ak, Bk)`  
   Build \(A_{\chi,k}\) and \(B_{\chi,k}\) for each horizon step.

2. `build_prediction_matrices(Aχ_seq, Bχ_seq)`  
   Build \(\Phi_k\) and \(\Gamma_k\).

3. `build_extraction_matrices(Φ, Γ, C_seq, G_seq, nx, nu, N)`  
   Build \(\Phi_{e,k}, \Gamma_{e,k}, \Phi_{v,k}, \Gamma_{v,k}, \Phi_{s,k}, \Gamma_{s,k}, \Phi_{y,k}, \Gamma_{y,k}, \Phi_{N,k}, \Gamma_{N,k}\).

4. `build_cost(...)`  
   Build `H` and `h`.

5. `build_constraints(...)`  
   Build `Aqp` and `bqp`.

6. `solve_sm_mpc_qp(...)`  
   Build and solve the JuMP model.

---

## 11. Julia data layout

A practical convention is:

- `A_seq[j]` = \(A_{k+j-1}\), `j = 1:N`
- `B_seq[j]` = \(B_{k+j-1}\), `j = 1:N`
- `C_seq[j]` = \(C_{k+j}\), `j = 1:N` for predicted outputs
- `G_seq[j]` = \(G_{k+j}\), `j = 1:N`
- `Qs_seq[j]` = \(Q_{s,k+j}\), `j = 1:N`
- `Rv_seq[j]` = \(R_{v,k+j-1}\), `j = 1:N`
- `RΔ_seq[j]` = \(R_{\Delta,k+j-1}\), `j = 1:N`
- `x_ref[:,j]` = \(x^r_{k+j}\), `j = 1:N`
- `u_ref[:,j]` = \(u^r_{k+j-1}\), `j = 1:N`
- `y_ref[:,j]` = \(y^r_{k+j}\), `j = 1:N`

Current augmented state:
\[
\chi_k = \begin{bmatrix} e_k \\ v_{k-1} \end{bmatrix}
\]

represented as a Julia vector of length `nx + nu`.

---

## 12. Julia helper routines

```julia
using LinearAlgebra
using SparseArrays

"""
Build Aχ and Bχ sequences for the horizon.

Inputs:
- A_seq::Vector{Matrix{Float64}}   length N, each nx×nx
- B_seq::Vector{Matrix{Float64}}   length N, each nx×nu

Returns:
- Achi_seq::Vector{Matrix{Float64}} length N, each (nx+nu)×(nx+nu)
- Bchi_seq::Vector{Matrix{Float64}} length N, each (nx+nu)×nu
"""
function build_augmented_matrices(A_seq, B_seq)
    N = length(A_seq)
    nx = size(A_seq[1], 1)
    nu = size(B_seq[1], 2)

    Achi_seq = Vector{Matrix{Float64}}(undef, N)
    Bchi_seq = Vector{Matrix{Float64}}(undef, N)

    Iu = Matrix{Float64}(I, nu, nu)
    Zux = zeros(nu, nx)

    for j in 1:N
        A = A_seq[j]
        B = B_seq[j]
        Achi_seq[j] = [A B;
                       Zux Iu]
        Bchi_seq[j] = [B;
                       Iu]
    end
    return Achi_seq, Bchi_seq
end

"""
LTV transition product Φχ(ℓ,k) over local horizon indices.

Local convention:
- start_idx and end_idx are 1-based local indices into Achi_seq
- if end_idx < start_idx, return I

Example:
transition_product(Achi_seq, 1, 3) = Aχ[3] * Aχ[2] * Aχ[1]
"""
function transition_product(Achi_seq, start_idx::Int, end_idx::Int)
    nχ = size(Achi_seq[1], 1)
    T = Matrix{Float64}(I, nχ, nχ)
    if end_idx < start_idx
        return T
    end
    for t in start_idx:end_idx
        T = Achi_seq[t] * T
    end
    return T
end

"""
Build condensed prediction matrices Φ and Γ such that:

Xstack = Φ * χk + Γ * ΔU

where:
- Xstack = [χ_{k+1|k}; χ_{k+2|k}; ...; χ_{k+N|k}]
- ΔU     = [Δv_{k|k}; Δv_{k+1|k}; ...; Δv_{k+N-1|k}]
"""
function build_prediction_matrices(Achi_seq, Bchi_seq)
    N = length(Achi_seq)
    nχ = size(Achi_seq[1], 1)
    nu = size(Bchi_seq[1], 2)

    Φ = zeros(N * nχ, nχ)
    Γ = zeros(N * nχ, N * nu)

    for i in 1:N
        Φ[(i-1)*nχ+1:i*nχ, :] = transition_product(Achi_seq, 1, i)

        for j in 1:i
            block = transition_product(Achi_seq, j+1, i) * Bchi_seq[j]
            Γ[(i-1)*nχ+1:i*nχ, (j-1)*nu+1:j*nu] = block
        end
    end

    return Φ, Γ
end

"""
Block diagonal from a vector of dense matrices.
"""
function blockdiag_dense(mats::Vector{<:AbstractMatrix})
    rows = sum(size(M, 1) for M in mats)
    cols = sum(size(M, 2) for M in mats)
    out = zeros(rows, cols)
    r = 1
    c = 1
    for M in mats
        rr, cc = size(M)
        out[r:r+rr-1, c:c+cc-1] .= M
        r += rr
        c += cc
    end
    return out
end
```

---

## 13. Extraction matrices and cost matrices

```julia
"""
Build extraction matrices and terminal selectors.

Inputs:
- Φ, Γ              prediction matrices
- C_seq             length N, each ny×nx
- G_seq             length N, each ns×nx
- nx, nu, N

Returns named tuple with:
- Φe, Γe
- Φv, Γv
- Φs, Γs
- Φy, Γy
- ΦN, ΓN
"""
function build_extraction_matrices(Φ, Γ, C_seq, G_seq, nx, nu, N)
    nχ = nx + nu
    ny = size(C_seq[1], 1)

    # M_e extracts e from χ = [e; v_prev]
    Me = [Matrix{Float64}(I, nx, nx) zeros(nx, nu)]

    # M_v extracts the "current" v from each predicted χ block
    Mv = [zeros(nu, nx) Matrix{Float64}(I, nu, nu)]

    I_N = Matrix{Float64}(I, N, N)
    Φe = kron(I_N, Me) * Φ
    Γe = kron(I_N, Me) * Γ

    Φv = kron(I_N, Mv) * Φ
    Γv = kron(I_N, Mv) * Γ

    Gbar = blockdiag_dense(G_seq)
    Φs = Gbar * Φe
    Γs = Gbar * Γe

    Cbar = blockdiag_dense(C_seq)
    Φy = Cbar * Φe
    Γy = Cbar * Γe

    # terminal e_{k+N|k}
    EN = zeros(nχ, N * nχ)
    EN[:, (N-1)*nχ+1:N*nχ] .= Matrix{Float64}(I, nχ, nχ)
    ΦN = Me * EN * Φ
    ΓN = Me * EN * Γ

    return (
        Φe = Φe, Γe = Γe,
        Φv = Φv, Γv = Γv,
        Φs = Φs, Γs = Γs,
        Φy = Φy, Γy = Γy,
        ΦN = ΦN, ΓN = ΓN
    )
end

"""
Build H and h for the condensed QP:

min 0.5*ΔU'HΔU + h'ΔU
"""
function build_cost(χk, mats, Qs_seq, Rv_seq, RΔ_seq, P)
    Φs, Γs = mats.Φs, mats.Γs
    Φv, Γv = mats.Φv, mats.Γv
    ΦN, ΓN = mats.ΦN, mats.ΓN

    Qbar = blockdiag_dense(Qs_seq)
    Rvbar = blockdiag_dense(Rv_seq)
    RΔbar = blockdiag_dense(RΔ_seq)

    H = 2.0 * (
        Γs' * Qbar * Γs +
        Γv' * Rvbar * Γv +
        RΔbar +
        ΓN' * P * ΓN
    )

    h = 2.0 * (
        Γs' * Qbar * Φs +
        Γv' * Rvbar * Φv +
        ΓN' * P * ΦN
    ) * χk

    # Symmetrize for numerical robustness
    H = 0.5 * (H + H')
    return H, h
end
```

---

## 14. Constraint assembly

This routine builds:
\[
A_{qp}\Delta U \le b_{qp}
\]

```julia
"""
Stack vectors columnwise into one long vector.
Input matrix is dim×N, output is vec([col1; col2; ...]).
"""
stackcols(X::AbstractMatrix) = reshape(X, :)

"""
Build condensed inequality constraints Aqp*ΔU <= bqp.

Expected shapes:
- U_ref :: nu×N
- X_ref :: nx×N
- Y_ref :: ny×N
- ΔVmin, ΔVmax :: nu×N
- Umin, Umax   :: nu×N
- Xmin, Xmax   :: nx×N
- Ymin, Ymax   :: ny×N
"""
function build_constraints(
    χk, mats;
    U_ref, X_ref, Y_ref,
    ΔVmin, ΔVmax,
    Umin, Umax,
    Xmin, Xmax,
    Ymin, Ymax
)
    Φe, Γe = mats.Φe, mats.Γe
    Φv, Γv = mats.Φv, mats.Γv
    Φy, Γy = mats.Φy, mats.Γy

    nv = size(Γv, 2)

    A_list = Matrix{Float64}[]
    b_list = Vector{Float64}[]

    # 1) Increment bounds
    IΔ = Matrix{Float64}(I, nv, nv)
    push!(A_list, IΔ)
    push!(b_list, stackcols(ΔVmax))
    push!(A_list, -IΔ)
    push!(b_list, -stackcols(ΔVmin))

    # 2) Input magnitude bounds
    push!(A_list, Γv)
    push!(b_list, stackcols(Umax) - stackcols(U_ref) - Φv * χk)
    push!(A_list, -Γv)
    push!(b_list, -stackcols(Umin) + stackcols(U_ref) + Φv * χk)

    # 3) State bounds
    push!(A_list, Γe)
    push!(b_list, stackcols(Xmax) - stackcols(X_ref) - Φe * χk)
    push!(A_list, -Γe)
    push!(b_list, -stackcols(Xmin) + stackcols(X_ref) + Φe * χk)

    # 4) Output bounds
    push!(A_list, Γy)
    push!(b_list, stackcols(Ymax) - stackcols(Y_ref) - Φy * χk)
    push!(A_list, -Γy)
    push!(b_list, -stackcols(Ymin) + stackcols(Y_ref) + Φy * χk)

    Aqp = vcat(A_list...)
    bqp = vcat(b_list...)
    return Aqp, bqp
end
```

---

## 15. JuMP + OSQP implementation

This is a direct condensed-QP implementation.

```julia
using JuMP
using OSQP
using MathOptInterface
const MOI = MathOptInterface

"""
Solve one condensed sliding-mode MPC QP.

Returns:
- ΔU_star :: Vector{Float64}
- v0_star :: Vector{Float64}
- u0_star :: Vector{Float64}
"""
function solve_sm_mpc_qp_osqp(
    χk, uref0;
    H, h, Aqp, bqp,
    nu::Int
)
    nΔ = length(h)

    model = Model(OSQP.Optimizer)
    set_silent(model)

    @variable(model, ΔU[1:nΔ])

    # Objective: 0.5*ΔU'HΔU + h'ΔU
    @objective(model, Min, 0.5 * sum(H[i,j] * ΔU[i] * ΔU[j] for i in 1:nΔ, j in 1:nΔ)
                           + sum(h[i] * ΔU[i] for i in 1:nΔ))

    # Linear inequalities
    m = size(Aqp, 1)
    @constraint(model, [r = 1:m], sum(Aqp[r,c] * ΔU[c] for c in 1:nΔ) <= bqp[r])

    optimize!(model)

    term = termination_status(model)
    if !(term in (MOI.OPTIMAL, MOI.LOCALLY_SOLVED, MOI.ALMOST_OPTIMAL))
        error("QP solve failed with termination status = $term")
    end

    ΔU_star = value.(ΔU)
    Δv0 = ΔU_star[1:nu]

    # χk = [e_k; v_{k-1}]
    nx = length(χk) - nu
    v_prev = χk[nx+1:end]
    v0_star = v_prev + Δv0
    u0_star = uref0 + v0_star

    return ΔU_star, v0_star, u0_star
end
```

---

## 16. JuMP + Clarabel alternative

If preferred, the same condensed QP can be solved with Clarabel.

```julia
using JuMP
using Clarabel
using MathOptInterface
const MOI = MathOptInterface

function solve_sm_mpc_qp_clarabel(
    χk, uref0;
    H, h, Aqp, bqp,
    nu::Int
)
    nΔ = length(h)

    model = Model(Clarabel.Optimizer)
    set_silent(model)

    @variable(model, ΔU[1:nΔ])

    @objective(model, Min, 0.5 * sum(H[i,j] * ΔU[i] * ΔU[j] for i in 1:nΔ, j in 1:nΔ)
                           + sum(h[i] * ΔU[i] for i in 1:nΔ))

    m = size(Aqp, 1)
    @constraint(model, [r = 1:m], sum(Aqp[r,c] * ΔU[c] for c in 1:nΔ) <= bqp[r])

    optimize!(model)

    term = termination_status(model)
    if !(term in (MOI.OPTIMAL, MOI.LOCALLY_SOLVED, MOI.ALMOST_OPTIMAL))
        error("QP solve failed with termination status = $term")
    end

    ΔU_star = value.(ΔU)
    Δv0 = ΔU_star[1:nu]

    nx = length(χk) - nu
    v_prev = χk[nx+1:end]
    v0_star = v_prev + Δv0
    u0_star = uref0 + v0_star

    return ΔU_star, v0_star, u0_star
end
```

---

## 17. End-to-end controller step

This routine ties everything together for one MPC step.

```julia
"""
One control step of sliding-mode MPC without integrator.

Inputs:
- ek        :: Vector{Float64}     current tracking error x - x_ref
- v_prev    :: Vector{Float64}     previous input error u_{k-1} - u^r_{k-1}
- A_seq, B_seq, C_seq, G_seq
- Qs_seq, Rv_seq, RΔ_seq, P
- X_ref, U_ref, Y_ref
- ΔVmin, ΔVmax, Umin, Umax, Xmin, Xmax, Ymin, Ymax

Returns:
- u0_star, v0_star, ΔU_star, H, h, Aqp, bqp
"""
function sm_mpc_step(
    ek, v_prev,
    A_seq, B_seq, C_seq, G_seq,
    Qs_seq, Rv_seq, RΔ_seq, P,
    X_ref, U_ref, Y_ref,
    ΔVmin, ΔVmax,
    Umin, Umax,
    Xmin, Xmax,
    Ymin, Ymax;
    solver::Symbol = :osqp
)
    nx = length(ek)
    nu = length(v_prev)
    N = length(A_seq)

    χk = vcat(ek, v_prev)

    Achi_seq, Bchi_seq = build_augmented_matrices(A_seq, B_seq)
    Φ, Γ = build_prediction_matrices(Achi_seq, Bchi_seq)
    mats = build_extraction_matrices(Φ, Γ, C_seq, G_seq, nx, nu, N)

    H, h = build_cost(χk, mats, Qs_seq, Rv_seq, RΔ_seq, P)

    Aqp, bqp = build_constraints(
        χk, mats;
        U_ref = U_ref,
        X_ref = X_ref,
        Y_ref = Y_ref,
        ΔVmin = ΔVmin,
        ΔVmax = ΔVmax,
        Umin = Umin,
        Umax = Umax,
        Xmin = Xmin,
        Xmax = Xmax,
        Ymin = Ymin,
        Ymax = Ymax
    )

    uref0 = U_ref[:, 1]

    if solver == :osqp
        ΔU_star, v0_star, u0_star = solve_sm_mpc_qp_osqp(
            χk, uref0; H = H, h = h, Aqp = Aqp, bqp = bqp, nu = nu
        )
    elseif solver == :clarabel
        ΔU_star, v0_star, u0_star = solve_sm_mpc_qp_clarabel(
            χk, uref0; H = H, h = h, Aqp = Aqp, bqp = bqp, nu = nu
        )
    else
        error("Unsupported solver: $solver")
    end

    return u0_star, v0_star, ΔU_star, H, h, Aqp, bqp
end
```

---

## 18. Minimal usage sketch

```julia
nx = 2
nu = 1
ny = 1
ns = 1
N  = 10

# Example time-varying horizon data
A_seq = [ [1.0 0.1; 0.0 0.98 + 0.001*j] for j in 1:N ]
B_seq = [ [0.0; 0.1] for _ in 1:N ]
C_seq = [ [1.0 0.0] for _ in 1:N ]
G_seq = [ [2.0 1.0] for _ in 1:N ]

Qs_seq = [ [10.0] for _ in 1:N ]
Rv_seq = [ [0.1] for _ in 1:N ]
RΔ_seq = [ [1.0] for _ in 1:N ]
P = [20.0 0.0; 0.0 5.0]

ek = [0.2, -0.1]
v_prev = [0.0]

X_ref = zeros(nx, N)
U_ref = zeros(nu, N)
Y_ref = zeros(ny, N)

ΔVmin = fill(-0.2, nu, N)
ΔVmax = fill( 0.2, nu, N)
Umin  = fill(-1.0, nu, N)
Umax  = fill( 1.0, nu, N)
Xmin  = fill(-10.0, nx, N)
Xmax  = fill( 10.0, nx, N)
Ymin  = fill(-10.0, ny, N)
Ymax  = fill( 10.0, ny, N)

u0_star, v0_star, ΔU_star, H, h, Aqp, bqp = sm_mpc_step(
    ek, v_prev,
    A_seq, B_seq, C_seq, G_seq,
    Qs_seq, Rv_seq, RΔ_seq, P,
    X_ref, U_ref, Y_ref,
    ΔVmin, ΔVmax,
    Umin, Umax,
    Xmin, Xmax,
    Ymin, Ymax;
    solver = :osqp
)
```

---

## 19. Notes for implementation

1. `H` must be positive semidefinite for a convex QP.  
   In practice:
   - choose \(Q_s \succeq 0\),
   - choose \(R_v \succeq 0\),
   - choose \(R_\Delta \succ 0\),
   - symmetrize `H = 0.5*(H + H')`.

2. For large horizons, use sparse matrices.  
   The condensed form is simple, but a sparse stagewise QP may scale better.

3. If the reference is not dynamically feasible, the error dynamics become affine:
   \[
   e_{k+1} = A_k e_k + B_k v_k + d_k
   \]
   This only changes affine terms in the prediction and constraints, not the basic QP structure.

4. A common design is constant sliding matrix:
   \[
   G_k = G
   \]
   with a choice like:
   \[
   G = \begin{bmatrix}\lambda & 1\end{bmatrix}
   \]
   for second-order tracking.

5. This note uses a **tracking-error formulation**.  
   The final applied control is:
   \[
   u_k = u^r_k + v_k
   \]

---

## 20. Package notes

A JuMP model can express quadratic objectives and linear constraints directly. JuMP’s documentation covers models, variables, constraints, quadratic expressions, and objectives. OSQP’s JuMP interface supports convex QPs in the standard form \( \tfrac12 x^\top P x + q^\top x \) with linear constraints, and Clarabel’s JuMP interface also supports quadratic objectives directly. See the current JuMP and solver package documentation when wiring the environment and solver options together.

Recommended imports:

```julia
using LinearAlgebra
using SparseArrays
using JuMP
using MathOptInterface
const MOI = MathOptInterface

# choose one:
using OSQP
# or:
using Clarabel
```

Typical package installation:

```julia
using Pkg
Pkg.add("JuMP")
Pkg.add("OSQP")
Pkg.add("Clarabel")
```

---

## 21. Codex-facing implementation checklist

A code generator should implement the following, in order:

1. Accept horizon sequences `A_seq`, `B_seq`, `C_seq`, `G_seq`.
2. Build `Achi_seq`, `Bchi_seq`.
3. Build `Φ`, `Γ`.
4. Build extraction matrices.
5. Build `H`, `h`.
6. Build `Aqp`, `bqp`.
7. Create JuMP variable `ΔU`.
8. Add quadratic objective `0.5*ΔU'HΔU + h'ΔU`.
9. Add linear inequalities `Aqp*ΔU <= bqp`.
10. Optimize.
11. Recover:
    - `Δv0 = ΔU[1:nu]`
    - `v0 = v_prev + Δv0`
    - `u0 = u_ref[:,1] + v0`
12. Apply `u0` and shift the horizon.

This is the condensed sliding-mode MPC implementation without an integrator.
