"""
Earth EDL Reentry Trajectory Linearization
=========================================
Dynamics use the project Earth EDL constants, converted to English units for
compatibility with the generated Jacobian API.

States:  x = [h, φ, θ, v, γ, ψ]  (altitude ft, longitude rad, latitude rad,
                                    velocity ft/s, FPA rad, azimuth rad)
Controls: u = [α, β]              (angle-of-attack rad, bank angle rad)

This script:
  1. Derives the continuous-time Jacobians A_c, B_c symbolically via Symbolics.jl
  2. Computes the same Jacobians numerically via ForwardDiff.jl
  3. Discretizes (A_c, B_c) → (A_d, B_d) via the matrix-exponential ZOH method
     using the van Loan / Higham expm approach (LinearAlgebra.exp)
  4. Prints a comparison between symbolic and ForwardDiff results at the
     chosen operating point.
  5. Saves the symbolic Jacobians in two forms:
       • reentry_jacobians.jl  — generated Julia source (eval_Ac!, eval_Bc!)
                                  with no Symbolics dependency at evaluation time
       • reentry_jacobians.bin — binary-serialized Symbolics.Num expression
                                  matrices for reload, re-substitution, or
                                  further symbolic manipulation

Units: English (ft, slug, rad, sec)
"""

using Symbolics          # symbolic Jacobians
using LinearAlgebra      # exp for ZOH discretization
using ForwardDiff        # numerical Jacobian check
using Serialization      # save/load raw Symbolics expression trees
using Printf
include("../model/earth_atmosphere_polyfit.jl")
include("../model/vehicle.jl")

# ─────────────────────────────────────────────────────────────────────────────
# 1.  CONSTANTS & AERODYNAMIC PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────

const ft_per_meter_val = 3.28084
const slug_per_kg_val = 0.06852176585679176
const kg_per_m3_to_slug_per_ft3_val = slug_per_kg_val / ft_per_meter_val^3

const μ_val  = 3.986004418e14 * ft_per_meter_val^3   # ft³/s²  Earth gravitational parameter
const Re_val = 6378137.0 * ft_per_meter_val          # ft      Earth radius
const S_val  = VEHICLE.reference_area * ft_per_meter_val^2  # ft² reference area
const g0_val = 32.174                                # ft/s²   standard gravity for unit conversions

const m_val  = VEHICLE.mass * slug_per_kg_val        # slug    vehicle mass
const linearization_density_polyfit_coefficients = earth_atmosphere_polyfit_coefficients(125000.0)

# Lift/drag polynomial coefficients
const a0_val =  -0.20704
const a1_val =   0.029244
const b0_val =   0.07854
const b1_val =  -0.61592e-2
const b2_val =   0.621408e-3

# Heating coefficients
const c0_val =   1.0672181
const c1_val =  -0.19213774e-1
const c2_val =   0.21286289e-3
const c3_val =  -0.10117249e-5

# ─────────────────────────────────────────────────────────────────────────────
# 2.  DYNAMICS  f(x, u) → ẋ   (pure-Julia, works for both Float64 and Dual)
# ─────────────────────────────────────────────────────────────────────────────

"""
    dynamics(xu)

Evaluate the 6-dimensional RHS of the Earth EDL reentry EOM.
`xu` is a length-8 vector: [h, φ, θ, v, γ, ψ, α, β]
Returns ẋ = [ḣ, φ̇, θ̇, v̇, γ̇, ψ̇].

The heating inequality q ≤ q_U is handled as a path constraint elsewhere;
it does not appear in the smooth DAE used for linearization.
"""
function dynamics(xu::AbstractVector{T}) where T
    h, φ, θ, v, γ, ψ, α, β = xu[1], xu[2], xu[3], xu[4], xu[5], xu[6],
                               xu[7], xu[8]

    r   = Re_val + h
    g   = μ_val / r^2
    h_km = h / ft_per_meter_val / 1000.0
    density_exponent = zero(h_km)
    for c in linearization_density_polyfit_coefficients
        density_exponent = density_exponent * h_km + c
    end
    ρ = exp(density_exponent) * kg_per_m3_to_slug_per_ft3_val
    # α in degrees for polynomial evaluation
    α_deg = 180 * α / π

    cL  = a0_val + a1_val * α_deg
    cD  = b0_val + b1_val * α_deg + b2_val * α_deg^2

    q_dyn = T(0.5) * ρ * S_val * v^2      # dynamic-pressure × S  [lbf ≡ slug·ft/s²]
    L   = cL * q_dyn
    D   = cD * q_dyn

    # EOM
    hdot = v * sin(γ)
    φdot = (v / r) * cos(γ) * sin(ψ) / cos(θ)
    θdot = (v / r) * cos(γ) * cos(ψ)
    vdot = -D / m_val - g * sin(γ)
    γdot = (L / (m_val * v)) * cos(β) + cos(γ) * (v / r - g / v)
    ψdot = (L / (m_val * v * cos(γ))) * sin(β) +
           (v / (r * cos(θ))) * cos(γ) * sin(ψ) * sin(θ)

    return [hdot, φdot, θdot, vdot, γdot, ψdot]
end

# Split for clarity
f_state(x, u) = dynamics([x; u])     # x ∈ R^6, u ∈ R^2

# ─────────────────────────────────────────────────────────────────────────────
# 3.  OPERATING POINT
# ─────────────────────────────────────────────────────────────────────────────

# Trim point: Earth EDL initial values from Main.jl (can be changed)
h0   = 125000.0 * ft_per_meter_val    # ft
φ0   = deg2rad(126.7)                 # rad
θ0   = deg2rad(-3.93)                 # rad
v0   = 5845.39 * ft_per_meter_val     # ft/s
γ0   = deg2rad(-15.49)                # rad
ψ0   = deg2rad(90.0)                  # rad
α0   = deg2rad(0.0)                   # rad
β0   = deg2rad(0.0)                   # rad  (wings level)

x0 = [h0, φ0, θ0, v0, γ0, ψ0]
u0 = [α0, β0]

println("=" ^ 65)
println("  Earth EDL Reentry — Linearization")
println("=" ^ 65)
@printf "\nOperating point:\n"
@printf "  h  = %.0f ft        v  = %.0f ft/s\n"  h0 v0
@printf "  φ  = %.4f rad    γ  = %.4f rad (%.2f deg)\n" φ0 γ0 rad2deg(γ0)
@printf "  θ  = %.4f rad    ψ  = %.4f rad (%.1f deg)\n" θ0 ψ0 rad2deg(ψ0)
@printf "  α  = %.4f rad (%.2f deg)\n" α0 rad2deg(α0)
@printf "  β  = %.4f rad (%.2f deg)\n\n" β0 rad2deg(β0)

# ─────────────────────────────────────────────────────────────────────────────
# 4.  SYMBOLIC JACOBIANS  (Symbolics.jl)
# ─────────────────────────────────────────────────────────────────────────────

println("─" ^ 65)
println("  4.  Symbolic Jacobians via Symbolics.jl")
println("─" ^ 65)

@variables h_s φ_s θ_s v_s γ_s ψ_s α_s β_s

xu_sym = [h_s, φ_s, θ_s, v_s, γ_s, ψ_s, α_s, β_s]
f_sym  = dynamics(xu_sym)               # vector of 6 symbolic expressions

# Full Jacobian ∂f/∂[x;u]  (6×8)
Jfull_sym = Symbolics.jacobian(f_sym, xu_sym)

A_sym = Jfull_sym[:, 1:6]   # ∂f/∂x
B_sym = Jfull_sym[:, 7:8]   # ∂f/∂u

println("  Symbolic A matrix (expression forms generated — evaluating numerically)...")
println("  Symbolic B matrix (expression forms generated — evaluating numerically)...")

# Build fast callable functions from symbolic expressions
subs_dict = Dict(
    h_s => h0, φ_s => φ0, θ_s => θ0,
    v_s => v0, γ_s => γ0, ψ_s => ψ0,
    α_s => α0, β_s => β0
)

evaluate_matrix(M) = Float64.(Symbolics.value.(Symbolics.substitute(M, subs_dict; fold = Val(true))))

A_sym_num = evaluate_matrix(A_sym)
B_sym_num = evaluate_matrix(B_sym)

println("\n  A_c  (symbolic, evaluated at operating point):")
display(A_sym_num)
println()
println("  B_c  (symbolic, evaluated at operating point):")
display(B_sym_num)

# ─────────────────────────────────────────────────────────────────────────────
# 5.  NUMERICAL JACOBIANS  (ForwardDiff.jl — verification)
# ─────────────────────────────────────────────────────────────────────────────

println()
println("─" ^ 65)
println("  5.  Numerical Jacobians via ForwardDiff.jl")
println("─" ^ 65)

# Wrap so ForwardDiff can differentiate
f_x(x) = f_state(x, u0)
f_u(u) = f_state(x0, u)

A_fd = ForwardDiff.jacobian(f_x, x0)
B_fd = ForwardDiff.jacobian(f_u, u0)

println("\n  A_c  (ForwardDiff):")
display(A_fd)
println()
println("  B_c  (ForwardDiff):")
display(B_fd)

# ─────────────────────────────────────────────────────────────────────────────
# 6.  COMPARISON
# ─────────────────────────────────────────────────────────────────────────────

println()
println("─" ^ 65)
println("  6.  Residual  |A_sym − A_fd|  and  |B_sym − B_fd|")
println("─" ^ 65)

A_err = norm(A_sym_num - A_fd)
B_err = norm(B_sym_num - B_fd)
@printf "\n  ||A_sym − A_fd||_F  = %.3e\n" A_err
@printf "  ||B_sym − B_fd||_F  = %.3e\n" B_err

if A_err < 1e-6 && B_err < 1e-6
    println("\n  ✓  Symbolic and ForwardDiff Jacobians agree to machine precision.")
else
    println("\n  ✗  Residual larger than expected — check operating point or sign conventions.")
end

# ─────────────────────────────────────────────────────────────────────────────
# 7.  DISCRETIZATION  via ZOH (matrix-exponential method)
# ─────────────────────────────────────────────────────────────────────────────

println()
println("─" ^ 65)
println("  7.  ZOH Discretization  (van Loan / matrix-exponential method)")
println("─" ^ 65)

"""
    zoh_discretize(Ac, Bc, dt)

Zero-order-hold discretization of (Ac, Bc) with sample time dt.

Uses the standard augmented-matrix exponentiation:

    M = exp([Ac  Bc; 0  0] * dt)

which gives  Ad = M[1:n, 1:n],  Bd = M[1:n, n+1:n+m].
"""
function zoh_discretize(Ac::Matrix, Bc::Matrix, dt::Real)
    n, m = size(Bc)
    Z = zeros(m, n + m)
    M = exp([Ac  Bc; Z] * dt)
    Ad = M[1:n, 1:n]
    Bd = M[1:n, n+1:n+m]
    return Ad, Bd
end

# Collect discrete matrices for all sample times so we can save them later
discrete_results = Dict{Float64, NamedTuple{(:Ad, :Bd), Tuple{Matrix{Float64}, Matrix{Float64}}}}()

for dt in [0.1, 1.0, 5.0]
    Ad, Bd = zoh_discretize(A_sym_num, B_sym_num, dt)
    discrete_results[dt] = (Ad = Ad, Bd = Bd)

    @printf "\n  dt = %.1f s\n" dt
    println("  Ad:")
    display(Ad)
    println("  Bd:")
    display(Bd)
    # Quick sanity: Ad should approach I + Ac*dt for small dt
    if dt == 0.1
        Ad_approx = I + A_sym_num * dt
        err_euler = norm(Ad - Ad_approx)
        @printf "  ||Ad − (I + Ac·dt)||_F  (Euler approx error, expected O(dt²))  = %.3e\n" err_euler
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# 7b.  SAVE SYMBOLIC JACOBIANS
#
# Two complementary artefacts are written:
#
#   reentry_jacobians.jl
#     A Julia source file generated by Symbolics.build_function that defines
#     two fast in-place functions:
#
#       eval_Ac!(out, h, φ, θ, v, γ, ψ, α, β)   # fills 6×6 matrix
#       eval_Bc!(out, h, φ, θ, v, γ, ψ, α, β)   # fills 6×2 matrix
#
#     These are plain Julia — no Symbolics dependency at evaluation time.
#     Load with:  include("reentry_jacobians.jl")
#     Then call:  Ac = zeros(6,6); eval_Ac!(Ac, h, φ, θ, v, γ, ψ, α, β)
#
#   reentry_jacobians.bin
#     Binary serialization of the raw Symbolics.Num expression matrices
#     (A_sym, B_sym) and the symbolic variable tuple, via Julia's built-in
#     Serialization stdlib.  Lets you reload the live symbolic objects to
#     re-run substitution, further differentiation, or code generation.
#     Load with:
#       using Serialization, Symbolics
#       A_sym, B_sym, sym_vars = deserialize("reentry_jacobians.bin")
#       # sym_vars is a NamedTuple: (h=h_s, φ=φ_s, θ=θ_s, v=v_s, γ=γ_s,
#       #                            ψ=ψ_s, α=α_s, β=β_s)
#       subs = Dict(sym_vars.h => my_h, sym_vars.v => my_v, ...)
#       Ac = Float64.(Symbolics.value.(Symbolics.substitute(A_sym, subs; fold=Val(true))))
# ─────────────────────────────────────────────────────────────────────────────

using Serialization

println()
println("─" ^ 65)
println("  7b.  Saving symbolic Jacobians")
println("─" ^ 65)

# ── Generated Julia source  (build_function) ─────────────────────────────────
# build_function produces a pair: (out-of-place, in-place).
# We keep the in-place form so the caller pre-allocates the output matrix
# and pays no allocation cost at evaluation time.
scalar_args = (h_s, φ_s, θ_s, v_s, γ_s, ψ_s, α_s, β_s)

Ac_fn_expr = Symbolics.build_function(A_sym, scalar_args...;
                 fname        = :eval_Ac!,
                 expression   = Val{true})[2]   # [2] = in-place form

Bc_fn_expr = Symbolics.build_function(B_sym, scalar_args...;
                 fname        = :eval_Bc!,
                 expression   = Val{true})[2]

src_path = joinpath(@__DIR__, "reentry_jacobians.jl")
open(src_path, "w") do io
    println(io, """
# Auto-generated by reentry_linearization.jl — do not edit by hand.
# Defines fast in-place Jacobian evaluators with NO Symbolics dependency.
#
# Usage:
#   include("reentry_jacobians.jl")
#   Ac = zeros(6, 6)
#   Bc = zeros(6, 2)
#   eval_Ac!(Ac, h, φ, θ, v, γ, ψ, α, β)
#   eval_Bc!(Bc, h, φ, θ, v, γ, ψ, α, β)
#
# All arguments are scalars in English units (ft, rad, ft/s).

if !isdefined(@__MODULE__, :NaNMath)
    const NaNMath = Base
end
""")
    print(io, "eval_Ac! = ")
    println(io, Ac_fn_expr)
    println(io)
    print(io, "eval_Bc! = ")
    println(io, Bc_fn_expr)
end
println("\n  ✓  Generated source  →  $src_path")
println("     Load:  include(\"$src_path\")")
println("     Use:   Ac = zeros(6,6); eval_Ac!(Ac, h, φ, θ, v, γ, ψ, α, β)")

# ── Binary serialization of live Symbolics objects ───────────────────────────
bin_path = joinpath(@__DIR__, "reentry_jacobians.bin")
sym_vars = (h=h_s, φ=φ_s, θ=θ_s, v=v_s, γ=γ_s, ψ=ψ_s, α=α_s, β=β_s)
serialize(bin_path, (A_sym, B_sym, sym_vars))
println("\n  ✓  Symbolic binary  →  $bin_path")
println("     Load:  using Serialization, Symbolics")
println("            A_sym, B_sym, vars = deserialize(\"$bin_path\")")
println("            subs = Dict(vars.h => h_val, vars.v => v_val, ...)")
println("            Ac = Float64.(Symbolics.value.(Symbolics.substitute(A_sym, subs; fold=Val(true))))")

# ── Sanity-check the generated source immediately ────────────────────────────
include(src_path)
Ac_gen = zeros(6, 6)
Bc_gen = zeros(6, 2)
eval_Ac!(Ac_gen, h0, φ0, θ0, v0, γ0, ψ0, α0, β0)
eval_Bc!(Bc_gen, h0, φ0, θ0, v0, γ0, ψ0, α0, β0)
err_gen_A = norm(Ac_gen - A_sym_num)
err_gen_B = norm(Bc_gen - B_sym_num)
@printf "\n  Sanity check — generated functions vs symbolic substitution:\n"
@printf "    ||eval_Ac! − A_sym_num||_F = %.3e\n" err_gen_A
@printf "    ||eval_Bc! − B_sym_num||_F = %.3e\n" err_gen_B
if err_gen_A < 1e-10 && err_gen_B < 1e-10
    println("  ✓  Generated source matches symbolic evaluation exactly.")
else
    println("  ✗  Mismatch — check build_function output.")
end

# ─────────────────────────────────────────────────────────────────────────────
# 8.  EIGENVALUE ANALYSIS of A_c
# ─────────────────────────────────────────────────────────────────────────────

println()
println("─" ^ 65)
println("  8.  Eigenvalues of A_c  (open-loop stability)")
println("─" ^ 65)

λ = eigvals(A_sym_num)
println()
for (i, ev) in enumerate(λ)
    @printf "  λ_%d = %+.6e  %+.6e im   |Re(λ)| = %.3e\n" i real(ev) imag(ev) abs(real(ev))
end

unstable = filter(ev -> real(ev) > 1e-10, λ)
if isempty(unstable)
    println("\n  ✓  All eigenvalues have Re(λ) ≤ 0  (marginally stable or stable).")
else
    println("\n  ✗  Unstable modes present (Re(λ) > 0).")
end

println("\n" * "=" ^ 65)
println("  Done.")
println("=" ^ 65)
