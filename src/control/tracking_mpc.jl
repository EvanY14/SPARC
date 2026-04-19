using LinearAlgebra
using JuMP
using Ipopt
using CSV
using DataFrames
using Interpolations

const MOI = JuMP.MOI
const _REENTRY_JACOBIANS_PATH = joinpath(@__DIR__, "..", "reference", "reentry_jacobians.jl")
include(_REENTRY_JACOBIANS_PATH)
const _FT_PER_METER = 3.28084
const _OPTIMAL_TRAJECTORY_PATH = joinpath(@__DIR__, "..", "..", "optimal_trajectory.csv")
const _OPTIMAL_CONTROL_CACHE = Ref{Any}(nothing)

# These are intentionally permissive: tracking feasibility should come from the
# reference cost and actuator limits, not brittle hard state cuts on a regenerated
# nominal trajectory.
const _TRACKING_STATE_WIDE_MIN = [-1.0e7, -100.0 * π, -100.0 * π, -1.0e5, -100.0 * π, -100.0 * π]
const _TRACKING_STATE_WIDE_MAX = [1.0e7, 100.0 * π, 100.0 * π, 1.0e5, 100.0 * π, 100.0 * π]

# Rate limits are bounds on the MPC correction move, not on the reference
# controls loaded from the optimal trajectory. Keep them loose enough for the QP
# to recover from reference/model mismatch while still discouraging chatter.
const _TRACKING_INPUT_RATE_LIMIT_RAD_PER_SEC = deg2rad.([30.0, 20.0])
const _TRACKING_POLYFIT_COEFFICIENTS = [
	-8.278592174668491e-43,
	1.2598495030132498e-38,
	-8.634065871212132e-35,
	3.5185552646901455e-31,
	-9.480197229347404e-28,
	1.7753104600795092e-24,
	-2.3622107295909874e-21,
	2.2393603867716714e-18,
	-1.487031340144351e-15,
	6.592111911218399e-13,
	-1.714014789283248e-10,
	1.3556252797088945e-8,
	5.196239221937857e-6,
	-0.0012393556758398866,
	-0.0500835105059738,
	-4.213431227716942,
]

function _blockdiag_dense(mats::Vector{<:AbstractMatrix})
	rows = sum(size(M, 1) for M in mats)
	cols = sum(size(M, 2) for M in mats)
	out = zeros(rows, cols)
	r = 1
	c = 1
	for M in mats
		rr, cc = size(M)
		out[r:r + rr - 1, c:c + cc - 1] .= M
		r += rr
		c += cc
	end
	return out
end

_stackcols(X::AbstractMatrix) = reshape(X, :)

function _csv_column(df::DataFrame, names::Tuple{Vararg{Symbol}})
	for name in names
		if name in propertynames(df)
			return Float64.(df[!, name])
		end
	end
	return nothing
end

function _load_optimal_control_cache(path::AbstractString = _OPTIMAL_TRAJECTORY_PATH)
	if !isfile(path)
		return nothing
	end

	mtime = stat(path).mtime
	cache = _OPTIMAL_CONTROL_CACHE[]
	if cache !== nothing && cache.path == path && cache.mtime == mtime
		return cache
	end

	df = CSV.read(path, DataFrame)
	t = _csv_column(df, (:Time_s, :time_s, :time, :t))
	α_deg = _csv_column(df, (:AngleOfAttack_deg, :Alpha_deg, :AOA_deg))
	β_deg = _csv_column(df, (:BankAngle_deg, :Beta_deg, :Bank_deg))
	α_rad = _csv_column(df, (:AngleOfAttack_rad, :Alpha_rad, :AOA_rad))
	β_rad = _csv_column(df, (:BankAngle_rad, :Beta_rad, :Bank_rad))

	if t === nothing || (α_deg === nothing && α_rad === nothing) || (β_deg === nothing && β_rad === nothing)
		return nothing
	end

	α = α_rad === nothing ? deg2rad.(α_deg) : α_rad
	β = β_rad === nothing ? deg2rad.(β_deg) : β_rad
	order = sortperm(t)
	t_sorted = t[order]
	α_sorted = α[order]
	β_sorted = β[order]

	cache = (
		path = String(path),
		mtime = mtime,
		t_min = first(t_sorted),
		t_max = last(t_sorted),
		α = linear_interpolation(t_sorted, α_sorted, extrapolation_bc = Line()),
		β = linear_interpolation(t_sorted, β_sorted, extrapolation_bc = Line()),
	)
	_OPTIMAL_CONTROL_CACHE[] = cache
	return cache
end

function _reference_control_at(τ::Real, fallback::AbstractVector{<:Real})
	cache = _load_optimal_control_cache()
	if cache === nothing
		return Float64.(fallback)
	end
	return [Float64(cache.α(τ)), Float64(cache.β(τ))]
end

function _transition_product(Achi_seq::Vector{Matrix{Float64}}, start_idx::Int, end_idx::Int)
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

function _build_augmented_matrices(
	A_seq::Vector{Matrix{Float64}},
	B_seq::Vector{Matrix{Float64}},
	d_seq::Vector{Vector{Float64}},
)
	N = length(A_seq)
	nx = size(A_seq[1], 1)
	nu = size(B_seq[1], 2)

	Achi_seq = Vector{Matrix{Float64}}(undef, N)
	Bchi_seq = Vector{Matrix{Float64}}(undef, N)
	cchi_seq = Vector{Vector{Float64}}(undef, N)

	Iu = Matrix{Float64}(I, nu, nu)
	Zux = zeros(nu, nx)
	for j in 1:N
		A = A_seq[j]
		B = B_seq[j]
		Achi_seq[j] = [A B; Zux Iu]
		Bchi_seq[j] = [B; Iu]
		cchi_seq[j] = vcat(d_seq[j], zeros(nu))
	end
	return Achi_seq, Bchi_seq, cchi_seq
end

function _zoh_discretize(Ac::Matrix{Float64}, Bc::Matrix{Float64}, dt::Real)
	n, m = size(Bc)
	M = exp([Ac Bc; zeros(m, n + m)] * dt)
	return M[1:n, 1:n], M[1:n, n + 1:n + m]
end

function _tracking_polyfit_density(h::Real)
	h_km = Float64(h) * 1.0e-3
	exponent = 0.0
	for i in eachindex(_TRACKING_POLYFIT_COEFFICIENTS)
		exponent += _TRACKING_POLYFIT_COEFFICIENTS[i] * h_km^(length(_TRACKING_POLYFIT_COEFFICIENTS) - i)
	end
	return exp(exponent)
end

function _nominal_reentry_dynamics_si(
	x::AbstractVector{<:Real},
	u::AbstractVector{<:Real},
)
	h = Float64(x[1])
	θ = Float64(x[3])
	v = Float64(x[4])
	γ = Float64(x[5])
	ψ = Float64(x[6])
	α = Float64(u[1])
	β = Float64(u[2])

	m = 3257.0
	S = 15.904
	μ = 4.2828372e13
	R = 3396200.0
	a0 = -0.20704
	a1 = 0.029244
	b0 = 0.07854
	b1 = -0.61592e-2
	b2 = 0.621408e-3

	ρ = _tracking_polyfit_density(h)
	α_deg = rad2deg(α)
	cL = a0 + a1 * α_deg
	cD = b0 + b1 * α_deg + b2 * α_deg^2
	r = R + h
	g = μ / r^2
	D = 0.5 * cD * S * ρ * v^2
	L = 0.5 * cL * S * ρ * v^2

	return [
		v * sin(γ),
		(v / r) * cos(γ) * sin(ψ) / cos(θ),
		(v / r) * cos(γ) * cos(ψ),
		-(D / m) - g * sin(γ),
		(L / (m * v)) * cos(β) + cos(γ) * ((v / r) - (g / v)),
		(L / (m * v * cos(γ))) * sin(β) + (v / (r * cos(θ))) * cos(γ) * sin(ψ) * sin(θ),
	]
end

function _nominal_reentry_step_si(
	x::AbstractVector{<:Real},
	u::AbstractVector{<:Real},
	dt::Real,
)
	k1 = _nominal_reentry_dynamics_si(x, u)
	k2 = _nominal_reentry_dynamics_si(x .+ 0.5 * dt .* k1, u)
	k3 = _nominal_reentry_dynamics_si(x .+ 0.5 * dt .* k2, u)
	k4 = _nominal_reentry_dynamics_si(x .+ dt .* k3, u)
	return Float64.(x) .+ (dt / 6.0) .* (k1 .+ 2.0 .* k2 .+ 2.0 .* k3 .+ k4)
end

function _generated_continuous_linearization_si(
	x_ref::AbstractVector{<:Real},
	u_ref::AbstractVector{<:Real},
)
	h_ft = Float64(x_ref[1]) * _FT_PER_METER
	φ = Float64(x_ref[2])
	θ = Float64(x_ref[3])
	v_fts = Float64(x_ref[4]) * _FT_PER_METER
	γ = Float64(x_ref[5])
	ψ = Float64(x_ref[6])
	α = Float64(u_ref[1])
	β = Float64(u_ref[2])

	Ac_english = zeros(6, 6)
	Bc_english = zeros(6, 2)
	eval_Ac!(Ac_english, h_ft, φ, θ, v_fts, γ, ψ, α, β)
	eval_Bc!(Bc_english, h_ft, φ, θ, v_fts, γ, ψ, α, β)

	scale_english_from_si = Diagonal([_FT_PER_METER, 1.0, 1.0, _FT_PER_METER, 1.0, 1.0])
	scale_si_from_english = inv(scale_english_from_si)
	Ac_si = Matrix(scale_si_from_english * Ac_english * scale_english_from_si)
	Bc_si = Matrix(scale_si_from_english * Bc_english)

	return Ac_si, Bc_si
end

function _generated_discrete_linearization_si(
	x_ref::AbstractVector{<:Real},
	u_ref::AbstractVector{<:Real},
	dt::Real,
)
	Ac, Bc = _generated_continuous_linearization_si(x_ref, u_ref)
	return _zoh_discretize(Ac, Bc, dt)
end

function _build_prediction_matrices(
	Achi_seq::Vector{Matrix{Float64}},
	Bchi_seq::Vector{Matrix{Float64}},
	cchi_seq::Vector{Vector{Float64}},
)
	N = length(Achi_seq)
	nχ = size(Achi_seq[1], 1)
	nu = size(Bchi_seq[1], 2)

	Φ = zeros(N * nχ, nχ)
	Γ = zeros(N * nχ, N * nu)
	η = zeros(N * nχ)

	for i in 1:N
		Φ[(i - 1) * nχ + 1:i * nχ, :] = _transition_product(Achi_seq, 1, i)
		for j in 1:i
			block = _transition_product(Achi_seq, j + 1, i) * Bchi_seq[j]
			Γ[(i - 1) * nχ + 1:i * nχ, (j - 1) * nu + 1:j * nu] = block
			η[(i - 1) * nχ + 1:i * nχ] .+= _transition_product(Achi_seq, j + 1, i) * cchi_seq[j]
		end
	end
	return Φ, Γ, η
end

function _build_extraction_matrices(
	Φ::Matrix{Float64},
	Γ::Matrix{Float64},
	η::Vector{Float64},
	C_seq::Vector{Matrix{Float64}},
	G_seq::Vector{Matrix{Float64}},
	nx::Int,
	nu::Int,
	N::Int,
)
	nχ = nx + nu
	Me = [Matrix{Float64}(I, nx, nx) zeros(nx, nu)]
	Mv = [zeros(nu, nx) Matrix{Float64}(I, nu, nu)]

	I_N = Matrix{Float64}(I, N, N)
	Φe = kron(I_N, Me) * Φ
	Γe = kron(I_N, Me) * Γ
	ηe = kron(I_N, Me) * η

	Φv = kron(I_N, Mv) * Φ
	Γv = kron(I_N, Mv) * Γ
	ηv = kron(I_N, Mv) * η

	Gbar = _blockdiag_dense(G_seq)
	Φs = Gbar * Φe
	Γs = Gbar * Γe
	ηs = Gbar * ηe

	Cbar = _blockdiag_dense(C_seq)
	Φy = Cbar * Φe
	Γy = Cbar * Γe
	ηy = Cbar * ηe

	EN = zeros(nχ, N * nχ)
	EN[:, (N - 1) * nχ + 1:N * nχ] .= Matrix{Float64}(I, nχ, nχ)
	ΦN = Me * EN * Φ
	ΓN = Me * EN * Γ
	ηN = Me * EN * η

	return (
		Φe = Φe,
		Γe = Γe,
		ηe = ηe,
		Φv = Φv,
		Γv = Γv,
		ηv = ηv,
		Φs = Φs,
		Γs = Γs,
		ηs = ηs,
		Φy = Φy,
		Γy = Γy,
		ηy = ηy,
		ΦN = ΦN,
		ΓN = ΓN,
		ηN = ηN,
	)
end

function _build_cost(
	χk::Vector{Float64},
	mats,
	Qs_seq::Vector{Matrix{Float64}},
	Rv_seq::Vector{Matrix{Float64}},
	RΔ_seq::Vector{Matrix{Float64}},
	P::Matrix{Float64},
)
	Φs, Γs = mats.Φs, mats.Γs
	Φv, Γv = mats.Φv, mats.Γv
	ΦN, ΓN = mats.ΦN, mats.ΓN
	ηs, ηv, ηN = mats.ηs, mats.ηv, mats.ηN

	Qbar = _blockdiag_dense(Qs_seq)
	Rvbar = _blockdiag_dense(Rv_seq)
	RΔbar = _blockdiag_dense(RΔ_seq)

	H = 2.0 * (
		Γs' * Qbar * Γs +
		Γv' * Rvbar * Γv +
		RΔbar +
		ΓN' * P * ΓN
	)

	h = 2.0 * (
		Γs' * Qbar * (Φs * χk + ηs) +
		Γv' * Rvbar * (Φv * χk + ηv) +
		ΓN' * P * (ΦN * χk + ηN)
	)

	H = 0.5 * (H + H')
	return H, h
end

function _build_constraints(
	χk::Vector{Float64},
	mats;
	U_ref::Matrix{Float64},
	X_ref::Matrix{Float64},
	Y_ref::Matrix{Float64},
	ΔVmin::Matrix{Float64},
	ΔVmax::Matrix{Float64},
	Umin::Matrix{Float64},
	Umax::Matrix{Float64},
	Xmin::Matrix{Float64},
	Xmax::Matrix{Float64},
	Ymin::Matrix{Float64},
	Ymax::Matrix{Float64},
)
	Φe, Γe = mats.Φe, mats.Γe
	Φv, Γv = mats.Φv, mats.Γv
	Φy, Γy = mats.Φy, mats.Γy
	ηe, ηv, ηy = mats.ηe, mats.ηv, mats.ηy

	nv = size(Γv, 2)
	A_list = Matrix{Float64}[]
	b_list = Vector{Float64}[]

	IΔ = Matrix{Float64}(I, nv, nv)
	push!(A_list, IΔ)
	push!(b_list, _stackcols(ΔVmax))
	push!(A_list, -IΔ)
	push!(b_list, -_stackcols(ΔVmin))

	push!(A_list, Γv)
	push!(b_list, _stackcols(Umax) - _stackcols(U_ref) - Φv * χk - ηv)
	push!(A_list, -Γv)
	push!(b_list, -_stackcols(Umin) + _stackcols(U_ref) + Φv * χk + ηv)

	push!(A_list, Γe)
	push!(b_list, _stackcols(Xmax) - _stackcols(X_ref) - Φe * χk - ηe)
	push!(A_list, -Γe)
	push!(b_list, -_stackcols(Xmin) + _stackcols(X_ref) + Φe * χk + ηe)

	push!(A_list, Γy)
	push!(b_list, _stackcols(Ymax) - _stackcols(Y_ref) - Φy * χk - ηy)
	push!(A_list, -Γy)
	push!(b_list, -_stackcols(Ymin) + _stackcols(Y_ref) + Φy * χk + ηy)

	return vcat(A_list...), vcat(b_list...)
end

function _solve_tracking_qp(
	χk::Vector{Float64},
	uref0::Vector{Float64};
	H::Matrix{Float64},
	h::Vector{Float64},
	Aqp::Matrix{Float64},
	bqp::Vector{Float64},
	nu::Int,
	warm_start::Vector{Float64} = Float64[],
)
	nΔ = length(h)
	model = Model(optimizer_with_attributes(Ipopt.Optimizer, "tol" => 1e-3, "print_level" => 0))
	set_silent(model)

	@variable(model, ΔU[1:nΔ])
	if length(warm_start) == nΔ
		set_start_value.(ΔU, warm_start)
	end
	@objective(
		model,
		Min,
		0.5 * sum(H[i, j] * ΔU[i] * ΔU[j] for i in 1:nΔ, j in 1:nΔ) +
		sum(h[i] * ΔU[i] for i in 1:nΔ),
	)

	m = size(Aqp, 1)
	@constraint(model, [r = 1:m], sum(Aqp[r, c] * ΔU[c] for c in 1:nΔ) <= bqp[r])

	optimize!(model)
	term = termination_status(model)
    println("QP solve termination status: $term")
	# if term == MOI.ALMOST_LOCALLY_FEASIBLE
	# 	@warn "tracking MPC QP near-infeasible (ALMOST_LOCALLY_FEASIBLE); using best available solution"
	# elseif !(term in (MOI.OPTIMAL, MOI.LOCALLY_SOLVED, MOI.ALMOST_OPTIMAL))
	# 	@warn("tracking MPC QP solve failed with status = $term")
	# end

	ΔU_star = value.(ΔU)
	Δv0 = ΔU_star[1:nu]
	nx = length(χk) - nu
	v_prev = χk[nx + 1:end]
	v0_star = v_prev + Δv0
	u0_star = uref0 + v0_star

	return ΔU_star, v0_star, u0_star
end

function trackingmpc(integrator)
	N = integrator.p.mpc_params.n_horizon
	dt = integrator.p.mpc_params.time_step
	t0 = integrator.t

	xk = Vector{Float64}(integrator.u[1:6])
	nu = 2
	nx = length(xk)
	current_control_fallback = [Float64(integrator.p.α), Float64(integrator.p.β)]

	nominal = integrator.p.nominal_trajectory
	xref_at(τ) = [
		nominal[1](τ),
		nominal[2](τ),
		nominal[3](τ),
		nominal[4](τ),
		nominal[5](τ),
		nominal[6](τ),
	]

	model_times = t0 .+ (0:(N - 1)) .* dt
	prediction_times = t0 .+ (1:N) .* dt
	x_ref_now = xref_at(t0)
	X_ref = zeros(nx, N)
	X_model = zeros(nx, N)
	for j in 1:N
		X_model[:, j] .= xref_at(model_times[j])
		X_ref[:, j] .= xref_at(prediction_times[j])
	end

	U_ref = reduce(hcat, (_reference_control_at(τ, current_control_fallback) for τ in model_times))
	Y_ref = copy(X_ref)

	ek = xk - x_ref_now
	prev_u = Vector{Float64}(integrator.p.mpc_params.prev_x[nx + 1:nx + nu])
	# Use the reference at the *previous* step as the baseline so that
	# v_{k-1} = u_{k-1} - u_ref_{k-1}, not u_ref_{k}.
	uref_prev = _reference_control_at(t0 - dt, current_control_fallback)
	if !all(isfinite, prev_u) || norm(prev_u) == 0.0
		prev_u .= uref_prev
	end
	v_prev = prev_u - uref_prev
	χk = vcat(ek, v_prev)

	A_seq = Vector{Matrix{Float64}}(undef, N)
	B_seq = Vector{Matrix{Float64}}(undef, N)
	d_seq = Vector{Vector{Float64}}(undef, N)
	C_seq = [Matrix{Float64}(I, nx, nx) for _ in 1:N]

	state_scales = [1.0e5, 1.0, 1.0, 1.0e4, 1.0, 1.0]
	G = Diagonal(1.0 ./ state_scales)
	G_seq = [Matrix{Float64}(G) for _ in 1:N]

	for j in 1:N
		A_seq[j], B_seq[j] = _generated_discrete_linearization_si(X_model[:, j], U_ref[:, j], dt)
		d_seq[j] = _nominal_reentry_step_si(X_model[:, j], U_ref[:, j], dt) - X_ref[:, j]
	end

	# Qs weights the normalised sliding variable s = G*e.  Because G scales altitude
	# by 1e-5 and velocity by 1e-4, the effective physical-unit cost is Qs*(G*e)^2.
	# Weights are chosen so that equal fractional deviations cost the same:
	#   h: 1 km error → s = 0.01   → want significant cost
	#   v: 100 m/s    → s = 0.01
	#   γ/θ: 0.01 rad → s = 0.01
	# Previous Qs[h]=10, Qs[v]=10 gave effective weights 1e9× smaller than angles;
	# the fix is to raise them proportionally.
	Qs = Diagonal([1000.0, 5000.0, 5000.0, 1000.0, 500.0, 1000.0])
	Qs_seq = [Matrix{Float64}(Qs) for _ in 1:N]
	# Keep controls free enough to reject model mismatch, but avoid using bank as
	# a nearly-free crossrange actuator when its predicted benefit is ambiguous.
	Rv = Diagonal([1.0e-2, 0.1])
	RΔ = Diagonal([0.5, 0.5])
	Rv_seq = [Matrix{Float64}(Rv) for _ in 1:N]
	RΔ_seq = [Matrix{Float64}(RΔ) for _ in 1:N]
	P_normalized = Diagonal([10000.0, 10000.0, 10000.0, 3000.0, 1000.0, 5000.0])
	P = Matrix{Float64}(G' * P_normalized * G)

	αmin = deg2rad(-90.0)
	αmax = deg2rad(90.0)
	βmin = deg2rad(-89.0)
	βmax = deg2rad(89.0)
	# Per-step rate bounds centred on the reference motion: Δv_j ∈ [−Δu_max − δu_ref_j,
	# Δu_max − δu_ref_j].  This ensures that merely following the reference does not
	# violate the rate constraint.  uref_prev was computed above for the v_prev fix.
	Δumax_vec = _TRACKING_INPUT_RATE_LIMIT_RAD_PER_SEC .* dt
	U_ref_ext = hcat(uref_prev, U_ref)   # col 1 = previous ref; cols 2..N+1 = horizon refs
	ΔVmin = zeros(nu, N)
	ΔVmax = zeros(nu, N)
	for j in 1:N
		δu_ref_j = U_ref_ext[:, j + 1] - U_ref_ext[:, j]
		ΔVmin[:, j] = -Δumax_vec - δu_ref_j
		ΔVmax[:, j] =  Δumax_vec - δu_ref_j
	end
	Umin = repeat([αmin, βmin], 1, N)
	Umax = repeat([αmax, βmax], 1, N)

	Xmin_vec = _TRACKING_STATE_WIDE_MIN
	Xmax_vec = _TRACKING_STATE_WIDE_MAX
	Xmin = hcat([Xmin_vec for _ in 1:N]...)
	Xmax = hcat([Xmax_vec for _ in 1:N]...)
	Ymin = copy(Xmin)
	Ymax = copy(Xmax)

	Achi_seq, Bchi_seq, cchi_seq = _build_augmented_matrices(A_seq, B_seq, d_seq)
	Φ, Γ, η = _build_prediction_matrices(Achi_seq, Bchi_seq, cchi_seq)
	mats = _build_extraction_matrices(Φ, Γ, η, C_seq, G_seq, nx, nu, N)

	H, h = _build_cost(χk, mats, Qs_seq, Rv_seq, RΔ_seq, P)
	Aqp, bqp = _build_constraints(
		χk,
		mats;
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
		Ymax = Ymax,
	)

	# Shift the previous solution by one step: drop the first nu elements (already
	# applied) and pad with zeros at the tail as a neutral guess for the new step.
	prev_ΔU = integrator.p.mpc_params.prev_ΔU[]
	warm_start = if length(prev_ΔU) == N * nu
		[prev_ΔU[nu + 1:end]; zeros(nu)]
	else
		zeros(N * nu)
	end

	ΔU_star, _, u0_star = _solve_tracking_qp(
		χk,
		U_ref[:, 1];
		H = H,
		h = h,
		Aqp = Aqp,
		bqp = bqp,
		nu = nu,
		warm_start = warm_start,
	)

	integrator.p.mpc_params.prev_ΔU[] = ΔU_star

	α_cmd = clamp(u0_star[1], αmin, αmax)
	β_cmd = clamp(u0_star[2], βmin, βmax)
	integrator.p.mpc_params.prev_x[nx + 1:nx + nu] .= [α_cmd, β_cmd]

	χ_pred = mats.Φe * χk + mats.Γe * ΔU_star + mats.ηe
	V_pred = mats.Φv * χk + mats.Γv * ΔU_star + mats.ηv
	E_pred = reshape(χ_pred, nx, N)
	U_err_pred = reshape(V_pred, nu, N)
	X_act_pred = X_ref + E_pred
	U_pred = U_ref + U_err_pred
	β_pred = vec(U_pred[2, :])

	integrator.p.optimization_states = OptimizationStates(
		h_c = vec(X_act_pred[1, :]),
		ϕ_c = vec(X_act_pred[2, :]),
		θ_c = vec(X_act_pred[3, :]),
		v_c = vec(X_act_pred[4, :]),
		γ_c = vec(X_act_pred[5, :]),
		ψ_c = vec(X_act_pred[6, :]),
		β_c = β_pred,
	)

	return β_cmd, α_cmd
end
