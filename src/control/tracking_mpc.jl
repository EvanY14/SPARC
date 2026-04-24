using LinearAlgebra
using JuMP
using Ipopt
using OSQP
using CSV
using DataFrames
using Interpolations

const MOI = JuMP.MOI
include(joinpath(@__DIR__, "..", "reference", "reentry_jacobians_si.jl"))
const _OPTIMAL_TRAJECTORY_PATH = joinpath(@__DIR__, "..", "..", "optimal_trajectory.csv")
const _OPTIMAL_CONTROL_CACHE = Ref{Any}(nothing)

# These are intentionally permissive: tracking feasibility should come from the
# reference cost and actuator limits, not brittle hard state cuts on a regenerated
# nominal trajectory.
const _TRACKING_STATE_WIDE_MIN = [-1.0e7, -100.0 * π, -100.0 * π, -1.0e5, -100.0 * π, -100.0 * π]
const _TRACKING_STATE_WIDE_MAX = [1.0e7, 100.0 * π, 100.0 * π, 1.0e5, 100.0 * π, 100.0 * π]
const _TRACKING_USE_ENERGY_REFERENCE_ALIGNMENT = false
const _TRACKING_SLIDING_STAGE_WEIGHTS = Diagonal([3000.0, 3000.0, 5000.0, 100.0, 10.0, 100.0])
const _TRACKING_SLIDING_TERMINAL_WEIGHTS = Diagonal([3000.0, 3000.0, 3000.0, 100.0, 10.0, 100.0])
const _TRACKING_CONTROL_DEVIATION_WEIGHTS = Diagonal([1.0e-2, 0.1])
const _TRACKING_CONTROL_INCREMENT_WEIGHTS = Diagonal([0.5, 0.5])

_tracking_state_scale_matrix() = _mpg_state_scale_matrix()
_tracking_sliding_stage_weight_matrix() = _TRACKING_SLIDING_STAGE_WEIGHTS
_tracking_sliding_terminal_weight_matrix() = _TRACKING_SLIDING_TERMINAL_WEIGHTS
_tracking_control_deviation_weight_matrix() = _TRACKING_CONTROL_DEVIATION_WEIGHTS
_tracking_control_increment_weight_matrix() = _TRACKING_CONTROL_INCREMENT_WEIGHTS

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
	if cache !== nothing && cache.path == path && cache.mtime == mtime && hasproperty(cache, :t)
		return cache
	end

	df = CSV.read(path, DataFrame)
	t = _csv_column(df, (:Time_s, :time_s, :time, :t))
	α_deg = _csv_column(df, (:AngleOfAttack_deg, :Alpha_deg, :AOA_deg))
	β_deg = _csv_column(df, (:BankAngle_deg, :Beta_deg, :Bank_deg))
	α_rad = _csv_column(df, (:AngleOfAttack_rad, :Alpha_rad, :AOA_rad))
	β_rad = _csv_column(df, (:BankAngle_rad, :Beta_rad, :Bank_rad))
	h_100km = _csv_column(df, (:Altitude_100km,))
	h_m = _csv_column(df, (:Altitude_m, :Altitude, :altitude_m, :h))
	v_1000mps = _csv_column(df, (:Velocity_1000mps,))
	v_mps = _csv_column(df, (:Velocity_mps, :Velocity_ms, :Velocity, :velocity_mps, :v))

	if t === nothing || (α_deg === nothing && α_rad === nothing) || (β_deg === nothing && β_rad === nothing)
		return nothing
	end

	α = α_rad === nothing ? deg2rad.(α_deg) : α_rad
	β = β_rad === nothing ? deg2rad.(β_deg) : β_rad
	h = h_m === nothing ? (h_100km === nothing ? nothing : h_100km .* 1.0e5) : h_m
	v = v_mps === nothing ? (v_1000mps === nothing ? nothing : v_1000mps .* 1.0e3) : v_mps
	order = sortperm(t)
	t_sorted = t[order]
	α_sorted = α[order]
	β_sorted = β[order]
	h_sorted = h === nothing ? nothing : h[order]
	v_sorted = v === nothing ? nothing : v[order]

	cache = (
		path = String(path),
		mtime = mtime,
		t_min = first(t_sorted),
		t_max = last(t_sorted),
		t = t_sorted,
		h = h_sorted,
		v = v_sorted,
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

function _reference_control_matrix(
	times::AbstractVector{<:Real},
	fallback::AbstractVector{<:Real},
)
	U_ref = zeros(2, length(times))
	for (j, τ) in pairs(times)
		U_ref[:, j] .= _reference_control_at(τ, fallback)
	end
	return U_ref
end

function _reference_time_bounds()
	cache = _load_optimal_control_cache()
	if cache === nothing
		return nothing
	end
	return cache.t_min, cache.t_max
end

function _specific_energy_si(
	x::AbstractVector{<:Real};
	μ::Real = 3.986004418e14,
	R::Real = 6378137.0,
)
	h = Float64(x[1])
	v = Float64(x[4])
	return Float64(μ) / (Float64(R) + h) - 0.5 * v^2
end

_specific_energy_si(h::Real, v::Real; μ::Real = 3.986004418e14, R::Real = 6378137.0) =
	Float64(μ) / (Float64(R) + Float64(h)) - 0.5 * Float64(v)^2

function _linear_interp_clamped(xs::AbstractVector{<:Real}, ys::AbstractVector{<:Real}, x::Real)
	n = length(xs)
	if n == 0
		return nothing
	elseif n == 1 || x <= xs[1]
		return Float64(ys[1])
	elseif x >= xs[end]
		return Float64(ys[end])
	end

	i = searchsortedlast(xs, x)
	i = clamp(i, 1, n - 1)
	x0 = Float64(xs[i])
	x1 = Float64(xs[i + 1])
	if x1 == x0
		return Float64(ys[i])
	end
	λ = (Float64(x) - x0) / (x1 - x0)
	return (1.0 - λ) * Float64(ys[i]) + λ * Float64(ys[i + 1])
end

function _reference_energy_time_samples(; μ::Real = 3.986004418e14, R::Real = 6378137.0)
	cache = _load_optimal_control_cache()
	if cache === nothing || cache.h === nothing || cache.v === nothing
		return nothing
	end

	e = [_specific_energy_si(cache.h[i], cache.v[i]; μ = μ, R = R) for i in eachindex(cache.t)]
	order = sortperm(e)
	e_sorted = Float64.(e[order])
	t_sorted = Float64.(cache.t[order])

	e_unique = Float64[]
	t_unique = Float64[]
	for i in eachindex(e_sorted)
		if isempty(e_unique) || abs(e_sorted[i] - e_unique[end]) > max(1.0, 1.0e-10 * abs(e_sorted[i]))
			push!(e_unique, e_sorted[i])
			push!(t_unique, t_sorted[i])
		end
	end

	if length(e_unique) < 2
		return nothing
	end
	return e_unique, t_unique
end

function _reference_time_at_energy(e::Real; μ::Real = 3.986004418e14, R::Real = 6378137.0)
	samples = _reference_energy_time_samples(; μ = μ, R = R)
	if samples === nothing
		return nothing
	end

	e_samples, t_samples = samples
	e_span = e_samples[end] - e_samples[1]
	if !(isfinite(e_span) && e_span > 0.0)
		return nothing
	end

	# A very large extrapolation means the vehicle is no longer on the reference
	# energy range, so falling back to time-indexed tracking is safer.
	margin = 0.02 * e_span
	if Float64(e) < e_samples[1] - margin || Float64(e) > e_samples[end] + margin
		return nothing
	end
	return _linear_interp_clamped(e_samples, t_samples, e)
end

function _reference_energy_at_time(t::Real; μ::Real = 3.986004418e14, R::Real = 6378137.0)
	cache = _load_optimal_control_cache()
	if cache === nothing || cache.h === nothing || cache.v === nothing
		return nothing
	end
	h = _linear_interp_clamped(cache.t, cache.h, t)
	v = _linear_interp_clamped(cache.t, cache.v, t)
	if h === nothing || v === nothing
		return nothing
	end
	return _specific_energy_si(h, v; μ = μ, R = R)
end

function _energy_indexed_horizon(
	x_current::AbstractVector{<:Real},
	t0::Real,
	dt::Real,
	max_horizon::Int;
	μ::Real = 3.986004418e14,
	R::Real = 6378137.0,
	shrinking::Bool = false,
)
	t_ref0 = _reference_time_at_energy(_specific_energy_si(x_current; μ = μ, R = R); μ = μ, R = R)
	if t_ref0 === nothing
		return nothing
	end

	if shrinking
		model_times, prediction_times, physical_step_sizes =
			_shrinking_horizon_times(t_ref0, dt, max_horizon)
	else
		N = max(max_horizon, 1)
		model_times = Float64(t_ref0) .+ (0:(N - 1)) .* Float64(dt)
		prediction_times = Float64(t_ref0) .+ (1:N) .* Float64(dt)
		physical_step_sizes = fill(Float64(dt), N)
	end

	N = length(prediction_times)
	energy_step_sizes = zeros(N)
	for j in 1:N
		e0 = _reference_energy_at_time(model_times[j]; μ = μ, R = R)
		e1 = _reference_energy_at_time(prediction_times[j]; μ = μ, R = R)
		if e0 === nothing || e1 === nothing
			return nothing
		end
		energy_step_sizes[j] = e1 - e0
	end

	if !all(isfinite, energy_step_sizes) || any(energy_step_sizes .<= 0.0)
		return nothing
	end

	return (
		reference_time = Float64(t_ref0),
		model_times = Float64.(model_times),
		prediction_times = Float64.(prediction_times),
		energy_step_sizes = energy_step_sizes,
		physical_step_sizes = Float64.(physical_step_sizes),
	)
end

function _normalized_stage_weights(step_sizes::AbstractVector{<:Real})
	w = Float64.(step_sizes)
	if isempty(w) || !all(isfinite, w) || any(w .<= 0.0)
		return ones(length(w))
	end
	mean_w = sum(w) / length(w)
	if !(isfinite(mean_w) && mean_w > 0.0)
		return ones(length(w))
	end
	return w ./ mean_w
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
	# Replace pure LTI matrix exponential with Heun's 2nd-order RK discretization
	# to match the Model Predictive Guidance (MPG) implementation.
	I_n = Matrix{Float64}(I, n, n)
	Ad = I_n + Ac * dt + 0.5 * (Ac^2) * dt^2
	Bd = Bc * dt + 0.5 * Ac * Bc * dt^2
	return Ad, Bd
end

function _nominal_reentry_dynamics_si(
	x::AbstractVector{<:Real},
	u::AbstractVector{<:Real},
	;
	mass::Real = VEHICLE_MASS,
	area::Real = VEHICLE_REFERENCE_AREA,
	μ::Real = 3.986004418e14,
	R::Real = 6378137.0,
)
	h = Float64(x[1])
	θ = Float64(x[3])
	v = Float64(x[4])
	γ = Float64(x[5])
	ψ = Float64(x[6])
	α = Float64(u[1])
	β = Float64(u[2])

	m = Float64(mass)
	S = Float64(area)
	μ_si = Float64(μ)
	R_si = Float64(R)
	a0 = -0.20704
	a1 = 0.029244
	b0 = 0.07854
	b1 = -0.61592e-2
	b2 = 0.621408e-3

	ρ = earth_atmosphere_density(h)
	α_deg = rad2deg(α)
	cL = a0 + a1 * α_deg
	cD = b0 + b1 * α_deg + b2 * α_deg^2
	r = R_si + h
	g = μ_si / r^2
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
	;
	mass::Real = VEHICLE_MASS,
	area::Real = VEHICLE_REFERENCE_AREA,
	μ::Real = 3.986004418e14,
	R::Real = 6378137.0,
)
	k1 = _nominal_reentry_dynamics_si(x, u; mass = mass, area = area, μ = μ, R = R)
	k2 = _nominal_reentry_dynamics_si(x .+ 0.5 * dt .* k1, u; mass = mass, area = area, μ = μ, R = R)
	k3 = _nominal_reentry_dynamics_si(x .+ 0.5 * dt .* k2, u; mass = mass, area = area, μ = μ, R = R)
	k4 = _nominal_reentry_dynamics_si(x .+ dt .* k3, u; mass = mass, area = area, μ = μ, R = R)
	return Float64.(x) .+ (dt / 6.0) .* (k1 .+ 2.0 .* k2 .+ 2.0 .* k3 .+ k4)
end

function _generated_continuous_linearization_si(
	x_ref::AbstractVector{<:Real},
	u_ref::AbstractVector{<:Real},
	;
	mass::Real = VEHICLE_MASS,
	area::Real = VEHICLE_REFERENCE_AREA,
	μ::Real = 3.986004418e14,
	R::Real = 6378137.0,
)
	return _analytical_continuous_linearization_si(x_ref, u_ref; mass = mass, area = area, μ = μ, R = R)
end

function _generated_discrete_linearization_si(
	x_ref::AbstractVector{<:Real},
	u_ref::AbstractVector{<:Real},
	dt::Real,
	;
	mass::Real = VEHICLE_MASS,
	area::Real = VEHICLE_REFERENCE_AREA,
	μ::Real = 3.986004418e14,
	R::Real = 6378137.0,
)
	Ac, Bc = _generated_continuous_linearization_si(x_ref, u_ref; mass = mass, area = area, μ = μ, R = R)
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

	Φ_prev = Matrix{Float64}(I, nχ, nχ)
	η_prev = zeros(nχ)
	for i in 1:N
		row = (i - 1) * nχ + 1:i * nχ
		A = Achi_seq[i]
		Φ_i = A * Φ_prev
		η_i = A * η_prev + cchi_seq[i]

		Φ[row, :] .= Φ_i
		η[row] .= η_i
		if i > 1
			prev_row = (i - 2) * nχ + 1:(i - 1) * nχ
			prev_cols = 1:(i - 1) * nu
			Γ[row, prev_cols] .= A * Γ[prev_row, prev_cols]
		end
		Γ[row, (i - 1) * nu + 1:i * nu] .= Bchi_seq[i]

		Φ_prev = Φ_i
		η_prev = η_i
	end
	return Φ, Γ, η
end

function _build_extraction_matrices(
	Φ::Matrix{Float64},
	Γ::Matrix{Float64},
	η::Vector{Float64},
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
	ΔVmin::Matrix{Float64},
	ΔVmax::Matrix{Float64},
	Umin::Matrix{Float64},
	Umax::Matrix{Float64},
	Xmin::Matrix{Float64},
	Xmax::Matrix{Float64},
)
	Φe, Γe = mats.Φe, mats.Γe
	Φv, Γv = mats.Φv, mats.Γv
	ηe, ηv = mats.ηe, mats.ηv

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

function _solve_tracking_sparse_qp(
	ek::Vector{Float64},
	v_prev::Vector{Float64};
	U_ref::Matrix{Float64},
	X_ref::Matrix{Float64},
	A_seq::Vector{Matrix{Float64}},
	B_seq::Vector{Matrix{Float64}},
	d_seq::Vector{Vector{Float64}},
	G::Diagonal,
	Qs::Diagonal,
	Rv::Diagonal,
	RΔ::Diagonal,
	P::Matrix{Float64},
	ΔVmin::Matrix{Float64},
	ΔVmax::Matrix{Float64},
	Umin::Matrix{Float64},
	Umax::Matrix{Float64},
	Xmin::Matrix{Float64},
	Xmax::Matrix{Float64},
	warm_start::Vector{Float64} = Float64[],
	stage_weights::Vector{Float64} = ones(size(X_ref, 2)),
)
	nx, N = size(X_ref)
	nu = size(U_ref, 1)
	if length(stage_weights) != N
		stage_weights = ones(N)
	end
	model = Model(
		optimizer_with_attributes(
			OSQP.Optimizer,
			"verbose" => false,
			"eps_abs" => 1e-4,
			"eps_rel" => 1e-4,
			"max_iter" => 4000,
			"polish" => true,
		),
	)
	set_silent(model)

	@variable(model, e[1:nx, 1:N])
	@variable(model, v[1:nu, 1:N])
	@variable(model, Δv[1:nu, 1:N])

	if length(warm_start) == N * nu
		for j in 1:N, i in 1:nu
			set_start_value(Δv[i, j], warm_start[(j - 1) * nu + i])
		end
	end

	for j in 1:N
		for i in 1:nu
			prev_v_i = j == 1 ? v_prev[i] : v[i, j - 1]
			@constraint(model, v[i, j] == prev_v_i + Δv[i, j])
			@constraint(model, ΔVmin[i, j] <= Δv[i, j] <= ΔVmax[i, j])
			@constraint(model, Umin[i, j] <= U_ref[i, j] + v[i, j] <= Umax[i, j])
		end

		for i in 1:nx
			prev_e = j == 1 ? ek : e[:, j - 1]
			@constraint(
				model,
				e[i, j] ==
				sum(A_seq[j][i, k] * prev_e[k] for k in 1:nx) +
				sum(B_seq[j][i, k] * v[k, j] for k in 1:nu) +
				d_seq[j][i],
			)
			@constraint(model, Xmin[i, j] <= X_ref[i, j] + e[i, j] <= Xmax[i, j])
		end
	end

	@objective(
		model,
		Min,
		sum(stage_weights[j] * Qs[i, i] * (G[i, i] * e[i, j])^2 for i in 1:nx, j in 1:N) +
		sum(stage_weights[j] * Rv[i, i] * v[i, j]^2 for i in 1:nu, j in 1:N) +
		sum(RΔ[i, i] * Δv[i, j]^2 for i in 1:nu, j in 1:N) +
		sum(P[i, k] * e[i, N] * e[k, N] for i in 1:nx, k in 1:nx),
	)

	optimize!(model)
	term = termination_status(model)
	if !(term in (MOI.OPTIMAL, MOI.LOCALLY_SOLVED, MOI.ALMOST_OPTIMAL))
		@warn "tracking MPC OSQP solve returned status $term"
		ΔU_star = zeros(N * nu)
		V_star = repeat(v_prev, 1, N)
		E_star = zeros(nx, N)
		u0_star = U_ref[:, 1] + v_prev
		return ΔU_star, V_star, E_star, u0_star
	end

	ΔU_star = vec(value.(Δv))
	V_star = value.(v)
	E_star = value.(e)
	u0_star = U_ref[:, 1] + V_star[:, 1]
	return ΔU_star, V_star, E_star, u0_star
end

function trackingmpc(integrator)
	N = max(integrator.p.mpc_params.n_horizon, 1)
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

	energy_horizon = _TRACKING_USE_ENERGY_REFERENCE_ALIGNMENT ?
		_energy_indexed_horizon(
			xk,
			t0,
			dt,
			N;
			μ = integrator.p.μ,
			R = integrator.p.R,
			shrinking = false,
		) :
		nothing
	reference_time_now = energy_horizon === nothing ? Float64(t0) : energy_horizon.reference_time
	model_times, prediction_times, physical_step_sizes = if energy_horizon === nothing
		_shrinking_horizon_times(t0, dt, N)
	else
		energy_horizon.model_times, energy_horizon.prediction_times, energy_horizon.physical_step_sizes
	end
	N = length(prediction_times)
	x_ref_now = xref_at(reference_time_now)
	X_ref = zeros(nx, N)
	X_model = zeros(nx, N)
	for j in 1:N
		X_model[:, j] .= xref_at(model_times[j])
		X_ref[:, j] .= xref_at(prediction_times[j])
	end

	U_ref = _reference_control_matrix(model_times, current_control_fallback)
	ek = xk - x_ref_now
	prev_u = Vector{Float64}(integrator.p.mpc_params.prev_x[nx + 1:nx + nu])
	# Use the reference at the *previous* step as the baseline so that
	# v_{k-1} = u_{k-1} - u_ref_{k-1}, not u_ref_{k}.
	uref_prev = _reference_control_at(model_times[1] - physical_step_sizes[1], current_control_fallback)
	if !all(isfinite, prev_u) || norm(prev_u) == 0.0
		prev_u .= uref_prev
	end
	v_prev = prev_u - uref_prev
	χk = vcat(ek, v_prev)

	A_seq = Vector{Matrix{Float64}}(undef, N)
	B_seq = Vector{Matrix{Float64}}(undef, N)
	d_seq = Vector{Vector{Float64}}(undef, N)
	G = _tracking_state_scale_matrix()
	G_seq = [Matrix{Float64}(G) for _ in 1:N]

	for j in 1:N
		Δt = physical_step_sizes[j]
		A_seq[j], B_seq[j] = _generated_discrete_linearization_si(
			X_model[:, j],
			U_ref[:, j],
			Δt;
			mass = integrator.p.mass,
			area = integrator.p.area,
			μ = integrator.p.μ,
			R = integrator.p.R,
		)
		d_seq[j] = _nominal_reentry_step_si(
			X_model[:, j],
			U_ref[:, j],
			Δt;
			mass = integrator.p.mass,
			area = integrator.p.area,
			μ = integrator.p.μ,
			R = integrator.p.R,
		) - X_ref[:, j]
	end

	# Qs weights the normalised sliding variable s = G*e.  Because G scales altitude
	# by 1e-5 and velocity by 1e-4, the effective physical-unit cost is Qs*(G*e)^2.
	# Weights are chosen so that equal fractional deviations cost the same:
	#   h: 1 km error → s = 0.01   → want significant cost
	#   v: 100 m/s    → s = 0.01
	#   γ/θ: 0.01 rad → s = 0.01
	# Previous Qs[h]=10, Qs[v]=10 gave effective weights 1e9× smaller than angles;
	# the fix is to raise them proportionally.
	# Emulate output-tracking: heavily weight Altitude, Lat, Lon. Relax v, γ, ψ.
	Qs = _tracking_sliding_stage_weight_matrix()
	Qs_seq = [Matrix{Float64}(Qs) for _ in 1:N]
	# Keep controls free enough to reject model mismatch, but avoid using bank as
	# a nearly-free crossrange actuator when its predicted benefit is ambiguous.
	Rv = _tracking_control_deviation_weight_matrix()
	RΔ = _tracking_control_increment_weight_matrix()
	Rv_seq = [Matrix{Float64}(Rv) for _ in 1:N]
	RΔ_seq = [Matrix{Float64}(RΔ) for _ in 1:N]
	P_normalized = _tracking_sliding_terminal_weight_matrix()
	P = Matrix{Float64}(G' * P_normalized * G)

	αmin, βmin = _CONTROL_MIN_RAD
	αmax, βmax = _CONTROL_MAX_RAD
	# Per-step rate bounds centred on the reference motion: Δv_j ∈ [−Δu_max − δu_ref_j,
	# Δu_max − δu_ref_j].  This ensures that merely following the reference does not
	# violate the rate constraint.  uref_prev was computed above for the v_prev fix.
	U_ref_ext = hcat(uref_prev, U_ref)   # col 1 = previous ref; cols 2..N+1 = horizon refs
	ΔVmin = zeros(nu, N)
	ΔVmax = zeros(nu, N)
	for j in 1:N
		Δumax_vec = _CONTROL_RATE_LIMIT_RAD_PER_SEC .* physical_step_sizes[j]
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

	# Shift the previous solution by one step: drop the first nu elements (already
	# applied) and pad with zeros at the tail as a neutral guess for the new step.
	prev_ΔU = integrator.p.mpc_params.prev_ΔU[]
	warm_start = if length(prev_ΔU) == N * nu
		[prev_ΔU[nu + 1:end]; zeros(nu)]
	else
		zeros(N * nu)
	end

	ΔU_star, V_star, E_star, u0_star = _solve_tracking_sparse_qp(
		ek,
		v_prev;
		U_ref = U_ref,
		X_ref = X_ref,
		A_seq = A_seq,
		B_seq = B_seq,
		d_seq = d_seq,
		G = G,
		Qs = Qs,
		Rv = Rv,
		RΔ = RΔ,
		P = P,
		ΔVmin = ΔVmin,
		ΔVmax = ΔVmax,
		Umin = Umin,
		Umax = Umax,
		Xmin = Xmin,
		Xmax = Xmax,
		warm_start = warm_start,
	)

	integrator.p.mpc_params.prev_ΔU[] = ΔU_star

	α_cmd = clamp(u0_star[1], αmin, αmax)
	β_cmd = clamp(u0_star[2], βmin, βmax)
	integrator.p.mpc_params.prev_x[nx + 1:nx + nu] .= [α_cmd, β_cmd]

	X_act_pred = X_ref + E_star
	U_pred = U_ref + V_star
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
