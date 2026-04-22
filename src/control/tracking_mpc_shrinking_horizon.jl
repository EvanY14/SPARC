function _shrinking_horizon_times(t0::Real, dt::Real, max_horizon::Int)
	max_horizon = max(max_horizon, 1)
	bounds = _reference_time_bounds()
	if bounds === nothing
		N = 1
		prediction_times = [Float64(t0 + dt)]
		model_times = [Float64(t0)]
		return model_times, prediction_times, [Float64(dt)]
	end

	_, t_ref_final = bounds
	remaining_time = Float64(t_ref_final - t0)
	if remaining_time <= 0.0
		N = 1
		prediction_times = [Float64(t0 + dt)]
	else
		N = min(max(ceil(Int, remaining_time / dt), 1), max_horizon)
		prediction_times = min.(Float64(t0) .+ collect(1:N) .* Float64(dt), Float64(t_ref_final))
	end

	model_times = [Float64(t0); prediction_times[1:end - 1]]
	step_sizes = prediction_times .- model_times
	return model_times, prediction_times, step_sizes
end

function _shifted_warm_start(prev_ΔU::Vector{Float64}, N::Int, nu::Int)
	warm_start = zeros(N * nu)
	if isempty(prev_ΔU)
		return warm_start
	end

	shifted = length(prev_ΔU) > nu ? prev_ΔU[nu + 1:end] : Float64[]
	ncopy = min(length(shifted), length(warm_start))
	if ncopy > 0
		warm_start[1:ncopy] .= shifted[1:ncopy]
	end
	return warm_start
end

function trackingmpc_shrinking_horizon(integrator)
	dt = integrator.p.mpc_params.time_step
	t0 = integrator.t
	max_horizon = max(integrator.p.mpc_params.n_horizon, 1)

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
			max_horizon;
			μ = integrator.p.μ,
			R = integrator.p.R,
			shrinking = true,
		) :
		nothing
	reference_time_now = energy_horizon === nothing ? Float64(t0) : energy_horizon.reference_time
	model_times, prediction_times, step_sizes = if energy_horizon === nothing
		_shrinking_horizon_times(t0, dt, max_horizon)
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
	uref_prev = _reference_control_at(model_times[1] - step_sizes[1], current_control_fallback)
	if !all(isfinite, prev_u) || norm(prev_u) == 0.0
		prev_u .= uref_prev
	end
	v_prev = prev_u - uref_prev
	χk = vcat(ek, v_prev)

	A_seq = Vector{Matrix{Float64}}(undef, N)
	B_seq = Vector{Matrix{Float64}}(undef, N)
	d_seq = Vector{Vector{Float64}}(undef, N)
	state_scales = [1.0e5, 1.0, 1.0, 1.0e4, 1.0, 1.0]
	G = Diagonal(1.0 ./ state_scales)
	G_seq = [Matrix{Float64}(G) for _ in 1:N]

	for j in 1:N
		Δt = step_sizes[j]
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

	# Emulate output-tracking: heavily weight Altitude, Lat, Lon. Relax v, γ, ψ.
	Qs = Diagonal([100.0, 30000.0, 30000.0, 10.0, 10.0, 10.0])
	Qs_seq = [Matrix{Float64}(Qs) for _ in 1:N]
	Rv = Diagonal([1.0e-2, 0.1])
	RΔ = Diagonal([0.5, 0.5])
	Rv_seq = [Matrix{Float64}(Rv) for _ in 1:N]
	RΔ_seq = [Matrix{Float64}(RΔ) for _ in 1:N]
	# Relax massive terminal weights to prevent late-trajectory chattering
	P_normalized = Diagonal([1000.0, 3000.0, 3000.0, 100.0, 100.0, 100.0])
	P = Matrix{Float64}(G' * P_normalized * G)

	αmin, βmin = _CONTROL_MIN_RAD
	αmax, βmax = _CONTROL_MAX_RAD
	U_ref_ext = hcat(uref_prev, U_ref)
	ΔVmin = zeros(nu, N)
	ΔVmax = zeros(nu, N)
	for j in 1:N
		Δumax_vec = _CONTROL_RATE_LIMIT_RAD_PER_SEC .* step_sizes[j]
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

	warm_start = _shifted_warm_start(integrator.p.mpc_params.prev_ΔU[], N, nu)
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

const trackingmpc_shrinking = trackingmpc_shrinking_horizon
