function _reference_time_bounds()
	cache = _load_optimal_control_cache()
	if cache === nothing
		return nothing
	end
	return cache.t_min, cache.t_max
end

function _shrinking_horizon_times(t0::Real, dt::Real)
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
		N = max(ceil(Int, remaining_time / dt), 1)
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

	model_times, prediction_times, step_sizes = _shrinking_horizon_times(t0, dt)
	N = length(prediction_times)
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
		Δt = step_sizes[j]
		A_seq[j], B_seq[j] = _generated_discrete_linearization_si(X_model[:, j], U_ref[:, j], Δt)
		d_seq[j] = _nominal_reentry_step_si(X_model[:, j], U_ref[:, j], Δt) - X_ref[:, j]
	end

	Qs = Diagonal([5000.0, 500.0, 5000.0, 5000.0, 500.0, 1000.0])
	Qs_seq = [Matrix{Float64}(Qs) for _ in 1:N]
	Rv = Diagonal([1.0e-2, 0.1])
	RΔ = Diagonal([0.5, 0.5])
	Rv_seq = [Matrix{Float64}(Rv) for _ in 1:N]
	RΔ_seq = [Matrix{Float64}(RΔ) for _ in 1:N]
	P_normalized = Diagonal([10000.0, 2000.0, 10000.0, 10000.0, 1000.0, 5000.0])
	P = Matrix{Float64}(G' * P_normalized * G)

	αmin = deg2rad(-90.0)
	αmax = deg2rad(90.0)
	βmin = deg2rad(-89.0)
	βmax = deg2rad(89.0)
	U_ref_ext = hcat(uref_prev, U_ref)
	ΔVmin = zeros(nu, N)
	ΔVmax = zeros(nu, N)
	for j in 1:N
		Δumax_vec = _TRACKING_INPUT_RATE_LIMIT_RAD_PER_SEC .* step_sizes[j]
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

	warm_start = _shifted_warm_start(integrator.p.mpc_params.prev_ΔU[], N, nu)
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

const trackingmpc_shrinking = trackingmpc_shrinking_horizon
