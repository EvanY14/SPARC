const _SM_MPG_STAGE_GAIN = 1.0
const _SM_MPG_TERMINAL_GAIN = 1.0
const _SM_MPG_CONTROL_GAIN = 1.0
const _SM_MPG_INCREMENT_GAIN = 1.0

function _sm_mpg_add_tracking_cost!(
	P_qp::Matrix{Float64},
	q_qp::Vector{Float64},
	Φ::Vector{Matrix{Float64}},
	Ψ::Vector{Matrix{Float64}},
	dx0::Vector{Float64},
	step_sizes::AbstractVector{<:Real},
	prev_u::AbstractVector{<:Real},
	prev_u_ref::AbstractVector{<:Real},
)
	N = length(step_sizes)
	m = length(prev_u)
	node_weights = _mpg_node_weights(step_sizes)
	interval_weights = _normalized_stage_weights(step_sizes)

	G = _tracking_state_scale_matrix()
	Q_slide = Matrix{Float64}(G' * _tracking_sliding_stage_weight_matrix() * G)
	F_slide = Matrix{Float64}(G' * _tracking_sliding_terminal_weight_matrix() * G)
	Rv = Matrix{Float64}(_tracking_control_deviation_weight_matrix())
	RΔ = Matrix{Float64}(_tracking_control_increment_weight_matrix())
	v_prev = Float64.(prev_u) .- Float64.(prev_u_ref)

	for k in 2:N + 1
		weight = _SM_MPG_STAGE_GAIN * Float64(interval_weights[k - 1])
		P_qp .+= weight .* (Ψ[k]' * Q_slide * Ψ[k])
		q_qp .+= weight .* (Ψ[k]' * (Q_slide * (Φ[k] * dx0)))
	end

	P_qp .+= _SM_MPG_TERMINAL_GAIN .* (Ψ[N + 1]' * F_slide * Ψ[N + 1])
	q_qp .+= _SM_MPG_TERMINAL_GAIN .* (Ψ[N + 1]' * (F_slide * (Φ[N + 1] * dx0)))

	for k in 1:N + 1
		du_rng = (m * (k - 1) + 1):(m * k)
		P_qp[du_rng, du_rng] .+= (_SM_MPG_CONTROL_GAIN * Float64(node_weights[k])) .* Rv
	end

	first_weight = _SM_MPG_INCREMENT_GAIN * Float64(interval_weights[1])
	P_qp[1:m, 1:m] .+= 2.0 .* first_weight .* RΔ
	q_qp[1:m] .+= -2.0 .* first_weight .* (RΔ * v_prev)

	for k in 2:N + 1
		weight = _SM_MPG_INCREMENT_GAIN * Float64(interval_weights[k - 1])
		prev_rng = (m * (k - 2) + 1):(m * (k - 1))
		du_rng = (m * (k - 1) + 1):(m * k)
		P_qp[prev_rng, prev_rng] .+= 2.0 .* weight .* RΔ
		P_qp[du_rng, du_rng] .+= 2.0 .* weight .* RΔ
		P_qp[prev_rng, du_rng] .-= 2.0 .* weight .* RΔ
		P_qp[du_rng, prev_rng] .-= 2.0 .* weight .* RΔ
	end

	return v_prev
end

function sm_mpg(integrator)
	dt = integrator.p.mpc_params.time_step
	t0 = integrator.t
	Nmax = max(min(integrator.p.mpc_params.n_horizon, _MPG_MAX_HORIZON), 2)

	x_current = Vector{Float64}(integrator.u[1:6])
	n = length(x_current)
	m = 2
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

	model_times, prediction_times, step_sizes = _shrinking_horizon_times(t0, dt, Nmax)
	N = length(prediction_times)
	node_times = [Float64(t0); Float64.(prediction_times)]

	x_ref_current = xref_at(t0)
	X_nodes = zeros(n, N + 1)
	for k in 1:N + 1
		X_nodes[:, k] .= xref_at(node_times[k])
	end
	U_nodes = _reference_control_matrix(node_times, current_control_fallback)

	A = Vector{Matrix{Float64}}(undef, N + 1)
	B = Vector{Matrix{Float64}}(undef, N + 1)
	for k in 1:N + 1
		A[k], B[k] = _generated_continuous_linearization_si(
			X_nodes[:, k],
			U_nodes[:, k];
			mass = integrator.p.mass,
			area = integrator.p.area,
			μ = integrator.p.μ,
			R = integrator.p.R,
		)
	end

	Φ, Ψ = _mpg_heun_prediction_matrices(A, B, step_sizes)
	dx0 = x_current - x_ref_current
	w = _mpg_node_weights(step_sizes)

	state_scales = [1.0e5, 1.0, 1.0, 1.0e4, 1.0, 1.0]
	G = Diagonal(1.0 ./ state_scales)
	Q_normalized = Diagonal([1200.0, 3500.0, 3500.0, 1200.0, 700.0, 1200.0])
	F_normalized = Diagonal([2500.0, 180000.0, 180000.0, 4500.0, 2500.0, 4000.0])
	Q = Matrix{Float64}(G' * Q_normalized * G)
	F = Matrix{Float64}(G' * F_normalized * G)
	R = Diagonal([1.0, 1.0])
	kR = 1.0
	kF = 1.0

	nz = m * (N + 1)
	P_qp = zeros(nz, nz)
	q_qp = zeros(nz)
	for k in 1:N + 1
		du_rng = (m * (k - 1) + 1):(m * k)
		P_qp .+= w[k] .* (Ψ[k]' * Q * Ψ[k])
		P_qp[du_rng, du_rng] .+= w[k] .* kR .* R
		q_qp .+= w[k] .* (Ψ[k]' * (Q * (Φ[k] * dx0)))
	end
	P_qp .+= kF .* (Ψ[N + 1]' * F * Ψ[N + 1])
	q_qp .+= kF .* (Ψ[N + 1]' * (F * (Φ[N + 1] * dx0)))

	u_min = _CONTROL_MIN_RAD
	u_max = _CONTROL_MAX_RAD
	prev_u = Vector{Float64}(integrator.p.mpc_params.prev_x[n + 1:n + m])
	if !all(isfinite, prev_u) || norm(prev_u) == 0.0
		prev_u .= [Float64(integrator.p.α), Float64(integrator.p.β)]
	end
	prev_u_ref = _reference_control_at(Float64(t0) - _control_step_size(integrator), current_control_fallback)
	_sm_mpg_add_tracking_cost!(P_qp, q_qp, Φ, Ψ, dx0, step_sizes, prev_u, prev_u_ref)

	P_qp = 0.5 .* (P_qp .+ P_qp') .+ 1.0e-9 .* Matrix{Float64}(I, nz, nz)

	A_qp, lb, ub = _mpg_control_constraint_matrices(
		U_nodes,
		prev_u,
		step_sizes,
		_control_step_size(integrator),
	)

	warm_start = _mpg_shift_warm_start(integrator.p.mpc_params.prev_ΔU[], N, m)
	du_star, status = _solve_mpg_qp(P_qp, q_qp, A_qp, lb, ub; warm_start = warm_start)
	if !(status in ("solved", "solved inaccurate"))
		@warn "SM-MPG OSQP solve returned status $status; falling back to reference control"
		du_star = zeros(nz)
	end
	integrator.p.mpc_params.prev_ΔU[] = du_star

	u_cmd = clamp.(U_nodes[:, 1] .+ du_star[1:m], u_min, u_max)
	α_cmd = u_cmd[1]
	β_cmd = u_cmd[2]
	integrator.p.mpc_params.prev_x[n + 1:n + m] .= [α_cmd, β_cmd]

	X_pred = zeros(n, N + 1)
	for k in 1:N + 1
		X_pred[:, k] .= X_nodes[:, k] .+ Φ[k] * dx0 .+ Ψ[k] * du_star
	end
	U_pred = U_nodes .+ reshape(du_star, m, N + 1)
	integrator.p.optimization_states = OptimizationStates(
		h_c = vec(X_pred[1, :]),
		ϕ_c = vec(X_pred[2, :]),
		θ_c = vec(X_pred[3, :]),
		v_c = vec(X_pred[4, :]),
		γ_c = vec(X_pred[5, :]),
		ψ_c = vec(X_pred[6, :]),
		β_c = vec(U_pred[2, :]),
	)

	return β_cmd, α_cmd
end
