using LinearAlgebra
using OSQP
using SparseArrays

const _MPG_MAX_HORIZON = 80

function _mpg_node_weights(step_sizes::AbstractVector{<:Real})
	N = length(step_sizes)
	w = zeros(N + 1)
	for i in 1:N
		wi = Float64(step_sizes[i]) / 4.0
		w[i] += wi
		w[i + 1] += wi
	end
	return w
end

function _mpg_heun_prediction_matrices(
	A::Vector{Matrix{Float64}},
	B::Vector{Matrix{Float64}},
	step_sizes::AbstractVector{<:Real},
)
	N = length(step_sizes)
	n = size(A[1], 1)
	m = size(B[1], 2)
	In = Matrix{Float64}(I, n, n)

	S = Vector{Matrix{Float64}}(undef, N)
	P_rk = Vector{Matrix{Float64}}(undef, N)
	Q_rk = Vector{Matrix{Float64}}(undef, N)
	for i in 1:N
		h = Float64(step_sizes[i])
		S[i] = In + (h / 2.0) * (A[i] + A[i + 1]) + (h^2 / 2.0) * (A[i + 1] * A[i])
		P_rk[i] = (h / 2.0) * (In + h * A[i + 1]) * B[i]
		Q_rk[i] = (h / 2.0) * B[i + 1]
	end

	Φ = Vector{Matrix{Float64}}(undef, N + 1)
	Φ[1] = In
	for k in 2:N + 1
		Φ[k] = S[k - 1] * Φ[k - 1]
	end

	Ψ = [zeros(n, m * (N + 1)) for _ in 1:N + 1]
	for k in 2:N + 1
		i = k - 1
		if k > 2
			Ψ[k][:, 1:m * (i - 1)] = S[i] * Ψ[k - 1][:, 1:m * (i - 1)]
		end

		col_P = (m * (i - 1) + 1):(m * i)
		Ψ[k][:, col_P] =
			(k > 2 ? S[i] * Ψ[k - 1][:, col_P] : zeros(n, m)) + P_rk[i]

		col_Q = (m * i + 1):(m * (i + 1))
		Ψ[k][:, col_Q] = Q_rk[i]
	end

	return Φ, Ψ
end

function _mpg_shift_warm_start(prev::Vector{Float64}, N::Int, m::Int)
	nz = m * (N + 1)
	if length(prev) != nz
		return zeros(nz)
	end
	return [prev[m + 1:end]; zeros(m)]
end

function _solve_mpg_box_qp(P_qp::Matrix{Float64}, q_qp::Vector{Float64}, lb::Vector{Float64}, ub::Vector{Float64}; warm_start::Vector{Float64} = Float64[])
	nz = length(q_qp)
	model = OSQP.Model()
	OSQP.setup!(
		model,
		P = sparse(triu(P_qp)),
		q = q_qp,
		A = sparse(Matrix{Float64}(I, nz, nz)),
		l = lb,
		u = ub,
		warm_starting = true,
		eps_abs = 1e-5,
		eps_rel = 1e-5,
		max_iter = 4000,
		verbose = false,
		polish = true,
	)
	if length(warm_start) == nz
		OSQP.warm_start!(model; x = warm_start)
	end
	result = OSQP.solve!(model)
	return result.x, lowercase(String(result.info.status))
end

function model_predictive_guidance(integrator)
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
		A[k], B[k] = _generated_continuous_linearization_si(X_nodes[:, k], U_nodes[:, k])
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
	P_qp = 0.5 .* (P_qp .+ P_qp') .+ 1.0e-9 .* Matrix{Float64}(I, nz, nz)

	αmin = deg2rad(-90.0)
	αmax = deg2rad(90.0)
	βmin = deg2rad(-89.0)
	βmax = deg2rad(89.0)
	u_min = [αmin, βmin]
	u_max = [αmax, βmax]
	du_limit = deg2rad.([15.0, 15.0])

	lb = zeros(nz)
	ub = zeros(nz)
	for k in 1:N + 1
		du_rng = (m * (k - 1) + 1):(m * k)
		lb[du_rng] .= max.(u_min .- U_nodes[:, k], .-du_limit)
		ub[du_rng] .= min.(u_max .- U_nodes[:, k], du_limit)
	end

	warm_start = _mpg_shift_warm_start(integrator.p.mpc_params.prev_ΔU[], N, m)
	du_star, status = _solve_mpg_box_qp(P_qp, q_qp, lb, ub; warm_start = warm_start)
	if !(status in ("solved", "solved inaccurate"))
		@warn "MPg OSQP solve returned status $status; falling back to reference control"
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

const mpg = model_predictive_guidance
