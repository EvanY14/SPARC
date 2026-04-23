using LinearAlgebra

const _SM_MPG_Q4_LAMBDA = 0.7
const _SM_MPG_Q4_NORMALIZED_WEIGHTS = Diagonal([1200.0, 3500.0, 3500.0, 1200.0, 700.0, 1200.0])

function _sm_mpg_q4_add_tracking_cost!(
    P_qp::Matrix{Float64},
    q_qp::Vector{Float64},
    Φ::Vector{Matrix{Float64}},
    Ψ::Vector{Matrix{Float64}},
    dx0::Vector{Float64},
    step_sizes::AbstractVector{<:Real},
)
    N = length(step_sizes)
    λ = _SM_MPG_Q4_LAMBDA
    interval_weights = _normalized_stage_weights(step_sizes)
    C_slide = Matrix{Float64}(_tracking_state_scale_matrix())
    W_slide = Matrix{Float64}(_SM_MPG_Q4_NORMALIZED_WEIGHTS)

    for k in 1:N
        # Homework 3, Question 4 sliding variable: s_k = e_{k+1} - λ e_k.
        SΦ = C_slide * (Φ[k + 1] .- λ .* Φ[k])
        SΨ = C_slide * (Ψ[k + 1] .- λ .* Ψ[k])
        weight = Float64(interval_weights[k])
        P_qp .+= weight .* (SΨ' * W_slide * SΨ)
        q_qp .+= weight .* (SΨ' * (W_slide * (SΦ * dx0)))
    end

    return nothing
end

function sm_mpg_q4_tracking(integrator)
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

    _sm_mpg_q4_add_tracking_cost!(P_qp, q_qp, Φ, Ψ, dx0, step_sizes)
    _add_mpg_actual_increment_cost!(P_qp, q_qp, U_nodes, prev_u, step_sizes)

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
        @warn "Q4 SM-MPG OSQP solve returned status $status; falling back to reference control"
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

const sm_mpg_q4 = sm_mpg_q4_tracking
