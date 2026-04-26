const _MPG_INTEGRAL_STATE_NORMALIZED_WEIGHTS = Ref(collect(diag(_mpg_stage_state_normalized_weight_matrix())))
const _MPG_INTEGRAL_INCREMENT_WEIGHTS = Ref([1.0, 1.0])
const _MPG_INTEGRAL_STATE_GAIN = Ref(0.01)
const _MPG_INTEGRAL_INCREMENT_GAIN = Ref(0.001)

function _mpg_integral_state_normalized_weight_matrix()
    return Diagonal(Float64.(_MPG_INTEGRAL_STATE_NORMALIZED_WEIGHTS[]))
end

function _mpg_integral_increment_weight_matrix()
    return Diagonal(Float64.(_MPG_INTEGRAL_INCREMENT_WEIGHTS[]))
end

function mpg_integral_tuning_snapshot()
    return (
        state_normalized_weights = Float64.(_MPG_INTEGRAL_STATE_NORMALIZED_WEIGHTS[]),
        increment_weights = Float64.(_MPG_INTEGRAL_INCREMENT_WEIGHTS[]),
        state_gain = _MPG_INTEGRAL_STATE_GAIN[],
        increment_gain = _MPG_INTEGRAL_INCREMENT_GAIN[],
    )
end

function set_mpg_integral_tuning!(;
    state_normalized_weights=nothing,
    increment_weights=nothing,
    state_gain=nothing,
    increment_gain=nothing,
)
    if state_normalized_weights !== nothing
        weights = Float64.(collect(state_normalized_weights))
        length(weights) == 6 || throw(ArgumentError("state_normalized_weights must have length 6"))
        all(isfinite, weights) || throw(ArgumentError("state_normalized_weights must be finite"))
        all(>=(0.0), weights) || throw(ArgumentError("state_normalized_weights must be nonnegative"))
        _MPG_INTEGRAL_STATE_NORMALIZED_WEIGHTS[] = weights
    end
    if increment_weights !== nothing
        weights = Float64.(collect(increment_weights))
        length(weights) == 2 || throw(ArgumentError("increment_weights must have length 2"))
        all(isfinite, weights) || throw(ArgumentError("increment_weights must be finite"))
        all(>=(0.0), weights) || throw(ArgumentError("increment_weights must be nonnegative"))
        _MPG_INTEGRAL_INCREMENT_WEIGHTS[] = weights
    end
    if state_gain !== nothing
        gain = Float64(state_gain)
        (isfinite(gain) && gain >= 0.0) || throw(ArgumentError("state_gain must be finite and nonnegative"))
        _MPG_INTEGRAL_STATE_GAIN[] = gain
    end
    if increment_gain !== nothing
        gain = Float64(increment_gain)
        (isfinite(gain) && gain >= 0.0) || throw(ArgumentError("increment_gain must be finite and nonnegative"))
        _MPG_INTEGRAL_INCREMENT_GAIN[] = gain
    end
    return mpg_integral_tuning_snapshot()
end

function _ensure_mpg_integral_state!(mpc_params, dim::Int)
    integral_error = mpc_params.integral_error[]
    if length(integral_error) != dim
        integral_error = zeros(dim)
        mpc_params.integral_error[] = integral_error
    end
    return integral_error
end

function _mpg_integral_prediction_matrices(
    C::AbstractMatrix{<:Real},
    Φ::Vector{Matrix{Float64}},
    Ψ::Vector{Matrix{Float64}},
    step_sizes::AbstractVector{<:Real},
)
    N = length(step_sizes)
    n_i = size(C, 1)
    n_x = size(Φ[1], 2)
    nz = size(Ψ[1], 2)

    Φi = Vector{Matrix{Float64}}(undef, N + 1)
    Ψi = Vector{Matrix{Float64}}(undef, N + 1)
    Φi[1] = zeros(n_i, n_x)
    Ψi[1] = zeros(n_i, nz)
    C_dense = Matrix{Float64}(C)

    for k in 2:N + 1
        h = Float64(step_sizes[k - 1])
        Φi[k] = Φi[k - 1] + h .* (C_dense * Φ[k - 1])
        Ψi[k] = Ψi[k - 1] + h .* (C_dense * Ψ[k - 1])
    end

    return Φi, Ψi
end

function _mpg_integral_weight_matrix(step_sizes::AbstractVector{<:Real})
    horizon_scale = max(sum(Float64.(step_sizes)), 1.0)
    integral_scales = mpg_state_scales() .* horizon_scale
    G_i = Diagonal(1.0 ./ integral_scales)
    return _MPG_INTEGRAL_STATE_GAIN[] .* Matrix{Float64}(G_i' * _mpg_integral_state_normalized_weight_matrix() * G_i)
end

function _add_mpg_integral_state_cost!(
    P_qp::Matrix{Float64},
    q_qp::Vector{Float64},
    Φi::Vector{Matrix{Float64}},
    Ψi::Vector{Matrix{Float64}},
    i0::AbstractVector{<:Real},
    dx0::AbstractVector{<:Real},
    step_sizes::AbstractVector{<:Real},
)
    Qi = _mpg_integral_weight_matrix(step_sizes)
    node_weights = _mpg_node_weights(step_sizes)

    for k in eachindex(node_weights)
        if node_weights[k] == 0.0
            continue
        end
        P_qp .+= node_weights[k] .* (Ψi[k]' * Qi * Ψi[k])
        q_qp .+= node_weights[k] .* (Ψi[k]' * (Qi * (Float64.(i0) + Φi[k] * Float64.(dx0))))
    end

    return Qi
end

function _add_mpg_actual_increment_cost!(
    P_qp::Matrix{Float64},
    q_qp::Vector{Float64},
    U_nodes::Matrix{Float64},
    previous_u::AbstractVector{<:Real},
    step_sizes::AbstractVector{<:Real};
    RΔ::AbstractMatrix{<:Real}=_mpg_integral_increment_weight_matrix(),
    gain::Real=_MPG_INTEGRAL_INCREMENT_GAIN[],
)
    m, nodes = size(U_nodes)
    interval_weights = _normalized_stage_weights(step_sizes)
    RΔ_matrix = Matrix{Float64}(RΔ)
    gain_scale = Float64(gain)

    first_weight = Float64(interval_weights[1])
    first_rng = 1:m
    first_ref_delta = U_nodes[:, 1] .- Float64.(previous_u)
    P_qp[first_rng, first_rng] .+= 2.0 .* gain_scale .* first_weight .* RΔ_matrix
    q_qp[first_rng] .+= 2.0 .* gain_scale .* first_weight .* (RΔ_matrix * first_ref_delta)

    for k in 2:nodes
        weight = Float64(interval_weights[k - 1])
        prev_rng = (m * (k - 2) + 1):(m * (k - 1))
        curr_rng = (m * (k - 1) + 1):(m * k)
        ref_delta = U_nodes[:, k] .- U_nodes[:, k - 1]

        P_qp[prev_rng, prev_rng] .+= 2.0 .* gain_scale .* weight .* RΔ_matrix
        P_qp[curr_rng, curr_rng] .+= 2.0 .* gain_scale .* weight .* RΔ_matrix
        P_qp[prev_rng, curr_rng] .-= 2.0 .* gain_scale .* weight .* RΔ_matrix
        P_qp[curr_rng, prev_rng] .-= 2.0 .* gain_scale .* weight .* RΔ_matrix

        q_qp[prev_rng] .-= 2.0 .* gain_scale .* weight .* (RΔ_matrix * ref_delta)
        q_qp[curr_rng] .+= 2.0 .* gain_scale .* weight .* (RΔ_matrix * ref_delta)
    end

    return RΔ_matrix
end

function mpg_integral_tracking(integrator)
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

    Q = _mpg_stage_state_cost_matrix()
    F = _mpg_terminal_state_cost_matrix()
    R = _mpg_control_cost_matrix()
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

    integral_state = _ensure_mpg_integral_state!(integrator.p.mpc_params, n)
    C = Matrix{Float64}(I, n, n)
    Φi, Ψi = _mpg_integral_prediction_matrices(C, Φ, Ψ, step_sizes)
    _add_mpg_integral_state_cost!(P_qp, q_qp, Φi, Ψi, integral_state, dx0, step_sizes)

    u_min = _CONTROL_MIN_RAD
    u_max = _CONTROL_MAX_RAD
    prev_u = Vector{Float64}(integrator.p.mpc_params.prev_x[n + 1:n + m])
    if !all(isfinite, prev_u) || norm(prev_u) == 0.0
        prev_u .= [Float64(integrator.p.α), Float64(integrator.p.β)]
    end
    _add_mpg_actual_increment_cost!(
        P_qp,
        q_qp,
        U_nodes,
        prev_u,
        step_sizes;
        RΔ = _mpg_integral_increment_weight_matrix(),
        gain = _MPG_INTEGRAL_INCREMENT_GAIN[],
    )

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
        @warn "Integral MPg OSQP solve returned status $status; falling back to reference control"
        du_star = zeros(nz)
    end
    integrator.p.mpc_params.prev_ΔU[] = du_star

    u_cmd = clamp.(U_nodes[:, 1] .+ du_star[1:m], u_min, u_max)
    α_cmd = u_cmd[1]
    β_cmd = u_cmd[2]
    integrator.p.mpc_params.prev_x[n + 1:n + m] .= [α_cmd, β_cmd]

    dt_update = _control_step_size(integrator)
    dt_update = isfinite(dt_update) && dt_update > 0.0 ? dt_update : integrator.p.mpc_params.time_step
    integrator.p.mpc_params.integral_error[] = integral_state .+ dt_update .* dx0

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

const mpg_integral = mpg_integral_tracking
