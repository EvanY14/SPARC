function edl_dynamics_ssi_mpc!(du::MVector{7, Float64}, u::MVector{7, Float64}, p::EDLParams, t::Float64, β::Float64)
    h, ϕ, θ, v, γ, ψ, q = u
    m = p.mass
    Cd = p.Cd
    Cl = p.Cl
    A = p.area
    μ = p.μ
    R = p.R # Planetary radius
    # β = β # Bank angle
    r = R + h # Distance from planet center
    C1 = 8.53e-13 # Constant for convective heat rate calculation
    n_exp = 0.82958 # Exponent for convective heat rate calculation
    m_exp = 4.512 # Exponent for convective heat rate calculation

    density = earth_atmosphere_density(h)
    # println("Time: $t s, Altitude: $h m, Velocity: $v m/s, Relative Velocity: $v_rel m/s, Density: $(p.atmospheric_density) kg/m³")
    drag = 0.5 * density * v^2 * Cd * A # Drag force
    lift = 0.5 * density * v^2 * Cl * A # Lift force

    # Trig functions
    sin_γ = sin(γ)
    cos_γ = cos(γ)
    sin_ψ = sin(ψ)
    cos_ψ = cos(ψ)
    tan_θ = tan(θ)
    cos_θ = cos(θ)
    sin_β = sin(β)
    cos_β = cos(β)

    g = μ / r^2 # Gravitational acceleration
    heat_rate = C1 * density^n_exp * v^m_exp # Convective heat rate (W/m^2)
    # p.cache.q_dot = heat_rate # Store heat rate in cache
    du[1] = (v * sin_γ) #  h_dot
    du[2] = (v/r) * cos_γ * sin_ψ / cos_θ # ϕ_dot (longitude)
    du[3] = (v/r) * cos_γ * cos_ψ # θ_dot (latitude)
    du[4] = (-drag / m - g * sin_γ) # v_dot
    du[5] = (lift/(m*v) * cos_β) + cos_γ*(v/r - g/v) # γ_dot (flight path angle)
    du[6] = (lift * sin_β) / (m * v * cos_γ) + (v * cos_γ * sin_ψ * tan_θ) / r # ψ_dot (azimuth)
    du[7] = heat_rate # heat rate (W/m^2)
end

function ssimpc(integrator)
    # Placeholder for SSI-MPC implementation
     # --- 1. Define Scaling Constants ---
    H_SCALE = integrator.p.mpc_params.H_SCALE   # Reference Altitude (100 km)
    V_SCALE = integrator.p.mpc_params.V_SCALE   # Reference Velocity (1 km/s)
    # T_SCALE = 100.0 # Reference Time (100 s) 
    T_SCALE = integrator.p.mpc_params.T_SCALE
    
    # Derived Scaled Rate Units (used for dynamics)
    DH_SCALE_RATE     = H_SCALE / T_SCALE     # 1000.0 m/s
    DV_SCALE_RATE     = V_SCALE / T_SCALE     # 10.0 m/s^2
    DANGLE_SCALE_RATE = 1.0 / T_SCALE         # 0.01 rad/s

    # --- 2. Initial Setup and Scaled Inputs ---
    user_options = ()
    integration_rule = "trapezoidal"
    h, ϕ, θ, v, γ, ψ, q = integrator.u # Current unscaled state
    h_s = h / H_SCALE
    ϕ_s = ϕ
    θ_s = θ
    v_s = v / V_SCALE
    γ_s = γ
    ψ_s = ψ
    q_s = q
    cD = integrator.p.Cd
    cL = integrator.p.Cl
    # Get parameters
    m = integrator.p.mass
    Rₑ = integrator.p.R
    μ = integrator.p.μ
    S = integrator.p.area

    a₀ = -0.20704
    a₁ = 0.029244
    b₀ = 0.07854
    b₁ = -0.61592e-2
    b₂ = 0.621408e-3
    
    # Scale physical constants to match H_SCALE
    R_E_SCALED = Rₑ / H_SCALE
    
    update_step!(integrator)
    
    # Atmosphere and Horizon
    C1 = 8.53e-13 # Constant for convective heat rate calculation
    n_exp = integrator.p.mpc_params.n_exp # Exponent for convective heat rate calculation
    m_exp = integrator.p.mpc_params.m_exp # Exponent for convective heat rate calculation

    n = integrator.p.mpc_params.n_horizon         # Prediction horizon steps
    time_step = integrator.p.mpc_params.time_step # seconds

    # Scale target states
    optimal_states = integrator.p.nominal_trajectory
    trajectory_times = integrator.t * ones(n) .+ cumsum(value.(time_step * ones(n)))
    # println("Trajectory times: ", trajectory_times)
    nominal_alts = optimal_states[1](trajectory_times)
    nominal_lons = optimal_states[2](trajectory_times)
    nominal_lats = optimal_states[3](trajectory_times)
    nominal_vels = optimal_states[4](trajectory_times)
    nominal_γs = optimal_states[5](trajectory_times)
    nominal_azimuths = optimal_states[6](trajectory_times)
    h_t = nominal_alts[end] / H_SCALE
    v_t = nominal_vels[end] / V_SCALE
    γ_t = nominal_γs[end]
    α_s = integrator.p.α
    β_s = integrator.p.β
    # --- 3. Define Scaled JuMP Variables ---
    model = Model(optimizer_with_attributes(Ipopt.Optimizer, user_options...))
    @operator(model, earth_density_op, 1, earth_atmosphere_density)
    
    @variables(model, begin
        0 ≤ scaled_h[1:n]                # altitude (ft) / 1e5
        ϕ[1:n]                # longitude (rad)
        deg2rad(-89) ≤ θ[1:n] ≤ deg2rad(89)  # latitude (rad)
        1e-4 ≤ scaled_v[1:n]                # velocity (ft/sec) / 1e4
        deg2rad(-89) ≤ γ[1:n] ≤ deg2rad(89)  # flight path angle (rad)
        ψ[1:n]                # azimuth (rad)
        deg2rad(-90) ≤ α[1:n] ≤ deg2rad(90)  # angle of attack (rad)
        deg2rad(-89) ≤ β[1:n] ≤ deg2rad(89)  # bank angle (rad)
        # 0.1 ≤       Δt[1:n] ≤ 1.0          # time step (sec)
        # 0.0 <= q_dot[1:n] <= 269.0               # heat rate (W/m^2)
        0.0 <= q[1:n] <= 6200.0                  # heat load (J/m^2)
        # Δt[1:n] == 4.0         # time step (sec)
    end);

    # Fix initial conditions
    fix(scaled_h[1], h_s; force = true)
    fix(ϕ[1], ϕ_s; force = true)
    fix(θ[1], θ_s; force = true)
    fix(scaled_v[1], v_s; force = true)
    fix(γ[1], γ_s; force = true)
    fix(ψ[1], ψ_s; force = true)
    fix(α[1], α_s; force = true)
    fix(β[1], β_s; force = true)
    # fix(q_dot[1], 0.0; force = true)
    fix(q[1], q_s; force = true)

    # Fix final conditions
    # fix(scaled_h[n], h_t; force = true)
    # fix(scaled_v[n], v_t; force = true)
    # fix(γ[n], γ_t; force = true)
    # fix(θ[n], deg2rad(-4.5); force = true)  # Target latitude in radians
    # fix(ϕ[n], deg2rad(137.4); force = true)  # Target longitude in radians

    # Initial guess: linear interpolation between boundary conditions
    x_s = [h_s, ϕ_s, θ_s, v_s, γ_s, ψ_s, α_s, β_s, q_s]
    x_t = [h_t, ϕ_s, θ_s, v_t, γ_t, ψ_s, α_s, β_s, q_s]
    interp_linear = Interpolations.LinearInterpolation([1, n], [x_s, x_t])
    initial_guess = mapreduce(transpose, vcat, interp_linear.(1:n))
    set_start_value.(all_variables(model), vec(initial_guess))

    # Functions to restore `h` and `v` to their true scale
    @expression(model, h[j=1:n], scaled_h[j] * H_SCALE)
    @expression(model, v[j=1:n], scaled_v[j] * V_SCALE)

    # Helper functions
    @expression(model, cL[j=1:n], a₀ + a₁ * rad2deg(α[j]))
    @expression(model, cD[j=1:n], b₀ + b₁ * rad2deg(α[j]) + b₂ * rad2deg(α[j])^2)
    @expression(model, ρ[j=1:n], earth_density_op(h[j]))
    @expression(model, D[j=1:n], 0.5 * cD[j] * S * ρ[j] * v[j]^2)
    @expression(model, L[j=1:n], 0.5 * cL[j] * S * ρ[j] * v[j]^2)
    @expression(model, r[j=1:n], Rₑ + h[j])
    @expression(model, g[j=1:n], μ / r[j]^2)
    @expression(model, q_dot[j=1:n], C1 * ρ[j]^n_exp * v[j]^m_exp)

    # Motion of the vehicle as a differential-algebraic system of equations (DAEs)
    n_states_plus_control = length(integrator.u) + 1
    n_rf = size(integrator.p.mpc_params.alpha)[2]
    ω = integrator.p.mpc_params.omega
    b = integrator.p.mpc_params.b
    # Z = SVector{n_states + 1, Float64}(integrator.u..., integrator.p.β)...)
    # wz_plus_b::SVector{n_rf, Float64} = ω * Z + b
    # rf_numeric = 1 / √(n_rf) * cos(wz_plus_b)
    alpha_in = integrator.p.mpc_params.alpha
    # alpha_in = ones(size(alpha_in)) * 0.01 # Zero out disturbance for testing
    
    println("Time: $(integrator.t), disturbance: ", alpha_in * (1 / √(n_rf) * cos.(ω * SVector{n_states_plus_control, Float64}(integrator.u..., integrator.p.β) + b)))
    # println("q_dot: ", q_dot)
    # println("h: ", h_s)
    # println("v: ", v_s)
    # println("γ: ", γ_s)
    # println("ψ: ", ψ_s)
    @expression(model, Z[j=1:n], [h[j], ϕ[j], θ[j], v[j], γ[j], ψ[j], q[j], β[j]])
    @expression(model, wz_plus_b[j=1:n], ω * Z[j] + b)
    @expression(model, rf_numeric[j=1:n], 1 / √(n_rf) * cos.(wz_plus_b[j]))
    @expression(model, disturbance[j=1:n, k=1:n_states_plus_control-1], (alpha_in * rf_numeric[j])[k])
    @expression(model, δh[j=1:n], v[j] * sin(γ[j]) + disturbance[j,1])
    @expression(model, δϕ[j=1:n], (v[j] / r[j]) * cos(γ[j]) * sin(ψ[j]) / cos(θ[j]) + disturbance[j,2])
    @expression(model, δθ[j=1:n], (v[j] / r[j]) * cos(γ[j]) * cos(ψ[j]) + disturbance[j,3])
    @expression(model, δv[j=1:n], -(D[j] / m) - g[j] * sin(γ[j]) + disturbance[j,4])
    @expression(
        model,
        δγ[j=1:n],
        (L[j] / (m * v[j])) * cos(β[j]) +
        cos(γ[j]) * ((v[j] / r[j]) - (g[j] / v[j])) + 
        disturbance[j,5]
    )
    @expression(
        model,
        δψ[j=1:n],
        (1 / (m * v[j] * cos(γ[j]))) * L[j] * sin(β[j]) +
        (v[j] / (r[j] * cos(θ[j]))) * cos(γ[j]) * sin(ψ[j]) * sin(θ[j]) + 
        disturbance[j,6]
    )

    # Dynamics constraints
    for j in 2:n
        i = j - 1  # index of previous knot

        if integration_rule == "rectangular"
            # Rectangular integration
            @constraint(model, h[j] == h[i] + time_step * δh[i])
            @constraint(model, ϕ[j] == ϕ[i] + time_step * δϕ[i])
            @constraint(model, θ[j] == θ[i] + time_step * δθ[i])
            @constraint(model, v[j] == v[i] + time_step * δv[i])
            @constraint(model, γ[j] == γ[i] + time_step * δγ[i])
            @constraint(model, ψ[j] == ψ[i] + time_step * δψ[i])
        elseif integration_rule == "trapezoidal"
            # Trapezoidal integration
            @constraint(model, h[j] == h[i] + 0.5 * time_step * (δh[j] + δh[i]))
            @constraint(model, ϕ[j] == ϕ[i] + 0.5 * time_step * (δϕ[j] + δϕ[i]))
            @constraint(model, θ[j] == θ[i] + 0.5 * time_step * (δθ[j] + δθ[i]))
            @constraint(model, v[j] == v[i] + 0.5 * time_step * (δv[j] + δv[i]))
            @constraint(model, γ[j] == γ[i] + 0.5 * time_step * (δγ[j] + δγ[i]))
            @constraint(model, ψ[j] == ψ[i] + 0.5 * time_step * (δψ[j] + δψ[i]))
        else
            @error "Unexpected integration rule '$(integration_rule)'"
        end
        @constraint(model, q[j] == q[i] + q_dot[j]*time_step)
    end

    @constraint(model, [j=2:n], α[j] - α[j - 1] <= _CONTROL_RATE_LIMIT_RAD_PER_SEC[1] * time_step)
    @constraint(model, [j=2:n], α[j - 1] - α[j] <= _CONTROL_RATE_LIMIT_RAD_PER_SEC[1] * time_step)
    @constraint(model, [j=2:n], β[j] - β[j - 1] <= _CONTROL_RATE_LIMIT_RAD_PER_SEC[2] * time_step)
    @constraint(model, [j=2:n], β[j - 1] - β[j] <= _CONTROL_RATE_LIMIT_RAD_PER_SEC[2] * time_step)

    # Heating constraints
    # Objective: Reach target latitude and longitude
    @constraint(model, [j=1:n], q_dot[j] <= 269.0)  # Max heat rate (W/m^2)
    @expression(model, alt_cost, (nominal_alts .- h[1:n]) / H_SCALE)
    @expression(model, lon_cost, nominal_lons .- ϕ[1:n])
    @expression(model, lat_cost, nominal_lats .- θ[1:n])
    @expression(model, vel_cost, (nominal_vels .- v[1:n]) / V_SCALE)
    @expression(model, γ_cost, nominal_γs .- γ[1:n])
    @expression(model, azimuth_cost, nominal_azimuths .- ψ[1:n])
    @objective(model, Min, sum(1.0 * alt_cost.^2 + 1.0 * lon_cost.^2 + 10000.0 * lat_cost.^2 + vel_cost.^2 + γ_cost.^2 + azimuth_cost.^2) + sum(0.001*β[2:n].^2 + 0.1*α[2:n].^2))
    # target_latitude = deg2rad(-4.5)  # Target latitude in radians
    # @constraint(model,target_latitude - deg2rad(0.1) <= θ[n] <= target_latitude + deg2rad(0.1))
    # @expression(model, latitude_error, θ[n] - target_latitude)
    # target_longitude = deg2rad(137.4)  # Target longitude in radians
    # @constraint(model,target_longitude - deg2rad(0.1) <= ϕ[n] <= target_longitude + deg2rad(0.1))
    # @constraint(model, γ[n] >= deg2rad(-6.0))
    # @expression(model, longitude_error, ϕ[n] - target_longitude)
    # @expression(model, altitude_error, h[n] / 1e5 - h_t)  # Target altitude in meters
    # @expression(model, velocity_error, v[n] / 1e4 - v_t) # Target velocity in m/s
    # @expression(model, flight_path_angle_error, γ[n] - γ_t) # Target flight path angle in radians
    # @objective(model, Min, sum(Δt))

    set_silent(model)  # Hide solver's verbose output
    # set_attribute(model, "tol", 1e-6)  # Set solver tolerance
    optimize!(model)  # Solve for the control and state
    # assert_is_solved_and_feasible(model)

    # --- 9. Extract and Return Unscaled Control ---
    u_opt = _rate_limited_control_from_integrator(integrator, [value.(α)[2], value.(β)[2]])
    α_opt = u_opt[1]
    β_opt = u_opt[2]

    # Save to integrator parameters for logging
    integrator.p.optimization_states = OptimizationStates(
        h_c = value.(h),
        ϕ_c = value.(ϕ),
        θ_c = value.(θ),
        v_c = value.(v),
        γ_c = value.(γ),
        ψ_c = value.(ψ),
        β_c = value.(β),
        # Δt_c = value.(Δt) * T_SCALE
    )

    return β_opt, α_opt
end

function update_step!(integrator)
    # Placeholder for SSI-MPC step update
    n_rf::Int64 = size(integrator.p.mpc_params.alpha)[2]
    n_states_plus_control::Int64 = length(integrator.u) + 1
    dt::Float64 = 1.0
    ω::MMatrix{n_rf, n_states_plus_control, Float64} = integrator.p.mpc_params.omega
    b::MVector{n_rf, Float64} = integrator.p.mpc_params.b

    if integrator.p.mpc_params.prev_x == MVector{n_states_plus_control, Float64}(zeros(n_states_plus_control))
        integrator.p.mpc_params.prev_x .= MVector{n_states_plus_control, Float64}(integrator.u..., 0.0)
    end

    if integrator.dt == 0.0
        dt = integrator.p.mpc_params.time_step
    end

    alpha_in = integrator.p.mpc_params.alpha
    x_in = MVector{7, Float64}(integrator.p.mpc_params.prev_x[1:end-1])
    u_in = integrator.p.mpc_params.prev_x[end]

    Z = SVector{n_states_plus_control, Float64}(integrator.p.mpc_params.prev_x...)
    wz_plus_b::SVector{n_rf, Float64} = ω * Z + b
    rf_numeric = 1 / √(n_rf) * cos.(wz_plus_b)

    alpha_zero = SMatrix{n_states_plus_control, n_rf, Float64}(zeros(n_states_plus_control, n_rf))
    x_dot_nominal = MVector{n_states_plus_control-1, Float64}(zeros(n_states_plus_control-1))
    edl_dynamics_ssi_mpc!(x_dot_nominal, x_in, integrator.p, integrator.t, integrator.p.β)
    x_pred_nominal = x_in + x_dot_nominal * dt
    h_meas = integrator.u - x_pred_nominal
    # println("Shape of α_in: ", size(alpha_in), " Shape of rf_numeric: ", size(rf_numeric), " dot product size: ", size(alpha_in * rf_numeric))
    pred_target = alpha_in * rf_numeric

    ∇A = -2.0 * (h_meas - pred_target) * rf_numeric'
    alpha_out = alpha_in - integrator.p.mpc_params.learning_rate * ∇A

    if length(integrator.p.mpc_params.prev_alphas) < 50
        push!(integrator.p.mpc_params.prev_alphas, alpha_out)
    else
        stds = dropdims(std(stack(integrator.p.mpc_params.prev_alphas); dims=3); dims=3) # find the std dev across previous alphas
        means = dropdims(mean(stack(integrator.p.mpc_params.prev_alphas); dims=3); dims=3) # find the mean across previous alphas
        alpha_out .= clamp.(alpha_out, means .- 3.0 .* stds, means .+ 3.0 .* stds)
        push!(integrator.p.mpc_params.prev_alphas, alpha_out)
        popfirst!(integrator.p.mpc_params.prev_alphas)
    end

    integrator.p.mpc_params.alpha .= alpha_out
    integrator.p.mpc_params.prev_x .= [integrator.u..., integrator.p.β]
    # integrator.p.mpc_params.alpha .= zeros(size(integrator.p.mpc_params.alpha))
    # integrator.p.prev_u = integrator.p.β
end
