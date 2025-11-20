# function mpc(integrator)
#     # Scaling constants
#     const H_SCALE = 1e5  # m
#     const V_SCALE = 1e3  # m/s
#     const T_SCALE = 100.0 # s (Choose 100 or 250s, whatever is appropriate for your total time)
#     const R_E_SCALED = Rₑ / H_SCALE # 63.78 

#     user_options = ()
#     model = Model(optimizer_with_attributes(Ipopt.Optimizer, user_options...))
#     h, ϕ, θ, v, γ, ψ = integrator.u
#     # Normalize h and v for numerical stability
#     h /= H_SCALE
#     v /= V_SCALE

#     # Get parameters
#     m = integrator.p.mass
#     target_states = integrator.p.target_states
#     target_altitude = target_states.altitude / H_SCALE # Scale altitude for numerical stability
#     target_longitude = target_states.longitude
#     target_latitude = target_states.latitude
#     target_velocity = target_states.velocity / V_SCALE # Scale velocity for numerical stability
#     target_γ = target_states.flight_path_angle
    
#     # Build atmosphere model
#     polyfit_coeffs = Float64[2.484093267854419e-35, -3.432059129183589e-32, 2.0998712380197567e-29, -7.374629031680772e-27, 1.5792723271745155e-24, -1.8603802534535614e-22, 1.1824450144926489e-21, 3.944724626716538e-18, -8.193458848294376e-16, 9.735891059182661e-14, -7.897816207129188e-12, 4.5807414555856416e-10, -1.9161056559474318e-08, 5.713547101023083e-07, -1.1780507222866087e-05, 0.00015839694888627217, -0.0012270664089332438, 0.0035825645308133545, 0.012231321466518718, -0.1691661107577747, -4.32384932627002]
#     polyfit_atmosphere = PolyfitAtmosphere{length(polyfit_coeffs)}(SVector{length(polyfit_coeffs), Float64}(polyfit_coeffs))
#     n = 500  # Prediction horizon steps
#     time_step = 0.5  # seconds

#     # Variables are defined in scaled units (e.g., h_c[j] is the altitude in 100km units)
#     @variables(model, begin
#         0 ≤ h_c[1:n]                   # scaled altitude (e.g., 0.0 to 1.5)
#         ϕ_c[1:n]
#         deg2rad(-89) ≤ θ_c[1:n] ≤ deg2rad(89)
#         1e-4 / V_SCALE ≤ v_c[1:n]     # scaled velocity (e.g., 0.1 to 10.0)
#         deg2rad(-89) ≤ γ_c[1:n] ≤ deg2rad(89)
#         ψ_c[1:n]
#         deg2rad(-89) ≤ β_c[1:n] ≤ deg2rad(89)  # bank angle (rad)
#         # Scale Δt by T_SCALE
#         (time_step*1e-9 / T_SCALE) <= Δt[1:n] <= (time_step*5.0 / T_SCALE)
#     end)

#     # Fix initial conditions using scaled inputs
#     fix(h_c[1], h / H_SCALE; force = true)
#     fix(v_c[1], v / V_SCALE; force = true)

#     # Fix final conditions using scaled targets
#     fix(h_c[n], target_altitude / H_SCALE; force = true)
#     fix(v_c[n], target_velocity / V_SCALE; force = true)
#     # @variables(model, begin
#     #     0 ≤ h_c[1:n]                # altitude (m)
#     #     ϕ_c[1:n]                # longitude (rad)
#     #     deg2rad(-89) ≤ θ_c[1:n] ≤ deg2rad(89)  # latitude (rad)
#     #     1e-4 ≤ v_c[1:n]                # velocity (ft/sec) / 1e4
#     #     deg2rad(-89) ≤ γ_c[1:n] ≤ deg2rad(89)  # flight path angle (rad)
#     #     ψ_c[1:n]                # azimuth (rad)
#     #     # deg2rad(-90) ≤ α[1:n] ≤ deg2rad(90)  # angle of attack (rad)
#     #     deg2rad(-89) ≤ β_c[1:n] ≤ deg2rad(89)  # bank angle (rad)
#     #     #        3.5 ≤       Δt[1:n] ≤ 4.5          # time step (sec)
#     #     time_step*1e-9 <= Δt[1:n] <= time_step*5.0        # time step (sec)
#     # end)

#     # Fix initial conditions
#     # fix(h_c[1], h; force = true)
#     fix(ϕ_c[1], ϕ; force = true)
#     fix(θ_c[1], θ; force = true)
#     # fix(v_c[1], v; force = true)
#     fix(γ_c[1], γ; force = true)
#     fix(ψ_c[1], ψ; force = true)

#     # Fix final conditions
#     # fix(h_c[n], target_altitude; force = true)
#     # fix(v_c[n], target_velocity; force = true)
#     # fix(γ_c[n], target_γ; force = true)

#     β = integrator.p.β # Current bank angle
#     # Initial guess: linear interpolation between boundary conditions
#     x_s = [h, ϕ, θ, v, γ, ψ, β, 0.0]
#     x_t = [target_altitude, target_longitude, target_latitude, target_velocity, target_γ, ψ, 0.0, n*time_step]
#     interp_linear = Interpolations.LinearInterpolation([1, n], [x_s, x_t])
#     initial_guess = mapreduce(transpose, vcat, interp_linear.(1:n))
#     set_start_value.(all_variables(model), vec(initial_guess))

#     # Helper functions
#     density_function = integrator.p.atmospheric_density_function
#     μ = integrator.p.μ / (H_SCALE * V_SCALE^2)  # Scaled gravitational parameter
#     Rₑ = integrator.p.R / H_SCALE
#     S = integrator.p.area 
#     @expression(model, c_D[j=1:n], integrator.p.Cd)
#     @expression(model, c_L[j=1:n], integrator.p.Cl)
#     # Dynamics constraints
#     # Unscaled state variables from scaled JuMP variables
#     @expression(model, h_unscaled[j=1:n], h_c[j] * H_SCALE)
#     @expression(model, v_unscaled[j=1:n], v_c[j] * V_SCALE)

#     # Recalculate physical expressions using unscaled values
#     @expression(model, ρ[j=1:n], atmospheric_density((θ_c[j], ϕ_c[j], h_unscaled[j]), (j-1)*time_step, polyfit_atmosphere)[1])

#     # Drag and Lift (D and L must be in Newtons)
#     @expression(model, D[j=1:n], 0.5 * c_D[j] * S * ρ[j] * v_unscaled[j]^2)
#     @expression(model, L[j=1:n], 0.5 * c_L[j] * S * ρ[j] * v_unscaled[j]^2)

#     # Radius and Gravity (r and g must be in unscaled units)
#     @expression(model, r[j=1:n], Rₑ + h_unscaled[j])
#     @expression(model, g[j=1:n], μ / r[j]^2)
#     # @expression(model, ρ[j=1:n], atmospheric_density((θ_c[j], ϕ_c[j], h_c[j]), (j-1)*time_step, polyfit_atmosphere)[1])
#     # @expression(model, D[j=1:n], 0.5 * c_D[j] * S * ρ[j] * v_c[j]^2)
#     # @expression(model, L[j=1:n], 0.5 * c_L[j] * S * ρ[j] * v_c[j]^2)
#     # @expression(model, r[j=1:n], Rₑ + h_c[j])
#     # @expression(model, g[j=1:n], μ / r[j]^2)

#     @expression(model, δh[j=1:n], v_unscaled[j] * sin(γ_c[j]))
#     @expression(model, δϕ[j=1:n], (v_unscaled[j] / r[j]) * cos(γ_c[j]) * sin(ψ_c[j]) / cos(θ_c[j]))
#     @expression(model, δθ[j=1:n], (v_unscaled[j] / r[j]) * cos(γ_c[j]) * cos(ψ_c[j]))
#     @expression(model, δv[j=1:n], -(D[j] / m) - g[j] * sin(γ_c[j]))
#     @expression(
#         model,
#         δγ[j=1:n],
#         (L[j] / (m * v_unscaled[j])) * cos(β_c[j]) +
#         cos(γ_c[j]) * ((v_unscaled[j] / r[j]) - (g[j] / v_unscaled[j]))
#     )
#     @expression(
#         model,
#         δψ[j=1:n],
#         (1 / (m * v_unscaled[j] * cos(γ_c[j]))) * L[j] * sin(β_c[j]) +
#         (v_unscaled[j] / (r[j] * cos(θ_c[j]))) * cos(γ_c[j]) * sin(ψ_c[j]) * sin(θ_c[j])
#     )

#     for j in 2:n
#         i = j - 1  # index of previous knot
#         # Trapezoidal integration
#         @constraint(model, h_c[j] == h_c[i] + 0.5 * Δt[i] * (δh[j] + δh[i]))
#         @constraint(model, ϕ_c[j] == ϕ_c[i] + 0.5 * Δt[i] * (δϕ[j] + δϕ[i]))
#         @constraint(model, θ_c[j] == θ_c[i] + 0.5 * Δt[i] * (δθ[j] + δθ[i]))
#         @constraint(model, v_c[j] == v_c[i] + 0.5 * Δt[i] * (δv[j] + δv[i]))
#         @constraint(model, γ_c[j] == γ_c[i] + 0.5 * Δt[i] * (δγ[j] + δγ[i]))
#         @constraint(model, ψ_c[j] == ψ_c[i] + 0.5 * Δt[i] * (δψ[j] + δψ[i]))
#     end

#     # Objective: minimize miss distance (lat, lon) at final time
#     final_latitude = θ_c[n]
#     final_longitude = ϕ_c[n]
#     @expression(model, lat_error, final_latitude - target_latitude)
#     @expression(model, lon_error, final_longitude - target_longitude)
#     @expression(model, miss_distance, lat_error^2 + lon_error^2)
#     @objective(model, Min, miss_distance)

#     # set_silent(model)  # Hide solver's verbose output
#     # set_attribute(model, "max_iter", 10) # Set solver tolerance
#     optimize!(model)  # Solve for the control and state

#     # Extract optimal control input at the next time step
#     β_opt = value.(β_c)[1]
#     return β_opt
# end

# using JuMP, Ipopt, StaticArrays, Interpolations

function mpc(integrator)
    # --- 1. Define Scaling Constants ---
    H_SCALE = 1e5   # Reference Altitude (100 km)
    V_SCALE = 1e3   # Reference Velocity (1 km/s)
    T_SCALE = 100.0 # Reference Time (100 s) 
    
    # Derived Scaled Rate Units (used for dynamics)
    DH_SCALE_RATE     = H_SCALE / T_SCALE     # 1000.0 m/s
    DV_SCALE_RATE     = V_SCALE / T_SCALE     # 10.0 m/s^2
    DANGLE_SCALE_RATE = 1.0 / T_SCALE         # 0.01 rad/s

    # --- 2. Initial Setup and Scaled Inputs ---
    user_options = ()
    model = Model(optimizer_with_attributes(Ipopt.Optimizer, user_options...))
    h, ϕ, θ, v, γ, ψ = integrator.u # Current unscaled state

    # Get parameters
    m = integrator.p.mass
    Rₑ = integrator.p.R
    μ = integrator.p.μ
    S = integrator.p.area
    
    # Scale physical constants to match H_SCALE
    R_E_SCALED = Rₑ / H_SCALE
    
    # Scale target states
    target_states = integrator.p.target_states
    target_altitude = target_states.altitude / H_SCALE
    target_velocity = target_states.velocity / V_SCALE
    target_longitude = target_states.longitude
    target_latitude = target_states.latitude
    target_γ = target_states.flight_path_angle

    # Atmosphere and Horizon
    polyfit_coeffs = SVector{21, Float64}(Float64[2.484093267854419e-35, -3.432059129183589e-32, 2.0998712380197567e-29, -7.374629031680772e-27, 1.5792723271745155e-24, -1.8603802534535614e-22, 1.1824450144926489e-21, 3.944724626716538e-18, -8.193458848294376e-16, 9.735891059182661e-14, -7.897816207129188e-12, 4.5807414555856416e-10, -1.9161056559474318e-08, 5.713547101023083e-07, -1.1780507222866087e-05, 0.00015839694888627217, -0.0012270664089332438, 0.0035825645308133545, 0.012231321466518718, -0.1691661107577747, -4.32384932627002])
    polyfit_atmosphere = PolyfitAtmosphere{length(polyfit_coeffs)}(polyfit_coeffs)
    n = 200         # Prediction horizon steps
    time_step = 0.5 # seconds
    # --- 3. Define Scaled JuMP Variables ---
    @variables(model, begin
    # h_c is scaled by H_SCALE (1e5)
    0 ≤ h_c[1:n] ≤ 1.5                # scaled altitude (~0 to 150 km)
    ϕ_c[1:n]
    deg2rad(-89) ≤ θ_c[1:n] ≤ deg2rad(89)
    # v_c is scaled by V_SCALE (1e3)
    (1e-4 / V_SCALE) ≤ v_c[1:n] ≤ 10.0   # scaled velocity (~0 to 10 km/s)
    deg2rad(-89) ≤ γ_c[1:n] ≤ deg2rad(89)
    ψ_c[1:n]
    # Δt is scaled by T_SCALE (100 s)
    (time_step*1e-9 / T_SCALE) <= Δt[1:n] <= (time_step*5.0 / T_SCALE)
    deg2rad(-89) ≤ β_c[1:n] ≤ deg2rad(89) # bank angle (unscaled)
end)
    if integrator.p.optimization_states.h_c != zeros(0)
        optim_states = integrator.p.optimization_states
        set_start_value.(h_c, optim_states.h_c)
        set_start_value.(ϕ_c, optim_states.ϕ_c)
        set_start_value.(θ_c, optim_states.θ_c)
        set_start_value.(v_c, optim_states.v_c)
        set_start_value.(γ_c, optim_states.γ_c)
        set_start_value.(ψ_c, optim_states.ψ_c)
        set_start_value.(β_c, optim_states.β_c)
        set_start_value.(Δt, optim_states.Δt_c)
        set_attribute(model, "warm_start_init_point", "yes")
    end
    # Fix initial conditions using scaled current state
    fix(h_c[1], h / H_SCALE; force = true)
    fix(ϕ_c[1], ϕ; force = true)
    fix(θ_c[1], θ; force = true)
    fix(v_c[1], v / V_SCALE; force = true)
    fix(γ_c[1], γ; force = true)
    fix(ψ_c[1], ψ; force = true)

    # Fix final conditions using scaled targets
    # fix(h_c[n], target_altitude; force = true)
    # fix(v_c[n], target_velocity; force = true)
    # fix(γ_c[n], target_γ; force = true)

    # Initial guess (Must use scaled state variables h/H_SCALE and v/V_SCALE)
    x_s_scaled = [h / H_SCALE, ϕ, θ, v / V_SCALE, γ, ψ, integrator.p.β, 0.0 / T_SCALE]
    x_t_scaled = [target_altitude, target_longitude, target_latitude, target_velocity, target_γ, ψ, 0.0, (n*time_step) / T_SCALE]
    interp_linear = Interpolations.LinearInterpolation([1, n], [x_s_scaled, x_t_scaled])
    
    # The initial guess must be a vector of all variables in the order they were declared
    initial_guess_vars = [h_c; ϕ_c; θ_c; v_c; γ_c; ψ_c; β_c; Δt]
    
    # Set the initial guess for state variables
    for j in 1:n
        start_vals = interp_linear(j)
        set_start_value(h_c[j], start_vals[1])
        set_start_value(ϕ_c[j], start_vals[2])
        set_start_value(θ_c[j], start_vals[3])
        set_start_value(v_c[j], start_vals[4])
        set_start_value(γ_c[j], start_vals[5])
        set_start_value(ψ_c[j], start_vals[6])
        set_start_value(β_c[j], start_vals[7])
    end
    # Set the initial guess for time steps (assumes constant time step)
    set_start_value.(Δt, time_step / T_SCALE)


    # --- 4. Define Unscaled State Expressions for Physics ---
    @expression(model, h_unscaled[j=1:n], h_c[j] * H_SCALE)
    @expression(model, v_unscaled[j=1:n], v_c[j] * V_SCALE)
    
    # Expressions (use unscaled h and v for physics)
    @expression(model, c_D[j=1:n], integrator.p.Cd)
    @expression(model, c_L[j=1:n], integrator.p.Cl)
    @expression(model, ρ[j=1:n], atmospheric_density((θ_c[j], ϕ_c[j], h_unscaled[j]), (j-1)*time_step, polyfit_atmosphere)[1])
    
    # D and L are in Newtons (unscaled)
    @expression(model, D[j=1:n], 0.5 * c_D[j] * S * ρ[j] * v_unscaled[j]^2)
    @expression(model, L[j=1:n], 0.5 * c_L[j] * S * ρ[j] * v_unscaled[j]^2)
    
    # r and g are unscaled
    @expression(model, r[j=1:n], Rₑ + h_unscaled[j])
    @expression(model, g[j=1:n], μ / r[j]^2)
    
    # --- 5. Define Scaled Dynamics (RHS) ---
    # The RHS (rate of change) expressions are divided by their corresponding scaled rate units
    
    # δh (m/s) -> Scaled by DH_SCALE_RATE (1000 m/s)
    @expression(model, δh_scaled[j=1:n], (v_unscaled[j] * sin(γ_c[j])) / DH_SCALE_RATE)
    
    # δϕ (rad/s) -> Scaled by DANGLE_SCALE_RATE (0.01 rad/s)
    @expression(model, δϕ_scaled[j=1:n], ((v_unscaled[j] / r[j]) * cos(γ_c[j]) * sin(ψ_c[j]) / cos(θ_c[j])) / DANGLE_SCALE_RATE)
    
    # δθ (rad/s) -> Scaled by DANGLE_SCALE_RATE (0.01 rad/s)
    @expression(model, δθ_scaled[j=1:n], ((v_unscaled[j] / r[j]) * cos(γ_c[j]) * cos(ψ_c[j])) / DANGLE_SCALE_RATE)
    
    # δv (m/s^2) -> Scaled by DV_SCALE_RATE (10 m/s^2)
    @expression(model, δv_scaled[j=1:n], (-(D[j] / m) - g[j] * sin(γ_c[j])) / DV_SCALE_RATE)
    
    # δγ (rad/s) -> Scaled by DANGLE_SCALE_RATE (0.01 rad/s)
    @expression(
        model,
        δγ_scaled[j=1:n],
        (((L[j] / (m * v_unscaled[j])) * cos(β_c[j]) +
        cos(γ_c[j]) * ((v_unscaled[j] / r[j]) - (g[j] / v_unscaled[j])))) / DANGLE_SCALE_RATE
    )
    
    # δψ (rad/s) -> Scaled by DANGLE_SCALE_RATE (0.01 rad/s)
    @expression(
        model,
        δψ_scaled[j=1:n],
        ((1 / (m * v_unscaled[j] * cos(γ_c[j]))) * L[j] * sin(β_c[j]) +
        (v_unscaled[j] / (r[j] * cos(θ_c[j]))) * cos(γ_c[j]) * sin(ψ_c[j]) * sin(θ_c[j])) / DANGLE_SCALE_RATE
    )

    # --- 6. Scaled Trapezoidal Constraints ---
    for j in 2:n
        i = j - 1  # index of previous knot
        # The equation now correctly connects scaled state (LHS) with scaled rate (RHS)
        @constraint(model, h_c[j] == h_c[i] + 0.5 * Δt[i] * (δh_scaled[j] + δh_scaled[i]))
        @constraint(model, ϕ_c[j] == ϕ_c[i] + 0.5 * Δt[i] * (δϕ_scaled[j] + δϕ_scaled[i]))
        @constraint(model, θ_c[j] == θ_c[i] + 0.5 * Δt[i] * (δθ_scaled[j] + δθ_scaled[i]))
        @constraint(model, v_c[j] == v_c[i] + 0.5 * Δt[i] * (δv_scaled[j] + δv_scaled[i]))
        @constraint(model, γ_c[j] == γ_c[i] + 0.5 * Δt[i] * (δγ_scaled[j] + δγ_scaled[i]))
        @constraint(model, ψ_c[j] == ψ_c[i] + 0.5 * Δt[i] * (δψ_scaled[j] + δψ_scaled[i]))
    end

    # --- 7. Objective Function (Unscaled angles are okay) ---
    final_latitude = θ_c[n]
    final_longitude = ϕ_c[n]
    # final_altitude = h_c[n]
    @expression(model, lat_error, final_latitude - target_latitude)
    @expression(model, lon_error, final_longitude - target_longitude)
    # @expression(model, alt_error, final_altitude - target_altitude)
    @expression(model, miss_distance, lat_error^2 + lon_error^2)
    @objective(model, Min, miss_distance)

    # --- 8. Solve ---
    # set_silent(model)
    # set_attribute(model, "max_iter", 50) # Increased max_iter for better convergence
    optimize!(model)

    # --- 9. Extract and Return Unscaled Control ---
    β_opt = value.(β_c)[1]

    # Save to integrator parameters for logging
    integrator.p.optimization_states = OptimizationStates(
        h_c = value.(h_c) * H_SCALE,
        ϕ_c = value.(ϕ_c),
        θ_c = value.(θ_c),
        v_c = value.(v_c) * V_SCALE,
        γ_c = value.(γ_c),
        ψ_c = value.(ψ_c),
        β_c = value.(β_c),
        Δt_c = value.(Δt) * T_SCALE
    )
    return β_opt
end