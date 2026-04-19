function altitude_condition(u, t, integrator)
    h = u[1]
    target_altitude = integrator.p.target_states.altitude
    return h - target_altitude
end

function altitude_effect!(integrator)
    terminate!(integrator)
end

altitude_termination_condition = ContinuousCallback(altitude_condition, altitude_effect!)

function atmospheric_density_effect!(integrator)
    # println(typeof(integrator))
    density_function = integrator.p.atmospheric_density_function
    h = integrator.u[1]
    LatLonAlt = (integrator.u[3], integrator.u[2], h)
    integrator.p.atmospheric_density, integrator.p.wind = density_function(LatLonAlt, integrator.t)
end

atmospheric_density_callback = DiscreteCallback((u, t, integrator) -> true, atmospheric_density_effect!)

SavedValueType = Tuple{Float64, Float64, Float64, Float64}  # (density, bank angle, heat rate)
saved_values = SavedValues(Float64, SavedValueType)

# 3. DEFINE THE "SAVE" FUNCTION
#    It just reads the values we already calculated. No extra work!
function save_func(u, t, integrator)
    density = integrator.p.atmospheric_density
    β = integrator.p.β
    α = integrator.p.α
    heat_rate = integrator.p.cache.q_dot
    return (density, β, α, heat_rate)
end

# 4. BUILD THE CALLBACK
#    save_everystep=true is the default, but good to be explicit
saving_callback = SavingCallback(save_func, saved_values, saveat=0.1)

function control_callback_effect!(integrator)
    integrator.p.β, integrator.p.α = Base.invokelatest(integrator.p.control_function, integrator)
    integrator.p.cache.last_control_update = integrator.t
end

function control_callback_condition(u, t, integrator)
    dt = integrator.p.mpc_params.time_step
    if dt <= 0.0
        return true
    end
    last_update = integrator.p.cache.last_control_update
    return !isfinite(last_update) || t - last_update >= dt - 10 * eps(max(abs(t), abs(last_update), 1.0))
end

control_callback = DiscreteCallback(control_callback_condition, control_callback_effect!)
