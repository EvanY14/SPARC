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

SavedValueType = Tuple{Float64, Float64, Float64}  # (density, bank angle, heat rate)
saved_values = SavedValues(Float64, SavedValueType)

# 3. DEFINE THE "SAVE" FUNCTION
#    It just reads the values we already calculated. No extra work!
function save_func(u, t, integrator)
    density = integrator.p.atmospheric_density
    β = integrator.p.β
    heat_rate = 0.5 * density * (u[4])^3
    return (density, β, heat_rate)
end

# 4. BUILD THE CALLBACK
#    save_everystep=true is the default, but good to be explicit
saving_callback = SavingCallback(save_func, saved_values, save_everystep=true)

function control_callback_effect!(integrator)
    integrator.p.β = integrator.p.control_function(integrator)
end

control_callback = PeriodicCallback(control_callback_effect!, 1.0) # Update every 1 second