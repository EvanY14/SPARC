function altitude_condition(u, t, integrator)
    h = u[1]
    return h - integrator.p.target_altitude
end

function altitude_effect!(integrator)
    terminate!(integrator)
end

altitude_termination_condition = ContinuousCallback(altitude_condition, altitude_effect!)

function atmospheric_density_effect!(integrator)
    # println(typeof(integrator))
    density_function = integrator.p.atmospheric_density_function
    h = integrator.u[1]
    integrator.p.atmospheric_density = density_function(integrator)
end

atmospheric_density_callback = DiscreteCallback((u, t, integrator) -> true, atmospheric_density_effect!)

SavedValueType = Tuple{Float64, Float64}
saved_values = SavedValues(Float64, SavedValueType)

# 3. DEFINE THE "SAVE" FUNCTION
#    It just reads the values we already calculated. No extra work!
function save_func(u, t, integrator)
    density = integrator.p.atmospheric_density
    β = integrator.p.β
    return (density, β)
end

# 4. BUILD THE CALLBACK
#    save_everystep=true is the default, but good to be explicit
saving_callback = SavingCallback(save_func, saved_values, save_everystep=true)