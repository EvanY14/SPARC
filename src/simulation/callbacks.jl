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
    if !isfinite(h) || integrator.p.R + h <= 0.0
        @warn "Terminating simulation before atmospheric density lookup because altitude is invalid" altitude=h time=integrator.t
        terminate!(integrator)
        return nothing
    end
    if h <= integrator.p.target_states.altitude
        terminate!(integrator)
        return nothing
    end
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

function _stores_two_control_prev_x(control_function)
    return control_function === trackingmpc ||
           control_function === trackingmpc_shrinking_horizon ||
           control_function === trackingmpc_shrinking ||
           control_function === model_predictive_guidance ||
           control_function === mpg ||
           control_function === mpg_integral_tracking ||
           control_function === mpg_integral ||
           control_function === sm_mpg ||
           control_function === sm_mpg_q4_tracking ||
           control_function === sm_mpg_q4 ||
           control_function === sm_mpg_integral_tracking ||
           control_function === sm_mpg_integral
end

function _sync_applied_control_to_prev_x!(integrator, α_limited, β_limited)
    if _stores_two_control_prev_x(integrator.p.control_function)
        prev_x = integrator.p.mpc_params.prev_x
        if length(prev_x) >= 8
            prev_x[7:8] .= [α_limited, β_limited]
        end
    end
    return nothing
end

function control_callback_effect!(integrator)
    β_cmd, α_cmd = Base.invokelatest(integrator.p.control_function, integrator)
    α_limited, β_limited = _rate_limited_control_from_integrator(integrator, [α_cmd, β_cmd])
    integrator.p.β = β_limited
    integrator.p.α = α_limited
    _sync_applied_control_to_prev_x!(integrator, α_limited, β_limited)
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
