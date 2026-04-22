const _CONTROL_MIN_RAD = deg2rad.([-90.0, -89.0])
const _CONTROL_MAX_RAD = deg2rad.([90.0, 89.0])
const _CONTROL_RATE_LIMIT_RAD_PER_SEC = deg2rad.([30.0, 20.0])

function _rate_limited_control(
    command::AbstractVector{<:Real},
    previous::AbstractVector{<:Real},
    dt::Real,
)
    u_cmd = clamp.(Float64.(command), _CONTROL_MIN_RAD, _CONTROL_MAX_RAD)
    if !(isfinite(dt) && dt > 0.0) || !all(isfinite, previous)
        return u_cmd
    end

    du_max = _CONTROL_RATE_LIMIT_RAD_PER_SEC .* Float64(dt)
    return clamp.(u_cmd, Float64.(previous) .- du_max, Float64.(previous) .+ du_max)
end

function _control_step_size(integrator)
    last_update = integrator.p.cache.last_control_update
    if isfinite(last_update)
        return Float64(integrator.t - last_update)
    end
    return Float64(integrator.p.mpc_params.time_step)
end

function _rate_limited_control_from_integrator(integrator, command::AbstractVector{<:Real})
    previous = [Float64(integrator.p.α), Float64(integrator.p.β)]
    return _rate_limited_control(command, previous, _control_step_size(integrator))
end

function _control_rate_bounds(previous::AbstractVector{<:Real}, step_size::Real)
    du_max = _CONTROL_RATE_LIMIT_RAD_PER_SEC .* Float64(step_size)
    lower = max.(_CONTROL_MIN_RAD, Float64.(previous) .- du_max)
    upper = min.(_CONTROL_MAX_RAD, Float64.(previous) .+ du_max)
    return lower, upper
end
