using LinearAlgebra
using DifferentialEquations
using StaticArrays

function edl_dynamics(du::MVector{6, Float64}, u::MVector{6, Float64}, p::EDLParams, t::Float64)
    h, ϕ, θ, v, γ, ψ = u
    m = p.mass
    Cd = p.Cd
    Cl = p.Cl
    A = p.area
    μ = p.μ
    R = p.R # Planetary radius
    β = p.β # Bank angle
    r = R + h # Distance from planet center

    drag = 0.5 * p.atmospheric_density(h) * v^2 * Cd * A # Drag force
    lift = 0.5 * p.atmospheric_density(h) * v^2 * Cl * A # Lift force

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

    du[1] = (v * sin_γ) #  h_dot
    du[2] = (v/r) * cos_γ * sin_ψ / cos_θ # ϕ_dot (longitude)
    du[3] = (v/r) * cos_γ * cos_ψ # θ_dot (latitude)
    du[4] = (-drag / m - g * sin_γ) # v_dot
    du[5] = (lift/(m*v) * cos_β) + cos_γ*(v/r - g/v) # γ_dot (flight path angle)
    du[6] = (lift * sin_β) / (m * v * cos_γ) + (v * cos_γ * sin_ψ * tan_θ) / r # ψ_dot (azimuth)
end


function altitude_condition(u, t, integrator)
    h = u[1] # Unnormalize altitude
    return h - integrator.p.target_altitude
end

function altitude_effect!(integrator)
    terminate!(integrator)
end

altitude_termination_condition = ContinuousCallback(altitude_condition, altitude_effect!)

function atmospheric_density(h::Float64, atmosphere_model::PolyfitAtmosphere)
    polyfit_coefficients = atmosphere_model.polyfit_coefficients
    power = zeros(length(polyfit_coefficients))
    # Convert height from meters to kilometers
    h = h * 1e-3
    # Calculate the polynomial value at height h
    for i=1:length(polyfit_coefficients)
        power[i] = (h)^(length(polyfit_coefficients)-i)
    end
    # Calculate the exponent term of the density using the polynomial coefficients
    exponent = sum(polyfit_coefficients .* power)
    # Calculate the density
    ρ = exp(exponent)
    return ρ
end

function atmospheric_density(h::Float64, atmosphere_model::ExponentialAtmosphere)
    return atmosphere_model.surface_density * exp(-h*1e-3/atmosphere_model.scale_height)
end