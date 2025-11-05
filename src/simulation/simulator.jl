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

    du[1] = v * sin_γ #  h_dot
    du[2] = (v * cos_γ * sin_ψ) / r / cos_θ # ϕ_dot
    du[3] = (v * cos_γ * cos_ψ) / r # θ_dot
    du[4] = -drag / m - μ * sin_γ # v_dot
    du[5] = (lift * cos_β) / (m * v) + cos_γ*(v/r - g/v) # γ_dot
    du[6] = (lift * sin_β) / (m * v * cos_γ) + (v * cos_γ * sin_ψ * tan_θ) / r # ψ_dot
end

function atmospheric_density(h::Float64, polyfit::PolyfitAtmosphere)
    polyfit_coefficients = polyfit.polyfit_coefficients
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