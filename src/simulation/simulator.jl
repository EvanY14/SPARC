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
    p.β = p.β_function(u, p, t) # Bank angle
    r = R + h # Distance from planet center

    drag = 0.5 * p.atmospheric_density * v^2 * Cd * A # Drag force
    lift = 0.5 * p.atmospheric_density * v^2 * Cl * A # Lift force

    # Trig functions
    sin_γ = sin(γ)
    cos_γ = cos(γ)
    sin_ψ = sin(ψ)
    cos_ψ = cos(ψ)
    tan_θ = tan(θ)
    cos_θ = cos(θ)
    sin_β = sin(p.β)
    cos_β = cos(p.β)

    g = μ / r^2 # Gravitational acceleration

    du[1] = (v * sin_γ) #  h_dot
    du[2] = (v/r) * cos_γ * sin_ψ / cos_θ # ϕ_dot (longitude)
    du[3] = (v/r) * cos_γ * cos_ψ # θ_dot (latitude)
    du[4] = (-drag / m - g * sin_γ) # v_dot
    du[5] = (lift/(m*v) * cos_β) + cos_γ*(v/r - g/v) # γ_dot (flight path angle)
    du[6] = (lift * sin_β) / (m * v * cos_γ) + (v * cos_γ * sin_ψ * tan_θ) / r # ψ_dot (azimuth)
end