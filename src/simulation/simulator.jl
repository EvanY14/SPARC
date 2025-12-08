using LinearAlgebra
using DifferentialEquations
using StaticArrays

function edl_dynamics(du::MVector{7, Float64}, u::MVector{7, Float64}, p::EDLParams, t::Float64)
    h, ϕ, θ, v, γ, ψ = u
    m = p.mass
    Cd = p.Cd
    Cl = p.Cl
    A = p.area
    μ = p.μ
    R = p.R # Planetary radius
    β = p.β # Bank angle
    r = R + h # Distance from planet center

    # calculate wind-relative velocity
    wind = p.wind
    v_vector = SVector{3, Float64}(
        v * cos(γ) * cos(ψ),
        v * cos(γ) * sin(ψ),
        v * sin(γ)
    )
    v_rel_vector = v_vector - wind
    v_rel = norm(v_rel_vector)
    drag = 0.5 * p.atmospheric_density * v_rel^2 * Cd * A # Drag force
    lift = 0.5 * p.atmospheric_density * v_rel^2 * Cl * A # Lift force

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
    heat_rate = 0.5 * p.atmospheric_density * v_rel^3 # Convective heat rate (W/m^2)

    du[1] = (v * sin_γ) #  h_dot
    du[2] = (v/r) * cos_γ * sin_ψ / cos_θ # ϕ_dot (longitude)
    du[3] = (v/r) * cos_γ * cos_ψ # θ_dot (latitude)
    du[4] = (-drag / m - g * sin_γ) # v_dot
    du[5] = (lift/(m*v) * cos_β) + cos_γ*(v/r - g/v) # γ_dot (flight path angle)
    du[6] = (lift * sin_β) / (m * v * cos_γ) + (v * cos_γ * sin_ψ * tan_θ) / r # ψ_dot (azimuth)
    du[7] = heat_rate # heat rate (W/m^2)
end