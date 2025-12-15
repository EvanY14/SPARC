using LinearAlgebra
using DifferentialEquations
using StaticArrays

function edl_dynamics(du::MVector{7, Float64}, u::MVector{7, Float64}, p::EDLParams, t::Float64)
    h, ϕ, θ, v, γ, ψ, q = u
    m = p.mass
    # Cd = p.Cd
    # Cl = p.Cl

    a₀ = -0.20704
    a₁ = 0.029244
    b₀ = 0.07854
    b₁ = -0.61592e-2
    b₂ = 0.621408e-3

    Cd = b₀ + b₁ * rad2deg(p.α) + b₂ * rad2deg(p.α)^2
    Cl = a₀ + a₁ * rad2deg(p.α)

    A = p.area
    μ = p.μ
    R = p.R # Planetary radius
    β = p.β # Bank angle
    r = R + h # Distance from planet center
    C1 = 8.53e-13 # Constant for convective heat rate calculation
    n = 0.82958 # Exponent for convective heat rate calculation
    m_exp = 4.512 # Exponent for convective heat rate calculation

    # calculate wind-relative velocity
    wind = p.wind
    v_vector = SVector{3, Float64}(
        v * cos(γ) * cos(ψ),
        v * cos(γ) * sin(ψ),
        v * sin(γ)
    )
    v_rel_vector = v_vector - wind
    v_rel = norm(v_rel_vector)
    # println("Time: $t s, Altitude: $h m, Velocity: $v m/s, Relative Velocity: $v_rel m/s, Density: $(p.atmospheric_density) kg/m³")
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
    heat_rate = C1 * p.atmospheric_density^n * v_rel^m_exp # Convective heat rate (W/m^2)
    p.cache.q_dot = heat_rate # Store heat rate in cache
    du[1] = (v * sin_γ) #  h_dot
    du[2] = (v/r) * cos_γ * sin_ψ / cos_θ # ϕ_dot (longitude)
    du[3] = (v/r) * cos_γ * cos_ψ # θ_dot (latitude)
    du[4] = (-drag / m - g * sin_γ) # v_dot
    du[5] = (lift/(m*v) * cos_β) + cos_γ*(v/r - g/v) # γ_dot (flight path angle)
    du[6] = (lift * sin_β) / (m * v * cos_γ) + (v * cos_γ * sin_ψ * tan_θ) / r # ψ_dot (azimuth)
    du[7] = heat_rate # heat rate (W/m^2)
end