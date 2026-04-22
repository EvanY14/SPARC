function reference_initial_conditions(optimal_control)
    h0 = Float64(optimal_control.Altitude_100km[1]) * 1e5
    ϕ0 = deg2rad(Float64(optimal_control.Longitude_deg[1]))
    θ0 = deg2rad(Float64(optimal_control.Latitude_deg[1]))
    v0 = Float64(optimal_control.Velocity_1000mps[1]) * 1e3
    γ0 = deg2rad(Float64(optimal_control.FlightPath_deg[1]))
    ψ0 = deg2rad(Float64(optimal_control.Azimuth_deg[1]))
    tspan = (Float64(optimal_control.Time_s[1]), Float64(optimal_control.Time_s[end]))

    return (
        h0 = h0,
        ϕ0 = ϕ0,
        θ0 = θ0,
        v0 = v0,
        γ0 = γ0,
        ψ0 = ψ0,
        tspan = tspan,
    )
end