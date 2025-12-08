using Revise
includet("model/SimulatorModel.jl")

using .SimulatorModel
using StaticArrays
using Plots
using DifferentialEquations
using ProgressMeter
using CSV
using DataFrames
using Interpolations
gr()
# Define initial conditions and parameters
h0 = 125000.0      # Initial altitude in meters
ϕ0 = deg2rad(126.8)            # Initial longitude in radians
θ0 = deg2rad(-3.9)            # Initial latitude in radians
v0 = 5845.39         # Initial velocity in m/s
γ0 = deg2rad(-15.49) # Initial flight path angle in radians
ψ0 = deg2rad(91.0) # Initial azimuth angle in radians

u0 = MVector{7, Float64}(h0, ϕ0, θ0, v0, γ0, ψ0, 0.0) # Initial state vector
tspan = (0.0, 500.0) # Time span for the simulation
# Define EDL parameters
mass = 3257.0 # kg
drag_coefficient = 1.46
lift_coefficient = 0.24*drag_coefficient
area = 15.904 # m^2
const μ = 4.2828372e13 # m^3/s^2 for Mars
R = 3389500.0 # m for Mars
# β = (u, p, t) -> deg2rad(45.0) # Constant bank angle in radians
optimal_control = CSV.read("optimal_trajectory.csv", DataFrame)
interp_optimal_control = linear_interpolation(optimal_control.Time_s, optimal_control.BankAngle_deg, extrapolation_bc=Line())
β = (integrator) -> deg2rad(interp_optimal_control(integrator.t)) # Bank angle function in radians
interp_altitude = linear_interpolation(optimal_control.Time_s, optimal_control.Altitude_100km .* 1e5, extrapolation_bc=Line())
interp_velocity = linear_interpolation(optimal_control.Time_s, optimal_control.Velocity_1000mps .* 1e3, extrapolation_bc=Line())
interp_longitude = linear_interpolation(optimal_control.Time_s, optimal_control.Longitude_deg .* (π / 180), extrapolation_bc=Line())
interp_latitude = linear_interpolation(optimal_control.Time_s, optimal_control.Latitude_deg .* (π / 180), extrapolation_bc=Line())
interp_flight_path = linear_interpolation(optimal_control.Time_s, optimal_control.FlightPath_deg .* (π / 180), extrapolation_bc=Line())
interp_azimuth = linear_interpolation(optimal_control.Time_s, optimal_control.Azimuth_deg .* (π / 180), extrapolation_bc=Line())
# Define time vector for interpolation
times = range(optimal_control.Time_s[1], optimal_control.Time_s[end], length=length(optimal_control.Time_s))
# Define optimal trajectory at evenly spaced time intervals
optimal_trajectory = SVector{6, AbstractInterpolation}(interp_altitude, interp_longitude, interp_latitude, interp_velocity, interp_flight_path, interp_azimuth)

# β = (u, p, t) -> deg2rad(45*sin(t*pi/87)) # Bank angle function in radians
# β = () -> deg2rad(rand() * 90.0 - 45.0) # Bank angle in radians

# Define target states
target_altitude = 11848.0 # Termination altitude in meters
target_velocity = 500.0 # Target final velocity in m/s
target_γ = deg2rad(-5.0) # Target final flight path angle in radians
target_states = TargetStates(altitude=target_altitude, longitude=deg2rad(137.4), latitude=deg2rad(-4.5), velocity=target_velocity, flight_path_angle=target_γ)

# Define the atmospheric model
# exponential_atmosphere = ExponentialAtmosphere(0.02, 11.1) # surface density in kg/m^3, scale height in km
gram_atmosphere = GramAtmosphere("GRAMpy/", "GRAM_Data", false, "mars", DateTime(2024, 1, 1, 0, 0, 0.0))

# Define integration parameters
edl_cache = EDLCache()
optimization_states = OptimizationStates()
edl_params = EDLParams(mass, drag_coefficient, lift_coefficient, area, μ, R, mpc, 0.0, (LatLonAlt, t) -> atmospheric_density(LatLonAlt, t, gram_atmosphere, false), 0.0, SVector{3, Float64}(zeros(3)), target_states, optimization_states, optimal_trajectory, edl_cache)

# Define callbacks
callbacks = CallbackSet(altitude_termination_condition, atmospheric_density_callback, saving_callback, control_callback)

# Define the ODE problem
prob = ODEProblem(edl_dynamics, u0, tspan, edl_params, callback=callbacks)
# Solve the ODE problem
sol = solve(prob, Tsit5(), reltol=1e-10, abstol=1e-12, dtmax=0.1)
# The solution `sol` now contains the state of the system over time
display(plot(sol.t, getindex.(sol.u, 1) ./ 1e3 , xlabel="Time (s)", ylabel="Altitude (km)", title="EDL Simulation: Altitude vs Time", legend=false))
display(plot(sol.t, getindex.(sol.u, 4) ./ 1e3 , xlabel="Time (s)", ylabel="Velocity (km/s)", title="EDL Simulation: Velocity vs Time", legend=false))
display(plot(getindex.(sol.u, 2) .* (180 / π), getindex.(sol.u, 3) .* (180 / π), xlabel="Longitude (deg)", ylabel="Latitude (deg)", title="EDL Simulation: Ground Track", legend=false))

# Get saved data
saved_data = saved_values.saveval
densities = zeros(length(saved_data))
betas = zeros(length(saved_data))
heat_rates = zeros(length(saved_data))
for i in 1:length(saved_data)
    densities[i] = saved_data[i][1]
    betas[i] = saved_data[i][2]
    heat_rates[i] = saved_data[i][3]
end
println(size(betas))
println(size(saved_values.t))
# display(plot(saved_values.t, densities, xlabel="Time (s)", ylabel="Atmospheric Density (kg/m³)", title="Atmospheric Density Profile", legend=false, yscale=:log10))
display(plot(saved_values.t, betas .* (180 / π), xlabel="Time (s)", ylabel="Bank Angle (deg)", title="Bank Angle Profile", legend=false))
heat_rate_plot = plot(saved_values.t, heat_rates, xlabel="Time (s)", ylabel="Convective Heat Rate (W/m²)", title="Convective Heat Rate Profile", legend=false)
heat_load_plot = plot(sol.t, getindex.(sol.u, 7), xlabel="Time (s)", ylabel="Convective Heat Load (J/m²)", title="Convective Heat Load Profile from State", legend=false)
display(plot(heat_rate_plot, heat_load_plot, layout=(2,1)))
# Monte Carlo to test atmospheric disturbances
num_simulations = 500
final_latitudes = zeros(num_simulations)
final_longitudes = zeros(num_simulations)
altitude_profiles = Vector{Vector{Float64}}(undef, num_simulations)
velocity_profiles = Vector{Vector{Float64}}(undef, num_simulations)
times = Vector{Vector{Float64}}(undef, num_simulations)
gram_atmosphere = GramAtmosphere("GRAMpy/", "GRAM_Data", false, "mars", DateTime(2024, 1, 1, 0, 0, 0.0))
edl_params.atmospheric_density_function = (LatLonAlt, t) -> atmospheric_density(LatLonAlt, t, gram_atmosphere, true)
@showprogress for sim in 1:num_simulations
    prob_mc = ODEProblem(edl_dynamics, u0, tspan, edl_params, callback=callbacks)
    sol_mc = solve(prob_mc, Tsit5(), reltol=1e-10, abstol=1e-12, dtmax=0.1)
    altitude_profiles[sim] = getindex.(sol_mc.u, 1) ./ 1e3
    velocity_profiles[sim] = getindex.(sol_mc.u, 4) ./ 1e3
    times[sim] = sol_mc.t
    final_latitudes[sim] = getindex(sol_mc.u[end], 3) * (180 / π)
    final_longitudes[sim] = getindex(sol_mc.u[end], 2) * (180 / π)
end

# Plot MC results
# Altitude profiles
plt1 = plot(title="Monte Carlo Simulations: Altitude Profiles", xlabel="Time (s)", ylabel="Altitude (km)", legend=true)
for i in 1:num_simulations
    plot!(plt1, times[i], altitude_profiles[i], color=:grey, alpha=0.3, label=false)
end
# Plot nominal trajectory
plot!(plt1, sol.t, getindex.(sol.u, 1) ./ 1e3, color=:red, label="Nominal Trajectory")
display(plt1)
savefig(plt1, "monte_carlo_altitude_profiles.pdf")

# Velocity profiles
plt2 = plot(title="Monte Carlo Simulations: Velocity Profiles", xlabel="Time (s)", ylabel="Velocity (km/s)", legend=true)
for i in 1:num_simulations
    plot!(plt2, times[i], velocity_profiles[i], color=:grey, alpha=0.3, label=false)
end
# Plot nominal trajectory
plot!(plt2, sol.t, getindex.(sol.u, 4) ./ 1e3, color=:red, label="Nominal Trajectory")
display(plt2)
savefig(plt2, "monte_carlo_velocity_profiles.pdf")

# Final landing locations
plt3 = plot(final_longitudes, final_latitudes, seriestype=:scatter, xlabel="Longitude (deg)", ylabel="Latitude (deg)", title="Monte Carlo Simulations: Final Landing Locations", legend=true, label="Landing Locations", alpha=0.6)
plot!(plt3, [getindex(sol.u[end], 2) * (180 / π)], [getindex(sol.u[end], 3) * (180 / π)], seriestype=:scatter, color=:red, markershape=:star5, markersize=8, label="Nominal Landing Location")
plot!(plt3, [rad2deg(target_states.longitude)], [rad2deg(target_states.latitude)], seriestype=:scatter, color=:blue, markershape=:diamond, markersize=8, label="Target Location")
display(plt3)
savefig(plt3, "monte_carlo_final_landing_locations.pdf")