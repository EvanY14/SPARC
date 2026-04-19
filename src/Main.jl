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
ϕ0 = deg2rad(126.7)            # Initial longitude in radians
θ0 = deg2rad(-3.93)            # Initial latitude in radians
v0 = 5845.39         # Initial velocity in m/s
γ0 = deg2rad(-15.49) # Initial flight path angle in radians
ψ0 = deg2rad(90.0) # Initial azimuth angle in radians

u0 = MVector{7, Float64}(h0, ϕ0, θ0, v0, γ0, ψ0, 0.0) # Initial state vector
tspan = (0.0, 500.0) # Time span for the simulation
# Define EDL parameters
mass = 3257.0 # kg
area = 15.904 # m^2
const μ = 3.986004418e14 # m^3/s^2 for Earth
R = 6378137.0 # m for Earth (WGS84 equatorial radius)

function latlonalt_to_cartesian_km(latitude, longitude, altitude, planet_radius)
    radius_km = (planet_radius + altitude) / 1e3
    x = radius_km * cos(latitude) * cos(longitude)
    y = radius_km * cos(latitude) * sin(longitude)
    z = radius_km * sin(latitude)
    return x, y, z
end

function cartesian_histories_km(altitudes, longitudes, latitudes, planet_radius)
    x = similar(Float64.(altitudes))
    y = similar(Float64.(altitudes))
    z = similar(Float64.(altitudes))
    for i in eachindex(altitudes)
        x[i], y[i], z[i] = latlonalt_to_cartesian_km(latitudes[i], longitudes[i], altitudes[i], planet_radius)
    end
    return x, y, z
end

optimal_control = CSV.read("optimal_trajectory.csv", DataFrame)
interp_optimal_control = linear_interpolation(optimal_control.Time_s, optimal_control.BankAngle_deg, extrapolation_bc=Line())
interp_optimal_alpha = linear_interpolation(optimal_control.Time_s, optimal_control.AngleOfAttack_deg, extrapolation_bc=Line())
β = (integrator) -> deg2rad(interp_optimal_control(integrator.t)) # Bank angle function in radians
α0_ref = deg2rad(interp_optimal_alpha(tspan[1]))
β0_ref = deg2rad(interp_optimal_control(tspan[1]))
# β = (integrator) -> deg2rad(45.0) # Bank angle function in radians
velocity_scale = abs(optimal_control.Velocity_1000mps[1] * 1e3 - v0) <= abs(optimal_control.Velocity_1000mps[1] * 1e4 - v0) ? 1e3 : 1e4
interp_altitude = linear_interpolation(optimal_control.Time_s, optimal_control.Altitude_100km .* 1e5, extrapolation_bc=Line())
interp_velocity = linear_interpolation(optimal_control.Time_s, optimal_control.Velocity_1000mps .* velocity_scale, extrapolation_bc=Line())
interp_longitude = linear_interpolation(optimal_control.Time_s, optimal_control.Longitude_deg .* (π / 180), extrapolation_bc=Line())
interp_latitude = linear_interpolation(optimal_control.Time_s, optimal_control.Latitude_deg .* (π / 180), extrapolation_bc=Line())
interp_flight_path = linear_interpolation(optimal_control.Time_s, optimal_control.FlightPath_deg .* (π / 180), extrapolation_bc=Line())
interp_azimuth = linear_interpolation(optimal_control.Time_s, optimal_control.Azimuth_deg .* (π / 180), extrapolation_bc=Line())
# Define time vector for interpolation
times = range(optimal_control.Time_s[1], optimal_control.Time_s[end], length=length(optimal_control.Time_s))
# Define optimal trajectory at evenly spaced time intervals
optimal_trajectory = SVector{6, AbstractInterpolation}(interp_altitude, interp_longitude, interp_latitude, interp_velocity, interp_flight_path, interp_azimuth)
optimal_ref_x_km, optimal_ref_y_km, optimal_ref_z_km = cartesian_histories_km(
    optimal_control.Altitude_100km .* 1e5,
    deg2rad.(optimal_control.Longitude_deg),
    deg2rad.(optimal_control.Latitude_deg),
    R,
)
display(plot(optimal_control.Time_s, optimal_control.Altitude_100km .* 1e5 ./ 1e3, xlabel="Time (s)", ylabel="Altitude (km)", title="Optimal Trajectory: Altitude vs Time", legend=false))
display(plot(optimal_control.Time_s, optimal_control.Velocity_1000mps .* velocity_scale, xlabel="Time (s)", ylabel="Velocity (m/s)", title="Optimal Trajectory: Velocity vs Time", legend=false))
display(plot(optimal_ref_x_km, optimal_ref_y_km, xlabel="X (km)", ylabel="Y (km)", title="Optimal Trajectory: Cartesian Ground Track", legend=false))

# β = (u, p, t) -> deg2rad(45*sin(t*pi/87)) # Bank angle function in radians
# β = () -> deg2rad(rand() * 90.0 - 45.0) # Bank angle in radians

# Define target states
target_altitude = 11848.0 # Termination altitude in meters
target_velocity = 500.0 # Target final velocity in m/s
target_γ = deg2rad(-5.0) # Target final flight path angle in radians
target_states = TargetStates(altitude=target_altitude, longitude=deg2rad(137.4), latitude=deg2rad(-4.5), velocity=target_velocity, flight_path_angle=target_γ)

# Define the atmospheric model
gram_atmosphere = GramAtmosphere("GRAMpy/", "GRAM_Data", false, "earth", DateTime(2012, 8, 6, 5, 10, 46.0), false)
# Define integration parameters
edl_cache = EDLCache()
optimization_states = OptimizationStates()
mpc_params = MPCParams{100, 7, 8, 0.1}(n_horizon=100, time_step=0.75, H_SCALE=1.0e5, V_SCALE=1.0e4, T_SCALE=1.0, n_exp=4.512, m_exp=0.82958, learning_rate=0.1)
edl_params = EDLParams(mass = mass, area = area, μ = μ, R = R, control_function = trackingmpc_shrinking, β = β0_ref, α = α0_ref, atmospheric_density_function = (LatLonAlt, t) -> atmospheric_density(LatLonAlt, t, gram_atmosphere, false), atmospheric_density = 0.0, wind = SVector{3, Float64}(zeros(3)), target_states = target_states, optimization_states = optimization_states, nominal_trajectory = optimal_trajectory, cache = edl_cache, mpc_params = mpc_params)
# Define callbacks
callbacks = CallbackSet(altitude_termination_condition, atmospheric_density_callback, saving_callback, control_callback)

# Define the ODE problem
prob = ODEProblem(edl_dynamics, u0, tspan, edl_params, callback=callbacks)
# Solve the ODE problem
sol = solve(prob, Tsit5(), dt=0.1, adaptive=false)#reltol=1e-10, abstol=1e-12,
# The solution `sol` now contains the state of the system over time
sim_altitudes = getindex.(sol.u, 1)
sim_longitudes = getindex.(sol.u, 2)
sim_latitudes = getindex.(sol.u, 3)
sim_velocities = getindex.(sol.u, 4)
sim_flight_paths = getindex.(sol.u, 5)
sim_azimuths = getindex.(sol.u, 6)
sim_heat_loads = getindex.(sol.u, 7)
sim_x_km, sim_y_km, sim_z_km = cartesian_histories_km(sim_altitudes, sim_longitudes, sim_latitudes, R)

display(plot(sol.t, sim_altitudes ./ 1e3 , xlabel="Time (s)", ylabel="Altitude (km)", title="EDL Simulation: Altitude vs Time", legend=false))
display(plot(sol.t, sim_velocities ./ 1e3 , xlabel="Time (s)", ylabel="Velocity (km/s)", title="EDL Simulation: Velocity vs Time", legend=false))
display(plot(sim_x_km, sim_y_km, xlabel="X (km)", ylabel="Y (km)", title="EDL Simulation: Cartesian Ground Track", legend=false))

# Get saved data
saved_data = saved_values.saveval
densities = zeros(length(saved_data))
betas = zeros(length(saved_data))
alphas = zeros(length(saved_data))
heat_rates = zeros(length(saved_data))
for i in 1:length(saved_data)
    densities[i] = saved_data[i][1]
    betas[i] = saved_data[i][2]
    alphas[i] = saved_data[i][3]
    heat_rates[i] = saved_data[i][4]
end

ref_altitudes_at_sol = interp_altitude.(sol.t)
ref_longitudes_at_sol = interp_longitude.(sol.t)
ref_latitudes_at_sol = interp_latitude.(sol.t)
ref_velocities_at_sol = interp_velocity.(sol.t)
ref_flight_paths_at_sol = interp_flight_path.(sol.t)
ref_azimuths_at_sol = interp_azimuth.(sol.t)
ref_alpha_at_sol = deg2rad.(interp_optimal_alpha.(sol.t))
ref_beta_at_sol = deg2rad.(interp_optimal_control.(sol.t))
ref_x_km, ref_y_km, ref_z_km = cartesian_histories_km(ref_altitudes_at_sol, ref_longitudes_at_sol, ref_latitudes_at_sol, R)

position_error_x_km = sim_x_km .- ref_x_km
position_error_y_km = sim_y_km .- ref_y_km
position_error_z_km = sim_z_km .- ref_z_km
position_error_norm_km = sqrt.(position_error_x_km.^2 .+ position_error_y_km.^2 .+ position_error_z_km.^2)

if length(saved_values.t) >= 2
    density_at_sol = linear_interpolation(saved_values.t, densities, extrapolation_bc=Line()).(sol.t)
    beta_at_sol = linear_interpolation(saved_values.t, betas, extrapolation_bc=Line()).(sol.t)
    alpha_at_sol = linear_interpolation(saved_values.t, alphas, extrapolation_bc=Line()).(sol.t)
    heat_rate_at_sol = linear_interpolation(saved_values.t, heat_rates, extrapolation_bc=Line()).(sol.t)
else
    density_at_sol = fill(NaN, length(sol.t))
    beta_at_sol = fill(NaN, length(sol.t))
    alpha_at_sol = fill(NaN, length(sol.t))
    heat_rate_at_sol = fill(NaN, length(sol.t))
end

simulation_output = DataFrame(
    Time_s = sol.t,
    Altitude_m = sim_altitudes,
    Longitude_rad = sim_longitudes,
    Latitude_rad = sim_latitudes,
    Longitude_deg = rad2deg.(sim_longitudes),
    Latitude_deg = rad2deg.(sim_latitudes),
    Velocity_mps = sim_velocities,
    FlightPath_rad = sim_flight_paths,
    FlightPath_deg = rad2deg.(sim_flight_paths),
    Azimuth_rad = sim_azimuths,
    Azimuth_deg = rad2deg.(sim_azimuths),
    HeatLoad_Jm2 = sim_heat_loads,
    PositionX_km = sim_x_km,
    PositionY_km = sim_y_km,
    PositionZ_km = sim_z_km,
    ReferenceAltitude_m = ref_altitudes_at_sol,
    ReferenceLongitude_rad = ref_longitudes_at_sol,
    ReferenceLatitude_rad = ref_latitudes_at_sol,
    ReferenceLongitude_deg = rad2deg.(ref_longitudes_at_sol),
    ReferenceLatitude_deg = rad2deg.(ref_latitudes_at_sol),
    ReferenceVelocity_mps = ref_velocities_at_sol,
    ReferenceFlightPath_rad = ref_flight_paths_at_sol,
    ReferenceFlightPath_deg = rad2deg.(ref_flight_paths_at_sol),
    ReferenceAzimuth_rad = ref_azimuths_at_sol,
    ReferenceAzimuth_deg = rad2deg.(ref_azimuths_at_sol),
    ReferencePositionX_km = ref_x_km,
    ReferencePositionY_km = ref_y_km,
    ReferencePositionZ_km = ref_z_km,
    PositionErrorX_km = position_error_x_km,
    PositionErrorY_km = position_error_y_km,
    PositionErrorZ_km = position_error_z_km,
    PositionErrorNorm_km = position_error_norm_km,
    AtmosphericDensity_kgm3 = density_at_sol,
    BankAngle_rad = beta_at_sol,
    BankAngle_deg = rad2deg.(beta_at_sol),
    AngleOfAttack_rad = alpha_at_sol,
    AngleOfAttack_deg = rad2deg.(alpha_at_sol),
    ReferenceBankAngle_rad = ref_beta_at_sol,
    ReferenceBankAngle_deg = rad2deg.(ref_beta_at_sol),
    ReferenceAngleOfAttack_rad = ref_alpha_at_sol,
    ReferenceAngleOfAttack_deg = rad2deg.(ref_alpha_at_sol),
    HeatRate_Wm2 = heat_rate_at_sol,
)
CSV.write("simulation_output.csv", simulation_output)

# println(size(betas))
# println(saved_values.t)
# println(sol.t)
display(plot(saved_values.t[2:end], densities[2:end], xlabel="Time (s)", ylabel="Atmospheric Density (kg/m³)", title="Atmospheric Density Profile", legend=false, yscale=:log10))
display(plot(saved_values.t, betas .* (180 / π), xlabel="Time (s)", ylabel="Bank Angle (deg)", title="Bank Angle Profile", legend=false))
heat_rate_plot = plot(saved_values.t, heat_rates, xlabel="Time (s)", ylabel="Convective Heat Rate (W/m²)", title="Convective Heat Rate Profile", legend=false)
heat_load_plot = plot(sol.t, sim_heat_loads, xlabel="Time (s)", ylabel="Convective Heat Load (J/m²)", title="Convective Heat Load Profile from State", legend=false)
display(plot(heat_rate_plot, heat_load_plot, layout=(2,1)))

# Compare the MPC-tracked simulation and applied controls with the optimal reference
mpc_altitude_plot = plot(
    optimal_control.Time_s,
    optimal_control.Altitude_100km .* 100.0,
    xlabel = "Time (s)",
    ylabel = "Altitude (km)",
    title = "MPC vs Optimal Reference: Altitude",
    label = "Optimal reference",
    linewidth = 2,
)
plot!(mpc_altitude_plot, sol.t, sim_altitudes ./ 1e3, label = "MPC simulation", linewidth = 2)

mpc_velocity_plot = plot(
    optimal_control.Time_s,
    optimal_control.Velocity_1000mps .* velocity_scale,
    xlabel = "Time (s)",
    ylabel = "Velocity (m/s)",
    title = "MPC vs Optimal Reference: Velocity",
    label = "Optimal reference",
    linewidth = 2,
)
plot!(mpc_velocity_plot, sol.t, sim_velocities, label = "MPC simulation", linewidth = 2)

mpc_fpa_plot = plot(
    optimal_control.Time_s,
    optimal_control.FlightPath_deg,
    xlabel = "Time (s)",
    ylabel = "Flight path angle (deg)",
    title = "MPC vs Optimal Reference: Flight Path",
    label = "Optimal reference",
    linewidth = 2,
)
plot!(mpc_fpa_plot, sol.t, rad2deg.(sim_flight_paths), label = "MPC simulation", linewidth = 2)
display(plot(mpc_altitude_plot, mpc_velocity_plot, mpc_fpa_plot, layout = (3, 1), size = (900, 900)))

mpc_ground_track_plot = plot(
    optimal_ref_x_km,
    optimal_ref_y_km,
    xlabel = "X (km)",
    ylabel = "Y (km)",
    title = "MPC vs Optimal Reference: Cartesian Ground Track",
    label = "Optimal reference",
    linewidth = 2,
)
plot!(
    mpc_ground_track_plot,
    sim_x_km,
    sim_y_km,
    label = "MPC simulation",
    linewidth = 2,
)
display(mpc_ground_track_plot)

cartesian_position_plot = plot(
    optimal_control.Time_s,
    optimal_ref_x_km,
    xlabel = "Time (s)",
    ylabel = "Position (km)",
    title = "MPC vs Optimal Reference: Cartesian Position",
    label = "Reference X",
    linewidth = 2,
)
plot!(cartesian_position_plot, optimal_control.Time_s, optimal_ref_y_km, label = "Reference Y", linewidth = 2)
plot!(cartesian_position_plot, optimal_control.Time_s, optimal_ref_z_km, label = "Reference Z", linewidth = 2)
plot!(cartesian_position_plot, sol.t, sim_x_km, label = "Simulation X", linestyle = :dash, linewidth = 2)
plot!(cartesian_position_plot, sol.t, sim_y_km, label = "Simulation Y", linestyle = :dash, linewidth = 2)
plot!(cartesian_position_plot, sol.t, sim_z_km, label = "Simulation Z", linestyle = :dash, linewidth = 2)
display(cartesian_position_plot)

display(plot(
    sol.t,
    position_error_norm_km,
    xlabel = "Time (s)",
    ylabel = "Position error (km)",
    title = "Cartesian Position Error Norm",
    legend = false,
    linewidth = 2,
))

if !isempty(saved_values.t)
    control_times = saved_values.t
    mpc_alpha_plot = plot(
        optimal_control.Time_s,
        optimal_control.AngleOfAttack_deg,
        xlabel = "Time (s)",
        ylabel = "Angle of attack (deg)",
        title = "MPC vs Optimal Reference: Angle of Attack",
        label = "Optimal reference",
        linewidth = 2,
    )
    plot!(mpc_alpha_plot, control_times, rad2deg.(alphas), label = "MPC command", linewidth = 2)

    mpc_bank_plot = plot(
        optimal_control.Time_s,
        optimal_control.BankAngle_deg,
        xlabel = "Time (s)",
        ylabel = "Bank angle (deg)",
        title = "MPC vs Optimal Reference: Bank Angle",
        label = "Optimal reference",
        linewidth = 2,
    )
    plot!(mpc_bank_plot, control_times, rad2deg.(betas), label = "MPC command", linewidth = 2)
    display(plot(mpc_alpha_plot, mpc_bank_plot, layout = (2, 1), size = (900, 650)))
end

opt_states = edl_params.optimization_states
if !isempty(opt_states.h_c)
    mpc_prediction_times = sol.t[end] .+ (1:length(opt_states.h_c)) .* edl_params.mpc_params.time_step
    latest_prediction_plot = plot(
        optimal_control.Time_s,
        optimal_control.Altitude_100km .* 100.0,
        xlabel = "Time (s)",
        ylabel = "Altitude (km)",
        title = "Latest MPC Horizon vs Optimal Reference",
        label = "Optimal reference",
        linewidth = 2,
    )
    plot!(latest_prediction_plot, sol.t, sim_altitudes ./ 1e3, label = "MPC simulation", linewidth = 2)
    plot!(
        latest_prediction_plot,
        mpc_prediction_times,
        opt_states.h_c ./ 1e3,
        label = "Latest MPC prediction",
        linestyle = :dot,
        linewidth = 2,
    )
    display(latest_prediction_plot)
end


# Monte Carlo to test atmospheric disturbances
# num_simulations = 100
# final_latitudes = zeros(num_simulations)
# final_longitudes = zeros(num_simulations)
# altitude_profiles = Vector{Vector{Float64}}(undef, num_simulations)
# velocity_profiles = Vector{Vector{Float64}}(undef, num_simulations)
# times = Vector{Vector{Float64}}(undef, num_simulations)
# gram_atmosphere = GramAtmosphere("GRAMpy/", "GRAM_Data", false, "earth", DateTime(2012, 8, 6, 5, 10, 46.0))
# edl_params.atmospheric_density_function = (LatLonAlt, t) -> atmospheric_density(LatLonAlt, t, gram_atmosphere, true)
# @showprogress for sim in 24:num_simulations
#     try
#         prob_mc = ODEProblem(edl_dynamics, u0, tspan, edl_params, callback=callbacks)
#         sol_mc = solve(prob_mc, Tsit5(), dt=0.5, adaptive=false)
#         altitude_profiles[sim] = getindex.(sol_mc.u, 1) ./ 1e3
#         velocity_profiles[sim] = getindex.(sol_mc.u, 4) ./ 1e3
#         times[sim] = sol_mc.t
#         final_latitudes[sim] = getindex(sol_mc.u[end], 3) * (180 / π)
#         final_longitudes[sim] = getindex(sol_mc.u[end], 2) * (180 / π)

#         saved_data = saved_values.saveval
#         densities = zeros(length(saved_data))
#         betas = zeros(length(saved_data))
#         alphas = zeros(length(saved_data))
#         heat_rates = zeros(length(saved_data))
#         for i in 1:length(saved_data)
#             densities[i] = saved_data[i][1]
#             betas[i] = saved_data[i][2]
#             alphas[i] = saved_data[i][3]
#             heat_rates[i] = saved_data[i][4]
#         end

#         df = DataFrame(
#             Time_s = times[sim],
#             Altitude_100km = altitude_profiles[sim],
#             Longitude_fin_deg = getindex.(sol_mc.u, 2) .* (180 / π),
#             Latitude_fin_deg = getindex.(sol_mc.u, 3) .* (180 / π),
#             Velocity_1000mps = velocity_profiles[sim],
#             FlightPath_deg = getindex.(sol_mc.u, 5) .* (180 / π),
#             Azimuth_deg = getindex.(sol_mc.u, 6) .* (180 / π),
#             HeatLoad_Jm2 = getindex.(sol_mc.u, 7)
#         )
#         CSV.write("optimal_trajectory_mpc_$(sim).csv", df)

#         df = DataFrame(
#             Time_s = saved_values.t,
#             AngleOfAttack_deg = alphas .* (180 / π),
#             BankAngle_deg = betas .* (180 / π),
#             HeatRate_Wm2 = heat_rates,
#         )
#         CSV.write("optimal_trajectory_mpc_$(sim)_control.csv", df)
#     catch e
#         continue
#     end
# end

# # Plot MC results
# # Altitude profiles
# plt1 = plot(title="Monte Carlo Simulations: Altitude Profiles", xlabel="Time (s)", ylabel="Altitude (km)", legend=true)
# for i in 1:num_simulations
#     plot!(plt1, times[i], altitude_profiles[i], color=:grey, alpha=0.3, label=false)
# end
# # Plot nominal trajectory
# plot!(plt1, sol.t, getindex.(sol.u, 1) ./ 1e3, color=:red, label="Nominal Trajectory")
# display(plt1)
# savefig(plt1, "monte_carlo_altitude_profiles.pdf")

# # Velocity profiles
# plt2 = plot(title="Monte Carlo Simulations: Velocity Profiles", xlabel="Time (s)", ylabel="Velocity (km/s)", legend=true)
# for i in 1:num_simulations
#     plot!(plt2, times[i], velocity_profiles[i], color=:grey, alpha=0.3, label=false)
# end
# # Plot nominal trajectory
# plot!(plt2, sol.t, getindex.(sol.u, 4) ./ 1e3, color=:red, label="Nominal Trajectory")
# display(plt2)
# savefig(plt2, "monte_carlo_velocity_profiles.pdf")

# # Final landing locations
# plt3 = plot(final_longitudes, final_latitudes, seriestype=:scatter, xlabel="Longitude (deg)", ylabel="Latitude (deg)", title="Monte Carlo Simulations: Final Landing Locations", legend=true, label="Landing Locations", alpha=0.6)
# plot!(plt3, [getindex(sol.u[end], 2) * (180 / π)], [getindex(sol.u[end], 3) * (180 / π)], seriestype=:scatter, color=:red, markershape=:star5, markersize=8, label="Nominal Landing Location")
# plot!(plt3, [rad2deg(target_states.longitude)], [rad2deg(target_states.latitude)], seriestype=:scatter, color=:blue, markershape=:diamond, markersize=8, label="Target Location")
# display(plt3)
# savefig(plt3, "monte_carlo_final_landing_locations.pdf")
