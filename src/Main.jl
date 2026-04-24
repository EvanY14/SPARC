try
    using Revise
    includet("model/SimulatorModel.jl")
catch err
    if err isa ArgumentError && occursin("Package Revise not found", sprint(showerror, err))
        include("model/SimulatorModel.jl")
    else
        rethrow()
    end
end
using .SimulatorModel
using StaticArrays
const BROWSER_PLOTS_AVAILABLE = haskey(ENV, "DISPLAY") && !isempty(ENV["DISPLAY"]) && Sys.which("xdg-open") !== nothing
if !BROWSER_PLOTS_AVAILABLE && !haskey(ENV, "GKSwstype")
    ENV["GKSwstype"] = "100"
end
using Plots
using DifferentialEquations
using CSV
using DataFrames
using Interpolations
include("reference/trajectory_initialization.jl")
# BROWSER_PLOTS_AVAILABLE ? plotly() : gr()
plotly()

optimal_control = CSV.read("optimal_trajectory.csv", DataFrame)
init = reference_initial_conditions(optimal_control)
tspan = init.tspan

h0 = init.h0
ϕ0 = init.ϕ0
θ0 = init.θ0
v0 = init.v0
γ0 = init.γ0
ψ0 = init.ψ0

u0 = MVector{7, Float64}(h0, ϕ0, θ0, v0, γ0, ψ0, 0.0)

# Define EDL parameters
mass = SimulatorModel.VEHICLE.mass
area = SimulatorModel.VEHICLE.reference_area
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
target_altitude = 25000.0 # Termination altitude in meters
target_velocity = 700.0 # Target final velocity in m/s
target_γ = deg2rad(-5.0) # Target final flight path angle in radians
target_states = SimulatorModel.TargetStates(altitude=target_altitude, longitude=deg2rad(133.4), latitude=deg2rad(-4.5), velocity=target_velocity, flight_path_angle=target_γ)

# Define the atmospheric model
gram_atmosphere = SimulatorModel.GramAtmosphere("GRAMpy/", "GRAM_Data", false, "earth", SimulatorModel.DateTime(2012, 8, 6, 5, 10, 46.0), false)
# Define integration parameters and run each controller with independent MPC state.
function reset_saved_values!()
    empty!(SimulatorModel.saved_values.t)
    empty!(SimulatorModel.saved_values.saveval)
end

function extract_saved_histories(saved_data)
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
    return densities, betas, alphas, heat_rates
end

function make_edl_params(control_function)
    mpc_params = SimulatorModel.MPCParams{100, 7, 8, 0.1}(
        n_horizon=SimulatorModel.mpg_default_horizon(),
        time_step=SimulatorModel.mpg_default_time_step(),
        H_SCALE=1.0e5,
        V_SCALE=1.0e4,
        T_SCALE=1.0,
        n_exp=4.512,
        m_exp=0.82958,
        learning_rate=0.1,
    )
    return SimulatorModel.EDLParams(mass = mass, area = area, μ = μ, R = R, control_function = control_function, β = β0_ref, α = α0_ref, atmospheric_density_function = (LatLonAlt, t) -> SimulatorModel.atmospheric_density(LatLonAlt, t, gram_atmosphere, false), atmospheric_density = 0.0, wind = SVector{3, Float64}(zeros(3)), target_states = target_states, optimization_states = SimulatorModel.OptimizationStates(), nominal_trajectory = optimal_trajectory, cache = SimulatorModel.EDLCache(), mpc_params = mpc_params)
end

function run_controller(control_function)
    reset_saved_values!()
    edl_params = make_edl_params(control_function)
    callbacks = CallbackSet(SimulatorModel.altitude_termination_condition, SimulatorModel.atmospheric_density_callback, SimulatorModel.saving_callback, SimulatorModel.control_callback)
    prob = ODEProblem(SimulatorModel.edl_dynamics, u0, tspan, edl_params, callback=callbacks)
    sol = solve(prob, Tsit5(), dt=0.1, adaptive=false)
    return sol, edl_params, copy(SimulatorModel.saved_values.t), copy(SimulatorModel.saved_values.saveval)
end

function open_loop_control(integrator)
    t = integrator.t
    β_cmd = deg2rad(interp_optimal_control(t))
    α_cmd = deg2rad(interp_optimal_alpha(t))
    return β_cmd, α_cmd
end

function build_simulation_output(label, sol, saved_t, saved_data)
    sim_altitudes = getindex.(sol.u, 1)
    sim_longitudes = getindex.(sol.u, 2)
    sim_latitudes = getindex.(sol.u, 3)
    sim_velocities = getindex.(sol.u, 4)
    sim_flight_paths = getindex.(sol.u, 5)
    sim_azimuths = getindex.(sol.u, 6)
    sim_heat_loads = getindex.(sol.u, 7)
    sim_x_km, sim_y_km, sim_z_km = cartesian_histories_km(sim_altitudes, sim_longitudes, sim_latitudes, R)

    densities, betas, alphas, heat_rates = extract_saved_histories(saved_data)

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

    if length(saved_t) >= 2
        density_at_sol = linear_interpolation(saved_t, densities, extrapolation_bc=Line()).(sol.t)
        beta_at_sol = linear_interpolation(saved_t, betas, extrapolation_bc=Line()).(sol.t)
        alpha_at_sol = linear_interpolation(saved_t, alphas, extrapolation_bc=Line()).(sol.t)
        heat_rate_at_sol = linear_interpolation(saved_t, heat_rates, extrapolation_bc=Line()).(sol.t)
    else
        density_at_sol = fill(NaN, length(sol.t))
        beta_at_sol = fill(NaN, length(sol.t))
        alpha_at_sol = fill(NaN, length(sol.t))
        heat_rate_at_sol = fill(NaN, length(sol.t))
    end

    df = DataFrame(
        Algorithm = fill(label, length(sol.t)),
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

    return (
        df = df,
        sol = sol,
        saved_t = saved_t,
        densities = densities,
        betas = betas,
        alphas = alphas,
        heat_rates = heat_rates,
        sim_altitudes = sim_altitudes,
        sim_longitudes = sim_longitudes,
        sim_latitudes = sim_latitudes,
        sim_velocities = sim_velocities,
        sim_flight_paths = sim_flight_paths,
        sim_azimuths = sim_azimuths,
        sim_heat_loads = sim_heat_loads,
        sim_x_km = sim_x_km,
        sim_y_km = sim_y_km,
        sim_z_km = sim_z_km,
        position_error_norm_km = position_error_norm_km,
    )
end

shrinking_sol, shrinking_params, shrinking_saved_t, shrinking_saved_data = run_controller(SimulatorModel.trackingmpc)
mpg_sol, mpg_params, mpg_saved_t, mpg_saved_data = run_controller(SimulatorModel.mpg)
integral_mpg_sol, integral_mpg_params, integral_mpg_saved_t, integral_mpg_saved_data = run_controller(SimulatorModel.mpg_integral)
open_loop_sol, open_loop_params, open_loop_saved_t, open_loop_saved_data = run_controller(open_loop_control)
sm_mpg_sol, sm_mpg_params, sm_mpg_saved_t, sm_mpg_saved_data = run_controller(SimulatorModel.sm_mpg)

shrinking = build_simulation_output("Shrinking horizon MPC", shrinking_sol, shrinking_saved_t, shrinking_saved_data)
mpg_result = build_simulation_output("MPg", mpg_sol, mpg_saved_t, mpg_saved_data)
integral_mpg_result = build_simulation_output("Integral MPg", integral_mpg_sol, integral_mpg_saved_t, integral_mpg_saved_data)
open_loop_result = build_simulation_output("Open Loop", open_loop_sol, open_loop_saved_t, open_loop_saved_data)
sm_mpg_result = build_simulation_output("SM-MPG", sm_mpg_sol, sm_mpg_saved_t, sm_mpg_saved_data)

CSV.write("simulation_output_shrinking_mpc.csv", shrinking.df)
CSV.write("simulation_output_mpg.csv", mpg_result.df)
CSV.write("simulation_output_integral_mpg.csv", integral_mpg_result.df)
CSV.write("simulation_output_open_loop.csv", open_loop_result.df)
CSV.write("simulation_output_sm_mpg.csv", sm_mpg_result.df)
CSV.write("simulation_output.csv", vcat(shrinking.df, mpg_result.df, integral_mpg_result.df, open_loop_result.df, sm_mpg_result.df))

edl_altitude_plot = plot(shrinking.sol.t, shrinking.sim_altitudes ./ 1e3, xlabel="Time (s)", ylabel="Altitude (km)", title="EDL Simulation: Altitude vs Time", label="Shrinking horizon MPC", linewidth=2)
plot!(edl_altitude_plot, mpg_result.sol.t, mpg_result.sim_altitudes ./ 1e3, label="MPg", linewidth=2)
plot!(edl_altitude_plot, integral_mpg_result.sol.t, integral_mpg_result.sim_altitudes ./ 1e3, label="Integral MPg", linewidth=2)
plot!(edl_altitude_plot, open_loop_result.sol.t, open_loop_result.sim_altitudes ./ 1e3, label="Open Loop", linewidth=2)
plot!(edl_altitude_plot, sm_mpg_result.sol.t, sm_mpg_result.sim_altitudes ./ 1e3, label="SM-MPG", linewidth=2)
display(edl_altitude_plot)
edl_velocity_plot = plot(shrinking.sol.t, shrinking.sim_velocities ./ 1e3, xlabel="Time (s)", ylabel="Velocity (km/s)", title="EDL Simulation: Velocity vs Time", label="Shrinking horizon MPC", linewidth=2)
plot!(edl_velocity_plot, mpg_result.sol.t, mpg_result.sim_velocities ./ 1e3, label="MPg", linewidth=2)
plot!(edl_velocity_plot, integral_mpg_result.sol.t, integral_mpg_result.sim_velocities ./ 1e3, label="Integral MPg", linewidth=2)
plot!(edl_velocity_plot, open_loop_result.sol.t, open_loop_result.sim_velocities ./ 1e3, label="Open Loop", linewidth=2)
plot!(edl_velocity_plot, sm_mpg_result.sol.t, sm_mpg_result.sim_velocities ./ 1e3, label="SM-MPG", linewidth=2)
display(edl_velocity_plot)
edl_ground_track_plot = plot(shrinking.sim_x_km, shrinking.sim_y_km, xlabel="X (km)", ylabel="Y (km)", title="EDL Simulation: Cartesian Ground Track", label="Shrinking horizon MPC", linewidth=2)
plot!(edl_ground_track_plot, mpg_result.sim_x_km, mpg_result.sim_y_km, label="MPg", linewidth=2)
plot!(edl_ground_track_plot, integral_mpg_result.sim_x_km, integral_mpg_result.sim_y_km, label="Integral MPg", linewidth=2)
plot!(edl_ground_track_plot, open_loop_result.sim_x_km, open_loop_result.sim_y_km, label="Open Loop", linewidth=2)
plot!(edl_ground_track_plot, sm_mpg_result.sim_x_km, sm_mpg_result.sim_y_km, label="SM-MPG", linewidth=2)
display(edl_ground_track_plot)

density_plot = plot(shrinking.saved_t[2:end], shrinking.densities[2:end], xlabel="Time (s)", ylabel="Atmospheric Density (kg/m³)", title="Atmospheric Density Profile", label="Shrinking horizon MPC", yscale=:log10, linewidth=2)
plot!(density_plot, mpg_result.saved_t[2:end], mpg_result.densities[2:end], label="MPg", linewidth=2)
plot!(density_plot, integral_mpg_result.saved_t[2:end], integral_mpg_result.densities[2:end], label="Integral MPg", linewidth=2)
plot!(density_plot, open_loop_result.saved_t[2:end], open_loop_result.densities[2:end], label="Open Loop", linewidth=2)
plot!(density_plot, sm_mpg_result.saved_t[2:end], sm_mpg_result.densities[2:end], label="SM-MPG", linewidth=2)
display(density_plot)
bank_profile_plot = plot(shrinking.saved_t, shrinking.betas .* (180 / π), xlabel="Time (s)", ylabel="Bank Angle (deg)", title="Bank Angle Profile", label="Shrinking horizon MPC", linewidth=2)
plot!(bank_profile_plot, mpg_result.saved_t, mpg_result.betas .* (180 / π), label="MPg", linewidth=2)
plot!(bank_profile_plot, integral_mpg_result.saved_t, integral_mpg_result.betas .* (180 / π), label="Integral MPg", linewidth=2)
plot!(bank_profile_plot, open_loop_result.saved_t, open_loop_result.betas .* (180 / π), label="Open Loop", linewidth=2)
plot!(bank_profile_plot, sm_mpg_result.saved_t, sm_mpg_result.betas .* (180 / π), label="SM-MPG", linewidth=2)
display(bank_profile_plot)
heat_rate_plot = plot(shrinking.saved_t, shrinking.heat_rates, xlabel="Time (s)", ylabel="Convective Heat Rate (W/m²)", title="Convective Heat Rate Profile", label="Shrinking horizon MPC", linewidth=2)
plot!(heat_rate_plot, mpg_result.saved_t, mpg_result.heat_rates, label="MPg", linewidth=2)
plot!(heat_rate_plot, integral_mpg_result.saved_t, integral_mpg_result.heat_rates, label="Integral MPg", linewidth=2)
plot!(heat_rate_plot, open_loop_result.saved_t, open_loop_result.heat_rates, label="Open Loop", linewidth=2)
plot!(heat_rate_plot, sm_mpg_result.saved_t, sm_mpg_result.heat_rates, label="SM-MPG", linewidth=2)
heat_load_plot = plot(shrinking.sol.t, shrinking.sim_heat_loads, xlabel="Time (s)", ylabel="Convective Heat Load (J/m²)", title="Convective Heat Load Profile from State", label="Shrinking horizon MPC", linewidth=2)
plot!(heat_load_plot, mpg_result.sol.t, mpg_result.sim_heat_loads, label="MPg", linewidth=2)
plot!(heat_load_plot, integral_mpg_result.sol.t, integral_mpg_result.sim_heat_loads, label="Integral MPg", linewidth=2)
plot!(heat_load_plot, open_loop_result.sol.t, open_loop_result.sim_heat_loads, label="Open Loop", linewidth=2)
plot!(heat_load_plot, sm_mpg_result.sol.t, sm_mpg_result.sim_heat_loads, label="SM-MPG", linewidth=2)
display(plot(heat_rate_plot, heat_load_plot, layout=(2,1)))

# Compare the tracked simulations and applied controls with the optimal reference
mpc_altitude_plot = plot(
    optimal_control.Time_s,
    optimal_control.Altitude_100km .* 100.0,
    xlabel = "Time (s)",
    ylabel = "Altitude (km)",
    title = "MPC vs Optimal Reference: Altitude",
    label = "Optimal reference",
    linewidth = 2,
)
plot!(mpc_altitude_plot, shrinking.sol.t, shrinking.sim_altitudes ./ 1e3, label = "Shrinking horizon MPC", linewidth = 2)
plot!(mpc_altitude_plot, mpg_result.sol.t, mpg_result.sim_altitudes ./ 1e3, label = "MPg", linewidth = 2)
plot!(mpc_altitude_plot, integral_mpg_result.sol.t, integral_mpg_result.sim_altitudes ./ 1e3, label = "Integral MPg", linewidth = 2)
plot!(mpc_altitude_plot, open_loop_result.sol.t, open_loop_result.sim_altitudes ./ 1e3, label = "Open Loop", linewidth = 2)
plot!(mpc_altitude_plot, sm_mpg_result.sol.t, sm_mpg_result.sim_altitudes ./ 1e3, label = "SM-MPG", linewidth = 2)

mpc_velocity_plot = plot(
    optimal_control.Time_s,
    optimal_control.Velocity_1000mps .* velocity_scale,
    xlabel = "Time (s)",
    ylabel = "Velocity (m/s)",
    title = "MPC vs Optimal Reference: Velocity",
    label = "Optimal reference",
    linewidth = 2,
)
plot!(mpc_velocity_plot, shrinking.sol.t, shrinking.sim_velocities, label = "Shrinking horizon MPC", linewidth = 2)
plot!(mpc_velocity_plot, mpg_result.sol.t, mpg_result.sim_velocities, label = "MPg", linewidth = 2)
plot!(mpc_velocity_plot, integral_mpg_result.sol.t, integral_mpg_result.sim_velocities, label = "Integral MPg", linewidth = 2)
plot!(mpc_velocity_plot, open_loop_result.sol.t, open_loop_result.sim_velocities, label = "Open Loop", linewidth = 2)
plot!(mpc_velocity_plot, sm_mpg_result.sol.t, sm_mpg_result.sim_velocities, label = "SM-MPG", linewidth = 2)

mpc_fpa_plot = plot(
    optimal_control.Time_s,
    optimal_control.FlightPath_deg,
    xlabel = "Time (s)",
    ylabel = "Flight path angle (deg)",
    title = "MPC vs Optimal Reference: Flight Path",
    label = "Optimal reference",
    linewidth = 2,
)
plot!(mpc_fpa_plot, shrinking.sol.t, rad2deg.(shrinking.sim_flight_paths), label = "Shrinking horizon MPC", linewidth = 2)
plot!(mpc_fpa_plot, mpg_result.sol.t, rad2deg.(mpg_result.sim_flight_paths), label = "MPg", linewidth = 2)
plot!(mpc_fpa_plot, integral_mpg_result.sol.t, rad2deg.(integral_mpg_result.sim_flight_paths), label = "Integral MPg", linewidth = 2)
plot!(mpc_fpa_plot, open_loop_result.sol.t, rad2deg.(open_loop_result.sim_flight_paths), label = "Open Loop", linewidth = 2)
plot!(mpc_fpa_plot, sm_mpg_result.sol.t, rad2deg.(sm_mpg_result.sim_flight_paths), label = "SM-MPG", linewidth = 2)
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
    shrinking.sim_x_km,
    shrinking.sim_y_km,
    label = "Shrinking horizon MPC",
    linewidth = 2,
)
plot!(
    mpc_ground_track_plot,
    mpg_result.sim_x_km,
    mpg_result.sim_y_km,
    label = "MPg",
    linewidth = 2,
)
plot!(
    mpc_ground_track_plot,
    integral_mpg_result.sim_x_km,
    integral_mpg_result.sim_y_km,
    label = "Integral MPg",
    linewidth = 2,
)
plot!(
    mpc_ground_track_plot,
    open_loop_result.sim_x_km,
    open_loop_result.sim_y_km,
    label = "Open Loop",
    linewidth = 2,
)
plot!(
    mpc_ground_track_plot,
    sm_mpg_result.sim_x_km,
    sm_mpg_result.sim_y_km,
    label = "SM-MPG",
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
plot!(cartesian_position_plot, shrinking.sol.t, shrinking.sim_x_km, label = "MPC X", linestyle = :dash, linewidth = 2)
plot!(cartesian_position_plot, shrinking.sol.t, shrinking.sim_y_km, label = "MPC Y", linestyle = :dash, linewidth = 2)
plot!(cartesian_position_plot, shrinking.sol.t, shrinking.sim_z_km, label = "MPC Z", linestyle = :dash, linewidth = 2)
plot!(cartesian_position_plot, mpg_result.sol.t, mpg_result.sim_x_km, label = "MPg X", linestyle = :dot, linewidth = 2)
plot!(cartesian_position_plot, mpg_result.sol.t, mpg_result.sim_y_km, label = "MPg Y", linestyle = :dot, linewidth = 2)
plot!(cartesian_position_plot, mpg_result.sol.t, mpg_result.sim_z_km, label = "MPg Z", linestyle = :dot, linewidth = 2)
plot!(cartesian_position_plot, integral_mpg_result.sol.t, integral_mpg_result.sim_x_km, label = "Integral MPg X", linewidth = 2)
plot!(cartesian_position_plot, integral_mpg_result.sol.t, integral_mpg_result.sim_y_km, label = "Integral MPg Y", linewidth = 2)
plot!(cartesian_position_plot, integral_mpg_result.sol.t, integral_mpg_result.sim_z_km, label = "Integral MPg Z", linewidth = 2)
plot!(cartesian_position_plot, open_loop_result.sol.t, open_loop_result.sim_x_km, label = "Open Loop X", linestyle = :dashdot, linewidth = 2)
plot!(cartesian_position_plot, open_loop_result.sol.t, open_loop_result.sim_y_km, label = "Open Loop Y", linestyle = :dashdot, linewidth = 2)
plot!(cartesian_position_plot, open_loop_result.sol.t, open_loop_result.sim_z_km, label = "Open Loop Z", linestyle = :dashdot, linewidth = 2)
plot!(cartesian_position_plot, sm_mpg_result.sol.t, sm_mpg_result.sim_x_km, label = "SM-MPG X", linestyle = :solid, linewidth = 2)
plot!(cartesian_position_plot, sm_mpg_result.sol.t, sm_mpg_result.sim_y_km, label = "SM-MPG Y", linestyle = :solid, linewidth = 2)
plot!(cartesian_position_plot, sm_mpg_result.sol.t, sm_mpg_result.sim_z_km, label = "SM-MPG Z", linestyle = :solid, linewidth = 2)
display(cartesian_position_plot)

position_error_plot = plot(
    shrinking.sol.t,
    shrinking.position_error_norm_km,
    xlabel = "Time (s)",
    ylabel = "Position error (km)",
    title = "Cartesian Position Error Norm",
    label = "Shrinking horizon MPC",
    linewidth = 2,
)
plot!(position_error_plot, mpg_result.sol.t, mpg_result.position_error_norm_km, label = "MPg", linewidth = 2)
plot!(position_error_plot, integral_mpg_result.sol.t, integral_mpg_result.position_error_norm_km, label = "Integral MPg", linewidth = 2)
plot!(position_error_plot, open_loop_result.sol.t, open_loop_result.position_error_norm_km, label = "Open Loop", linewidth = 2)
plot!(position_error_plot, sm_mpg_result.sol.t, sm_mpg_result.position_error_norm_km, label = "SM-MPG", linewidth = 2)
display(position_error_plot)

if !isempty(shrinking.saved_t) || !isempty(mpg_result.saved_t) || !isempty(integral_mpg_result.saved_t) || !isempty(sm_mpg_result.saved_t)
    mpc_alpha_plot = plot(
        optimal_control.Time_s,
        optimal_control.AngleOfAttack_deg,
        xlabel = "Time (s)",
        ylabel = "Angle of attack (deg)",
        title = "MPC vs Optimal Reference: Angle of Attack",
        label = "Optimal reference",
        linewidth = 2,
    )
    plot!(mpc_alpha_plot, shrinking.saved_t, rad2deg.(shrinking.alphas), label = "Shrinking horizon MPC", linewidth = 2)
    plot!(mpc_alpha_plot, mpg_result.saved_t, rad2deg.(mpg_result.alphas), label = "MPg", linewidth = 2)
    plot!(mpc_alpha_plot, integral_mpg_result.saved_t, rad2deg.(integral_mpg_result.alphas), label = "Integral MPg", linewidth = 2)
    plot!(mpc_alpha_plot, open_loop_result.saved_t, rad2deg.(open_loop_result.alphas), label = "Open Loop", linewidth = 2)
    plot!(mpc_alpha_plot, sm_mpg_result.saved_t, rad2deg.(sm_mpg_result.alphas), label = "SM-MPG", linewidth = 2)

    mpc_bank_plot = plot(
        optimal_control.Time_s,
        optimal_control.BankAngle_deg,
        xlabel = "Time (s)",
        ylabel = "Bank angle (deg)",
        title = "MPC vs Optimal Reference: Bank Angle",
        label = "Optimal reference",
        linewidth = 2,
    )
    plot!(mpc_bank_plot, shrinking.saved_t, rad2deg.(shrinking.betas), label = "Shrinking horizon MPC", linewidth = 2)
    plot!(mpc_bank_plot, mpg_result.saved_t, rad2deg.(mpg_result.betas), label = "MPg", linewidth = 2)
    plot!(mpc_bank_plot, integral_mpg_result.saved_t, rad2deg.(integral_mpg_result.betas), label = "Integral MPg", linewidth = 2)
    plot!(mpc_bank_plot, open_loop_result.saved_t, rad2deg.(open_loop_result.betas), label = "Open Loop", linewidth = 2)
    plot!(mpc_bank_plot, sm_mpg_result.saved_t, rad2deg.(sm_mpg_result.betas), label = "SM-MPG", linewidth = 2)
    display(plot(mpc_alpha_plot, mpc_bank_plot, layout = (2, 1), size = (900, 650)))
end

mpg_opt_states = mpg_params.optimization_states
integral_mpg_opt_states = integral_mpg_params.optimization_states
sm_mpg_opt_states = sm_mpg_params.optimization_states
if !isempty(mpg_opt_states.h_c) || !isempty(integral_mpg_opt_states.h_c) || !isempty(sm_mpg_opt_states.h_c)
    latest_prediction_plot = plot(
        optimal_control.Time_s,
        optimal_control.Altitude_100km .* 100.0,
        xlabel = "Time (s)",
        ylabel = "Altitude (km)",
        title = "Latest MPC Horizon vs Optimal Reference",
        label = "Optimal reference",
        linewidth = 2,
    )
    plot!(latest_prediction_plot, shrinking.sol.t, shrinking.sim_altitudes ./ 1e3, label = "Shrinking horizon MPC", linewidth = 2)
    plot!(latest_prediction_plot, mpg_result.sol.t, mpg_result.sim_altitudes ./ 1e3, label = "MPg", linewidth = 2)
    plot!(latest_prediction_plot, integral_mpg_result.sol.t, integral_mpg_result.sim_altitudes ./ 1e3, label = "Integral MPg", linewidth = 2)
    plot!(latest_prediction_plot, open_loop_result.sol.t, open_loop_result.sim_altitudes ./ 1e3, label = "Open Loop", linewidth = 2)
    plot!(latest_prediction_plot, sm_mpg_result.sol.t, sm_mpg_result.sim_altitudes ./ 1e3, label = "SM-MPG", linewidth = 2)
    if !isempty(mpg_opt_states.h_c)
        mpc_prediction_times = mpg_result.sol.t[end] .+ (1:length(mpg_opt_states.h_c)) .* mpg_params.mpc_params.time_step
        plot!(
            latest_prediction_plot,
            mpc_prediction_times,
            mpg_opt_states.h_c ./ 1e3,
            label = "Latest MPg prediction",
            linestyle = :dot,
            linewidth = 2,
        )
    end
    if !isempty(integral_mpg_opt_states.h_c)
        integral_mpg_prediction_times = integral_mpg_result.sol.t[end] .+ (1:length(integral_mpg_opt_states.h_c)) .* integral_mpg_params.mpc_params.time_step
        plot!(
            latest_prediction_plot,
            integral_mpg_prediction_times,
            integral_mpg_opt_states.h_c ./ 1e3,
            label = "Latest Integral MPg prediction",
            linestyle = :dash,
            linewidth = 2,
        )
    end
    if !isempty(sm_mpg_opt_states.h_c)
        sm_mpg_prediction_times = sm_mpg_result.sol.t[end] .+ (1:length(sm_mpg_opt_states.h_c)) .* sm_mpg_params.mpc_params.time_step
        plot!(
            latest_prediction_plot,
            sm_mpg_prediction_times,
            sm_mpg_opt_states.h_c ./ 1e3,
            label = "Latest SM-MPG prediction",
            linestyle = :dashdot,
            linewidth = 2,
        )
    end
    display(latest_prediction_plot)
end

function plot_combined_state_errors(results_labels...)
    p1 = plot(title="Altitude Error (m)", xlabel="Time (s)", legend=true)
    p2 = plot(title="Longitude Error (deg)", xlabel="Time (s)", legend=true)
    p3 = plot(title="Latitude Error (deg)", xlabel="Time (s)", legend=true)
    p4 = plot(title="Velocity Error (m/s)", xlabel="Time (s)", legend=true)
    p5 = plot(title="FPA Error (deg)", xlabel="Time (s)", legend=true)
    p6 = plot(title="Azimuth Error (deg)", xlabel="Time (s)", legend=true)

    for (res, label) in results_labels
        df = res.df
        t = df.Time_s
        plot!(p1, t, df.Altitude_m .- df.ReferenceAltitude_m, label=label, linewidth=2)
        plot!(p2, t, df.Longitude_deg .- df.ReferenceLongitude_deg, label=label, linewidth=2)
        plot!(p3, t, df.Latitude_deg .- df.ReferenceLatitude_deg, label=label, linewidth=2)
        plot!(p4, t, df.Velocity_mps .- df.ReferenceVelocity_mps, label=label, linewidth=2)
        plot!(p5, t, df.FlightPath_deg .- df.ReferenceFlightPath_deg, label=label, linewidth=2)
        plot!(p6, t, df.Azimuth_deg .- df.ReferenceAzimuth_deg, label=label, linewidth=2)
    end

    fig = plot(p1, p2, p3, p4, p5, p6, layout=(3, 2), size=(1200, 1000), margin=5Plots.mm, plot_title="State Tracking Errors")
    display(fig)
    return fig
end

plot_combined_state_errors(
    (shrinking, "Shrinking horizon MPC"), 
    (mpg_result, "MPg"),
    (integral_mpg_result, "Integral MPg"),
    (open_loop_result, "Open Loop"),
    (sm_mpg_result, "SM-MPG")
)
