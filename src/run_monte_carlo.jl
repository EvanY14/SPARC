include("model/SimulatorModel.jl")

using .SimulatorModel
using StaticArrays
ENV["GKSwstype"] = get(ENV, "GKSwstype", "100")
using Plots
using DifferentialEquations
using ProgressMeter
using CSV
using DataFrames
using Interpolations
include("reference/trajectory_initialization.jl")

gr()

const NUM_SIMULATIONS = parse(Int, get(ENV, "SPARC_MC_RUNS", "50"))
const OUTPUT_DIR = get(ENV, "SPARC_MC_OUTPUT_DIR", "monte_carlo_output")
const RUN_NOMINAL_OVERLAY = parse(Bool, get(ENV, "SPARC_MC_NOMINAL_OVERLAY", "true"))
const DISPLAY_PLOTS = parse(Bool, get(ENV, "SPARC_MC_DISPLAY", "false"))
const DATA_DIR = joinpath(OUTPUT_DIR, "data")
const PLOTS_DIR = joinpath(OUTPUT_DIR, "plots")

struct ControllerCase
    name::String
    slug::String
    control_function::Function
    color::Symbol
end

function open_loop_control(integrator)
    t = integrator.t
    β_cmd = deg2rad(interp_optimal_control(t))
    α_cmd = deg2rad(interp_optimal_alpha(t))
    return β_cmd, α_cmd
end

const CONTROLLER_CASES = (
    ControllerCase("Tracking MPC", "tracking_mpc", SimulatorModel.trackingmpc_shrinking, :blue),
    ControllerCase("MPG", "mpg", SimulatorModel.mpg, :green),
    ControllerCase("Open Loop", "open_loop", open_loop_control, :red),
)

mutable struct MonteCarloResults
    case::ControllerCase
    final_latitudes::Vector{Float64}
    final_longitudes::Vector{Float64}
    final_altitudes::Vector{Float64}
    final_velocities::Vector{Float64}
    final_heat_loads::Vector{Float64}
    final_x_positions::Vector{Float64}
    final_y_positions::Vector{Float64}
    final_z_positions::Vector{Float64}
    altitude_profiles::Vector{Union{Nothing, Vector{Float64}}}
    longitude_profiles::Vector{Union{Nothing, Vector{Float64}}}
    latitude_profiles::Vector{Union{Nothing, Vector{Float64}}}
    velocity_profiles::Vector{Union{Nothing, Vector{Float64}}}
    flight_path_profiles::Vector{Union{Nothing, Vector{Float64}}}
    azimuth_profiles::Vector{Union{Nothing, Vector{Float64}}}
    x_profiles::Vector{Union{Nothing, Vector{Float64}}}
    y_profiles::Vector{Union{Nothing, Vector{Float64}}}
    z_profiles::Vector{Union{Nothing, Vector{Float64}}}
    alpha_profiles::Vector{Union{Nothing, Vector{Float64}}}
    beta_profiles::Vector{Union{Nothing, Vector{Float64}}}
    heat_rate_profiles::Vector{Union{Nothing, Vector{Float64}}}
    times::Vector{Union{Nothing, Vector{Float64}}}
    control_times::Vector{Union{Nothing, Vector{Float64}}}
end

function MonteCarloResults(case::ControllerCase, n::Int)
    profiles = Vector{Union{Nothing, Vector{Float64}}}(undef, n)
    longitude_profiles = Vector{Union{Nothing, Vector{Float64}}}(undef, n)
    latitude_profiles = Vector{Union{Nothing, Vector{Float64}}}(undef, n)
    velocity_profiles = Vector{Union{Nothing, Vector{Float64}}}(undef, n)
    flight_path_profiles = Vector{Union{Nothing, Vector{Float64}}}(undef, n)
    azimuth_profiles = Vector{Union{Nothing, Vector{Float64}}}(undef, n)
    x_profiles = Vector{Union{Nothing, Vector{Float64}}}(undef, n)
    y_profiles = Vector{Union{Nothing, Vector{Float64}}}(undef, n)
    z_profiles = Vector{Union{Nothing, Vector{Float64}}}(undef, n)
    alpha_profiles = Vector{Union{Nothing, Vector{Float64}}}(undef, n)
    beta_profiles = Vector{Union{Nothing, Vector{Float64}}}(undef, n)
    heat_rate_profiles = Vector{Union{Nothing, Vector{Float64}}}(undef, n)
    times = Vector{Union{Nothing, Vector{Float64}}}(undef, n)
    control_times = Vector{Union{Nothing, Vector{Float64}}}(undef, n)
    fill!(profiles, nothing)
    fill!(longitude_profiles, nothing)
    fill!(latitude_profiles, nothing)
    fill!(velocity_profiles, nothing)
    fill!(flight_path_profiles, nothing)
    fill!(azimuth_profiles, nothing)
    fill!(x_profiles, nothing)
    fill!(y_profiles, nothing)
    fill!(z_profiles, nothing)
    fill!(alpha_profiles, nothing)
    fill!(beta_profiles, nothing)
    fill!(heat_rate_profiles, nothing)
    fill!(times, nothing)
    fill!(control_times, nothing)
    return MonteCarloResults(
        case,
        fill(NaN, n),
        fill(NaN, n),
        fill(NaN, n),
        fill(NaN, n),
        fill(NaN, n),
        fill(NaN, n),
        fill(NaN, n),
        fill(NaN, n),
        profiles,
        longitude_profiles,
        latitude_profiles,
        velocity_profiles,
        flight_path_profiles,
        azimuth_profiles,
        x_profiles,
        y_profiles,
        z_profiles,
        alpha_profiles,
        beta_profiles,
        heat_rate_profiles,
        times,
        control_times,
    )
end

mass = SimulatorModel.VEHICLE.mass
area = SimulatorModel.VEHICLE.reference_area
const μ = 3.986004418e14
R = 6378137.0

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

interp_optimal_control = linear_interpolation(optimal_control.Time_s, optimal_control.BankAngle_deg, extrapolation_bc=Line())
interp_optimal_alpha = linear_interpolation(optimal_control.Time_s, optimal_control.AngleOfAttack_deg, extrapolation_bc=Line())

velocity_scale = abs(optimal_control.Velocity_1000mps[1] * 1e3 - v0) <= abs(optimal_control.Velocity_1000mps[1] * 1e4 - v0) ? 1e3 : 1e4
interp_altitude = linear_interpolation(optimal_control.Time_s, optimal_control.Altitude_100km .* 1e5, extrapolation_bc=Line())
interp_velocity = linear_interpolation(optimal_control.Time_s, optimal_control.Velocity_1000mps .* velocity_scale, extrapolation_bc=Line())
interp_longitude = linear_interpolation(optimal_control.Time_s, optimal_control.Longitude_deg .* (π / 180), extrapolation_bc=Line())
interp_latitude = linear_interpolation(optimal_control.Time_s, optimal_control.Latitude_deg .* (π / 180), extrapolation_bc=Line())
interp_flight_path = linear_interpolation(optimal_control.Time_s, optimal_control.FlightPath_deg .* (π / 180), extrapolation_bc=Line())
interp_azimuth = linear_interpolation(optimal_control.Time_s, optimal_control.Azimuth_deg .* (π / 180), extrapolation_bc=Line())
optimal_trajectory = SVector{6, AbstractInterpolation}(interp_altitude, interp_longitude, interp_latitude, interp_velocity, interp_flight_path, interp_azimuth)

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

optimal_ref_x_km, optimal_ref_y_km, optimal_ref_z_km = cartesian_histories_km(
    optimal_control.Altitude_100km .* 1e5,
    deg2rad.(optimal_control.Longitude_deg),
    deg2rad.(optimal_control.Latitude_deg),
    R,
)

target_altitude = 11848.0
target_velocity = 500.0
target_γ = deg2rad(-5.0)
target_states = SimulatorModel.TargetStates(altitude=target_altitude, longitude=deg2rad(133.4), latitude=deg2rad(-4.2), velocity=target_velocity, flight_path_angle=target_γ)

function build_edl_params(control_function::Function; disturbance::Bool, monte_carlo::Bool)
    atmosphere = SimulatorModel.GramAtmosphere("GRAMpy/", "GRAM_Data", monte_carlo, "earth", SimulatorModel.DateTime(2012, 8, 6, 5, 10, 46.0))
    mpc_params = SimulatorModel.MPCParams{100, 7, 8, 0.1}(
        n_horizon=100,
        time_step=0.75,
        H_SCALE=1.0e5,
        V_SCALE=1.0e4,
        T_SCALE=1.0,
        n_exp=4.512,
        m_exp=0.82958,
        learning_rate=0.1,
    )
    return SimulatorModel.EDLParams(
        mass=mass,
        area=area,
        μ=μ,
        R=R,
        control_function=control_function,
        β=deg2rad(interp_optimal_control(tspan[1])),
        α=deg2rad(interp_optimal_alpha(tspan[1])),
        atmospheric_density_function=(LatLonAlt, t) -> SimulatorModel.atmospheric_density(LatLonAlt, t, atmosphere, disturbance),
        atmospheric_density=0.0,
        wind=SVector{3, Float64}(zeros(3)),
        target_states=target_states,
        optimization_states=SimulatorModel.OptimizationStates(),
        nominal_trajectory=optimal_trajectory,
        cache=SimulatorModel.EDLCache(),
        mpc_params=mpc_params,
    )
end

function run_case(control_function::Function; disturbance::Bool, monte_carlo::Bool, dt::Float64)
    local_saved_values = SavedValues(Float64, Tuple{Float64, Float64, Float64, Float64})
    local_saving_callback = SavingCallback(
        (u, t, integrator) -> (
            integrator.p.atmospheric_density,
            integrator.p.β,
            integrator.p.α,
            integrator.p.cache.q_dot,
        ),
        local_saved_values,
        saveat=0.1,
    )
    callbacks = CallbackSet(SimulatorModel.altitude_termination_condition, monte_carlo_atmospheric_density_callback, local_saving_callback, SimulatorModel.control_callback)
    params = build_edl_params(control_function, disturbance=disturbance, monte_carlo=monte_carlo)
    prob = ODEProblem(SimulatorModel.edl_dynamics, copy(u0), tspan, params, callback=callbacks)
    sol = solve(prob, Tsit5(), dt=dt, adaptive=false)
    return sol, local_saved_values
end

function saved_value_vectors(local_saved_values)
    saved_data = local_saved_values.saveval
    densities = zeros(length(saved_data))
    betas = zeros(length(saved_data))
    alphas = zeros(length(saved_data))
    heat_rates = zeros(length(saved_data))
    for i in eachindex(saved_data)
        densities[i] = saved_data[i][1]
        betas[i] = saved_data[i][2]
        alphas[i] = saved_data[i][3]
        heat_rates[i] = saved_data[i][4]
    end
    return densities, betas, alphas, heat_rates
end

function monte_carlo_atmospheric_density_effect!(integrator)
    h = integrator.u[1]
    ϕ = integrator.u[2]
    θ = integrator.u[3]
    if !all(isfinite, integrator.u) || h <= integrator.p.target_states.altitude || integrator.p.R + h <= 0.0 || abs(θ) >= deg2rad(89.9) || !isfinite(ϕ)
        terminate!(integrator)
        return
    end

    density_function = integrator.p.atmospheric_density_function
    LatLonAlt = (θ, ϕ, h)
    integrator.p.atmospheric_density, integrator.p.wind = density_function(LatLonAlt, integrator.t)
end

monte_carlo_atmospheric_density_callback = DiscreteCallback((u, t, integrator) -> true, monte_carlo_atmospheric_density_effect!)

function valid_indices(result::MonteCarloResults)
    return findall(
        i -> result.times[i] !== nothing &&
            result.altitude_profiles[i] !== nothing &&
            result.velocity_profiles[i] !== nothing,
        eachindex(result.times),
    )
end

function write_case_summary(result::MonteCarloResults, data_dir::AbstractString)
    summary_df = DataFrame(
        Controller=fill(result.case.name, NUM_SIMULATIONS),
        Simulation=collect(1:NUM_SIMULATIONS),
        Completed=.!isnan.(result.final_latitudes),
        FinalLongitude_deg=result.final_longitudes,
        FinalLatitude_deg=result.final_latitudes,
        FinalAltitude_km=result.final_altitudes,
        FinalVelocity_kms=result.final_velocities,
        FinalHeatLoad_Jm2=result.final_heat_loads,
        FinalPositionX_km=result.final_x_positions,
        FinalPositionY_km=result.final_y_positions,
        FinalPositionZ_km=result.final_z_positions,
    )
    CSV.write(joinpath(data_dir, "summary.csv"), summary_df)
    return summary_df
end

function write_case_output!(result::MonteCarloResults, sim::Int, sol_mc, local_saved_values, data_dir::AbstractString)
    altitudes_m = getindex.(sol_mc.u, 1)
    longitudes_rad = getindex.(sol_mc.u, 2)
    latitudes_rad = getindex.(sol_mc.u, 3)
    x_km, y_km, z_km = cartesian_histories_km(altitudes_m, longitudes_rad, latitudes_rad, R)

    result.altitude_profiles[sim] = altitudes_m ./ 1e3
    result.longitude_profiles[sim] = rad2deg.(longitudes_rad)
    result.latitude_profiles[sim] = rad2deg.(latitudes_rad)
    result.velocity_profiles[sim] = getindex.(sol_mc.u, 4) ./ 1e3
    result.flight_path_profiles[sim] = rad2deg.(getindex.(sol_mc.u, 5))
    result.azimuth_profiles[sim] = rad2deg.(getindex.(sol_mc.u, 6))
    result.x_profiles[sim] = x_km
    result.y_profiles[sim] = y_km
    result.z_profiles[sim] = z_km
    result.times[sim] = sol_mc.t
    result.final_latitudes[sim] = rad2deg(getindex(sol_mc.u[end], 3))
    result.final_longitudes[sim] = rad2deg(getindex(sol_mc.u[end], 2))
    result.final_altitudes[sim] = getindex(sol_mc.u[end], 1) / 1e3
    result.final_velocities[sim] = getindex(sol_mc.u[end], 4) / 1e3
    result.final_heat_loads[sim] = getindex(sol_mc.u[end], 7)
    result.final_x_positions[sim] = x_km[end]
    result.final_y_positions[sim] = y_km[end]
    result.final_z_positions[sim] = z_km[end]

    _, betas, alphas, heat_rates = saved_value_vectors(local_saved_values)
    result.alpha_profiles[sim] = rad2deg.(alphas)
    result.beta_profiles[sim] = rad2deg.(betas)
    result.heat_rate_profiles[sim] = heat_rates
    result.control_times[sim] = local_saved_values.t

    trajectory_df = DataFrame(
        Controller=fill(result.case.name, length(sol_mc.t)),
        Simulation=fill(sim, length(sol_mc.t)),
        Time_s=sol_mc.t,
        Altitude_km=result.altitude_profiles[sim],
        Longitude_deg=rad2deg.(longitudes_rad),
        Latitude_deg=rad2deg.(latitudes_rad),
        PositionX_km=x_km,
        PositionY_km=y_km,
        PositionZ_km=z_km,
        Velocity_kms=result.velocity_profiles[sim],
        FlightPath_deg=rad2deg.(getindex.(sol_mc.u, 5)),
        Azimuth_deg=rad2deg.(getindex.(sol_mc.u, 6)),
        HeatLoad_Jm2=getindex.(sol_mc.u, 7),
    )
    CSV.write(joinpath(data_dir, "trajectory_$(sim).csv"), trajectory_df)

    control_df = DataFrame(
        Controller=fill(result.case.name, length(local_saved_values.t)),
        Simulation=fill(sim, length(local_saved_values.t)),
        Time_s=local_saved_values.t,
        AngleOfAttack_deg=result.alpha_profiles[sim],
        BankAngle_deg=result.beta_profiles[sim],
        HeatRate_Wm2=heat_rates,
    )
    CSV.write(joinpath(data_dir, "control_$(sim).csv"), control_df)
end

function run_monte_carlo_case(case::ControllerCase)
    data_dir = joinpath(DATA_DIR, case.slug)
    mkpath(data_dir)

    result = MonteCarloResults(case, NUM_SIMULATIONS)
    nominal_sol = nothing
    if RUN_NOMINAL_OVERLAY
        nominal_sol, _ = run_case(case.control_function; disturbance=false, monte_carlo=false, dt=0.1)
    end

    println("Running $(case.name) Monte Carlo cases")
    @showprogress for sim in 1:NUM_SIMULATIONS
        try
            sol_mc, local_saved_values = run_case(case.control_function; disturbance=true, monte_carlo=true, dt=0.5)
            write_case_output!(result, sim, sol_mc, local_saved_values, data_dir)
        catch e
            @warn "Monte Carlo case failed" controller=case.name sim exception=(e, catch_backtrace())
        end
    end

    summary_df = write_case_summary(result, data_dir)
    return result, nominal_sol, summary_df
end

function plot_profiles(results, nominal_solutions, profile_field::Symbol; title::String, ylabel::String, filename::String)
    plt = plot(title=title, xlabel="Time (s)", ylabel=ylabel, legend=true)
    for (result, nominal_sol) in zip(results, nominal_solutions)
        valid = valid_indices(result)
        for (j, i) in enumerate(valid)
            profiles = getfield(result, profile_field)
            plot!(
                plt,
                something(result.times[i]),
                something(profiles[i]),
                color=result.case.color,
                alpha=0.25,
                label=j == 1 ? "$(result.case.name) MC" : false,
            )
        end
        if nominal_sol !== nothing
            nominal_y = profile_field == :altitude_profiles ? getindex.(nominal_sol.u, 1) ./ 1e3 : getindex.(nominal_sol.u, 4) ./ 1e3
            plot!(plt, nominal_sol.t, nominal_y, color=result.case.color, linewidth=2, linestyle=:dash, label="$(result.case.name) nominal")
        end
    end
    if DISPLAY_PLOTS
        display(plt)
    end
    savefig(plt, joinpath(PLOTS_DIR, filename))
end

function plot_control_profiles(results, profile_field::Symbol; title::String, ylabel::String, filename::String)
    plt = plot(title=title, xlabel="Time (s)", ylabel=ylabel, legend=true)
    for result in results
        control_profiles = getfield(result, profile_field)
        valid = findall(i -> result.control_times[i] !== nothing && control_profiles[i] !== nothing, eachindex(result.control_times))
        for (j, i) in enumerate(valid)
            plot!(
                plt,
                something(result.control_times[i]),
                something(control_profiles[i]),
                color=result.case.color,
                alpha=0.25,
                label=j == 1 ? "$(result.case.name) MC" : false,
            )
        end
    end
    if DISPLAY_PLOTS
        display(plt)
    end
    savefig(plt, joinpath(PLOTS_DIR, filename))
end

function plot_landing_locations(results, nominal_solutions)
    plt = plot(
        xlabel="Longitude (deg)",
        ylabel="Latitude (deg)",
        title="Monte Carlo Simulations: Final Landing Locations",
        legend=true,
    )
    for (result, nominal_sol) in zip(results, nominal_solutions)
        valid = valid_indices(result)
        plot!(
            plt,
            result.final_longitudes[valid],
            result.final_latitudes[valid],
            seriestype=:scatter,
            color=result.case.color,
            label="$(result.case.name) landings",
            alpha=0.65,
        )
        if nominal_sol !== nothing
            plot!(
                plt,
                [rad2deg(getindex(nominal_sol.u[end], 2))],
                [rad2deg(getindex(nominal_sol.u[end], 3))],
                seriestype=:scatter,
                color=result.case.color,
                markershape=:star5,
                markersize=8,
                label="$(result.case.name) nominal",
            )
        end
    end
    plot!(plt, [rad2deg(target_states.longitude)], [rad2deg(target_states.latitude)], seriestype=:scatter, color=:black, markershape=:diamond, markersize=8, label="Target")
    if DISPLAY_PLOTS
        display(plt)
    end
    savefig(plt, joinpath(PLOTS_DIR, "monte_carlo_final_landing_locations.pdf"))
end

function plot_cartesian_ground_track(results, nominal_solutions)
    plt = plot(
        optimal_ref_x_km,
        optimal_ref_y_km,
        xlabel="X (km)",
        ylabel="Y (km)",
        title="Monte Carlo Simulations: Cartesian Ground Track",
        label="Reference",
        color=:black,
        linewidth=2,
        legend=true,
    )
    for (result, nominal_sol) in zip(results, nominal_solutions)
        valid = findall(i -> result.x_profiles[i] !== nothing && result.y_profiles[i] !== nothing, eachindex(result.x_profiles))
        for (j, i) in enumerate(valid)
            plot!(
                plt,
                something(result.x_profiles[i]),
                something(result.y_profiles[i]),
                color=result.case.color,
                alpha=0.25,
                label=j == 1 ? "$(result.case.name) MC" : false,
            )
        end
        if nominal_sol !== nothing
            nominal_x, nominal_y, _ = cartesian_histories_km(
                getindex.(nominal_sol.u, 1),
                getindex.(nominal_sol.u, 2),
                getindex.(nominal_sol.u, 3),
                R,
            )
            plot!(plt, nominal_x, nominal_y, color=result.case.color, linewidth=2, linestyle=:dash, label="$(result.case.name) nominal")
        end
    end
    target_x, target_y, _ = latlonalt_to_cartesian_km(target_states.latitude, target_states.longitude, target_states.altitude, R)
    plot!(plt, [target_x], [target_y], seriestype=:scatter, color=:black, markershape=:diamond, markersize=8, label="Target")
    if DISPLAY_PLOTS
        display(plt)
    end
    savefig(plt, joinpath(PLOTS_DIR, "monte_carlo_cartesian_ground_track.pdf"))
end

function plot_final_cartesian_locations(results, nominal_solutions)
    plt = plot(
        xlabel="X (km)",
        ylabel="Y (km)",
        title="Monte Carlo Simulations: Final Cartesian Locations",
        legend=true,
    )
    for (result, nominal_sol) in zip(results, nominal_solutions)
        valid = findall(i -> isfinite(result.final_x_positions[i]) && isfinite(result.final_y_positions[i]), eachindex(result.final_x_positions))
        plot!(
            plt,
            result.final_x_positions[valid],
            result.final_y_positions[valid],
            seriestype=:scatter,
            color=result.case.color,
            label="$(result.case.name) final positions",
            alpha=0.65,
        )
        if nominal_sol !== nothing
            nominal_x, nominal_y, _ = latlonalt_to_cartesian_km(
                getindex(nominal_sol.u[end], 3),
                getindex(nominal_sol.u[end], 2),
                getindex(nominal_sol.u[end], 1),
                R,
            )
            plot!(
                plt,
                [nominal_x],
                [nominal_y],
                seriestype=:scatter,
                color=result.case.color,
                markershape=:star5,
                markersize=8,
                label="$(result.case.name) nominal",
            )
        end
    end
    target_x, target_y, _ = latlonalt_to_cartesian_km(target_states.latitude, target_states.longitude, target_states.altitude, R)
    plot!(plt, [target_x], [target_y], seriestype=:scatter, color=:black, markershape=:diamond, markersize=8, label="Target")
    if DISPLAY_PLOTS
        display(plt)
    end
    savefig(plt, joinpath(PLOTS_DIR, "monte_carlo_final_cartesian_locations.pdf"))
end

mkpath(DATA_DIR)
mkpath(PLOTS_DIR)

case_outputs = []
for case in CONTROLLER_CASES
    push!(case_outputs, run_monte_carlo_case(case))
end
results = first.(case_outputs)
nominal_solutions = getindex.(case_outputs, 2)
summary_dfs = getindex.(case_outputs, 3)
CSV.write(joinpath(DATA_DIR, "summary.csv"), vcat(summary_dfs...))

plot_profiles(
    results,
    nominal_solutions,
    :altitude_profiles;
    title="Monte Carlo Simulations: Altitude Profiles",
    ylabel="Altitude (km)",
    filename="monte_carlo_altitude_profiles.pdf",
)
plot_profiles(
    results,
    nominal_solutions,
    :velocity_profiles;
    title="Monte Carlo Simulations: Velocity Profiles",
    ylabel="Velocity (km/s)",
    filename="monte_carlo_velocity_profiles.pdf",
)
plot_landing_locations(results, nominal_solutions)
plot_cartesian_ground_track(results, nominal_solutions)
plot_final_cartesian_locations(results, nominal_solutions)
plot_control_profiles(
    results,
    :alpha_profiles;
    title="Monte Carlo Simulations: Angle of Attack Profiles",
    ylabel="Angle of Attack (deg)",
    filename="monte_carlo_angle_of_attack_profiles.pdf",
)
plot_control_profiles(
    results,
    :beta_profiles;
    title="Monte Carlo Simulations: Bank Angle Profiles",
    ylabel="Bank Angle (deg)",
    filename="monte_carlo_bank_angle_profiles.pdf",
)
plot_control_profiles(
    results,
    :heat_rate_profiles;
    title="Monte Carlo Simulations: Heat Rate Profiles",
    ylabel="Heat Rate (W/m²)",
    filename="monte_carlo_heat_rate_profiles.pdf",
)

function plot_mc_state_errors(results)
    plt_alt = plot(title="Altitude Error (km)", xlabel="Time (s)", ylabel="Error (km)", legend=false)
    plt_lon = plot(title="Longitude Error (deg)", xlabel="Time (s)", ylabel="Error (deg)", legend=false)
    plt_lat = plot(title="Latitude Error (deg)", xlabel="Time (s)", ylabel="Error (deg)", legend=false)
    plt_vel = plot(title="Velocity Error (km/s)", xlabel="Time (s)", ylabel="Error (km/s)", legend=false)
    plt_fpa = plot(title="FPA Error (deg)", xlabel="Time (s)", ylabel="Error (deg)", legend=false)
    plt_azi = plot(title="Azimuth Error (deg)", xlabel="Time (s)", ylabel="Error (deg)", legend=false)
    
    for result in results
        valid = valid_indices(result)
        for (j, i) in enumerate(valid)
            t = something(result.times[i])
            
            alt_err = something(result.altitude_profiles[i]) .- (interp_altitude.(t) ./ 1e3)
            plot!(plt_alt, t, alt_err, color=result.case.color, alpha=0.25)
            
            lon_err = something(result.longitude_profiles[i]) .- rad2deg.(interp_longitude.(t))
            plot!(plt_lon, t, lon_err, color=result.case.color, alpha=0.25)
            
            lat_err = something(result.latitude_profiles[i]) .- rad2deg.(interp_latitude.(t))
            plot!(plt_lat, t, lat_err, color=result.case.color, alpha=0.25)
            
            vel_err = something(result.velocity_profiles[i]) .- (interp_velocity.(t) ./ 1e3)
            plot!(plt_vel, t, vel_err, color=result.case.color, alpha=0.25)
            
            fpa_err = something(result.flight_path_profiles[i]) .- rad2deg.(interp_flight_path.(t))
            plot!(plt_fpa, t, fpa_err, color=result.case.color, alpha=0.25)
            
            azi_err = something(result.azimuth_profiles[i]) .- rad2deg.(interp_azimuth.(t))
            plot!(plt_azi, t, azi_err, color=result.case.color, alpha=0.25)
        end
    end
    
    fig = plot(plt_alt, plt_lon, plt_lat, plt_vel, plt_fpa, plt_azi, layout=(3, 2), size=(1200, 1000), margin=5Plots.mm, plot_title="Monte Carlo State Errors")
    if DISPLAY_PLOTS
        display(fig)
    end
    savefig(fig, joinpath(PLOTS_DIR, "monte_carlo_state_errors.pdf"))
end

plot_mc_state_errors(results)
