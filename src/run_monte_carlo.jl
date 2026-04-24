include("model/SimulatorModel.jl")

using .SimulatorModel
using StaticArrays
using PythonCall
using LinearAlgebra
using Statistics
ENV["GKSwstype"] = get(ENV, "GKSwstype", "100")
using Plots
using DifferentialEquations
using ProgressMeter
using CSV
using DataFrames
using Interpolations
include("reference/trajectory_initialization.jl")

plotly()

const NUM_SIMULATIONS = parse(Int, get(ENV, "SPARC_MC_RUNS", "100"))
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
    ControllerCase("MPG", "mpg", SimulatorModel.mpg, :green),
    ControllerCase("MPG + Integral Action", "mpg_integral", SimulatorModel.mpg_integral, :orange),
    ControllerCase("SM-MPG", "sm_mpg", SimulatorModel.sm_mpg, :purple),
    ControllerCase("Open Loop", "open_loop", open_loop_control, :red),
)
const GRAM_DIRECTORY = "GRAMpy/"
const GRAM_DATA_DIRECTORY = "GRAM_Data"
const GRAM_PLANET = "earth"
const GRAM_START_TIME = SimulatorModel.DateTime(2012, 8, 6, 5, 10, 46.0)
const NOMINAL_ATMOSPHERE_CACHE = Ref{Any}(nothing)
const MONTE_CARLO_ATMOSPHERE_CACHE = Ref{Any}(nothing)

function _reset_gram_atmosphere!(
    atmosphere_model::SimulatorModel.GramAtmosphere;
    monte_carlo::Bool,
)
    atmosphere = atmosphere_model.gram_atmosphere
    gram = atmosphere_model.gram
    atmosphere.setPerturbationScales(1.5)
    atmosphere.setMinRelativeStepSize(0.5)
    atmosphere.setSeed(monte_carlo ? rand(1:10_000) : 1001)

    ttime = gram.GramTime()
    ttime.setStartTime(
        GRAM_START_TIME.year,
        GRAM_START_TIME.month,
        GRAM_START_TIME.day,
        GRAM_START_TIME.hours,
        GRAM_START_TIME.minutes,
        GRAM_START_TIME.secs,
        gram.UTC,
        gram.PET,
    )
    atmosphere.setStartTime(ttime)
    return atmosphere_model
end

function _cached_gram_atmosphere(; monte_carlo::Bool)
    cache = monte_carlo ? MONTE_CARLO_ATMOSPHERE_CACHE : NOMINAL_ATMOSPHERE_CACHE
    if cache[] === nothing
        cache[] = SimulatorModel.GramAtmosphere(
            GRAM_DIRECTORY,
            GRAM_DATA_DIRECTORY,
            monte_carlo,
            GRAM_PLANET,
            GRAM_START_TIME,
        )
    end
    return _reset_gram_atmosphere!(cache[]; monte_carlo = monte_carlo)
end

function _collect_runtime_garbage!()
    GC.gc(false)
    try
        PythonCall.GC.gc()
    catch
        nothing
    end
    return nothing
end

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
    final_cartesian_position_error_norms::Vector{Float64}
    final_cartesian_velocity_error_norms::Vector{Float64}
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

function local_velocity_to_cartesian_mps(
    longitude::Real,
    latitude::Real,
    speed::Real,
    flight_path_angle::Real,
    azimuth::Real,
)
    lon = Float64(longitude)
    lat = Float64(latitude)
    v = Float64(speed)
    γ = Float64(flight_path_angle)
    ψ = Float64(azimuth)

    up = SVector(cos(lat) * cos(lon), cos(lat) * sin(lon), sin(lat))
    north = SVector(-sin(lat) * cos(lon), -sin(lat) * sin(lon), cos(lat))
    east = SVector(-sin(lon), cos(lon), 0.0)

    horizontal_speed = v * cos(γ)
    v_up = v * sin(γ)
    v_north = horizontal_speed * cos(ψ)
    v_east = horizontal_speed * sin(ψ)

    return v_up .* up .+ v_north .* north .+ v_east .* east
end

optimal_ref_x_km, optimal_ref_y_km, optimal_ref_z_km = cartesian_histories_km(
    optimal_control.Altitude_100km .* 1e5,
    deg2rad.(optimal_control.Longitude_deg),
    deg2rad.(optimal_control.Latitude_deg),
    R,
)
const REFERENCE_FINAL_CARTESIAN_POSITION_KM = SVector(
    optimal_ref_x_km[end],
    optimal_ref_y_km[end],
    optimal_ref_z_km[end],
)
const REFERENCE_FINAL_CARTESIAN_VELOCITY_MPS = local_velocity_to_cartesian_mps(
    deg2rad(Float64(optimal_control[end, :Longitude_deg])),
    deg2rad(Float64(optimal_control[end, :Latitude_deg])),
    Float64(optimal_control[end, :Velocity_1000mps]) * velocity_scale,
    deg2rad(Float64(optimal_control[end, :FlightPath_deg])),
    deg2rad(Float64(optimal_control[end, :Azimuth_deg])),
)

target_altitude = Float64(optimal_control[end, :Altitude_100km]) * 1e5
target_velocity = Float64(optimal_control[end, :Velocity_1000mps]) * velocity_scale
target_γ = deg2rad(Float64(optimal_control[end, :FlightPath_deg]))
target_longitude = deg2rad(Float64(optimal_control[end, :Longitude_deg]))
target_latitude = deg2rad(Float64(optimal_control[end, :Latitude_deg]))
target_states = SimulatorModel.TargetStates(
    altitude=target_altitude,
    longitude=target_longitude,
    latitude=target_latitude,
    velocity=target_velocity,
    flight_path_angle=target_γ,
)

function build_edl_params(control_function::Function; disturbance::Bool, monte_carlo::Bool)
    atmosphere = _cached_gram_atmosphere(; monte_carlo = monte_carlo)
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

function _nominal_summary_row(case::ControllerCase, nominal_sol)
    if nominal_sol === nothing
        return DataFrame()
    end

    final_altitude_m = getindex(nominal_sol.u[end], 1)
    final_longitude_rad = getindex(nominal_sol.u[end], 2)
    final_latitude_rad = getindex(nominal_sol.u[end], 3)
    final_velocity_mps = getindex(nominal_sol.u[end], 4)
    final_fpa_rad = getindex(nominal_sol.u[end], 5)
    final_azimuth_rad = getindex(nominal_sol.u[end], 6)
    final_heat_load = getindex(nominal_sol.u[end], 7)
    nominal_x_km, nominal_y_km, nominal_z_km = latlonalt_to_cartesian_km(
        final_latitude_rad,
        final_longitude_rad,
        final_altitude_m,
        R,
    )
    final_velocity_cartesian_mps = local_velocity_to_cartesian_mps(
        final_longitude_rad,
        final_latitude_rad,
        final_velocity_mps,
        final_fpa_rad,
        final_azimuth_rad,
    )

    return DataFrame(
        Controller=[case.name],
        ControllerSlug=[case.slug],
        RunType=["Nominal"],
        Simulation=[0],
        Completed=[true],
        FinalLongitude_deg=[rad2deg(final_longitude_rad)],
        FinalLatitude_deg=[rad2deg(final_latitude_rad)],
        FinalAltitude_km=[final_altitude_m / 1e3],
        FinalVelocity_kms=[final_velocity_mps / 1e3],
        FinalHeatLoad_Jm2=[final_heat_load],
        FinalPositionX_km=[nominal_x_km],
        FinalPositionY_km=[nominal_y_km],
        FinalPositionZ_km=[nominal_z_km],
        FinalCartesianPositionErrorNorm_km=[
            norm(SVector(nominal_x_km, nominal_y_km, nominal_z_km) .- REFERENCE_FINAL_CARTESIAN_POSITION_KM)
        ],
        FinalCartesianVelocityErrorNorm_kms=[
            norm(final_velocity_cartesian_mps .- REFERENCE_FINAL_CARTESIAN_VELOCITY_MPS) / 1e3
        ],
    )
end

function build_case_summary(result::MonteCarloResults, nominal_sol)
    monte_carlo_df = DataFrame(
        Controller=fill(result.case.name, NUM_SIMULATIONS),
        ControllerSlug=fill(result.case.slug, NUM_SIMULATIONS),
        RunType=fill("MonteCarlo", NUM_SIMULATIONS),
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
        FinalCartesianPositionErrorNorm_km=result.final_cartesian_position_error_norms,
        FinalCartesianVelocityErrorNorm_kms=result.final_cartesian_velocity_error_norms,
    )
    nominal_df = _nominal_summary_row(result.case, nominal_sol)
    return isempty(nominal_df) ? monte_carlo_df : vcat(monte_carlo_df, nominal_df)
end

function store_case_output!(result::MonteCarloResults, sim::Int, sol_mc)
    altitudes_m = getindex.(sol_mc.u, 1)
    longitudes_rad = getindex.(sol_mc.u, 2)
    latitudes_rad = getindex.(sol_mc.u, 3)
    x_km, y_km, z_km = cartesian_histories_km(altitudes_m, longitudes_rad, latitudes_rad, R)

    result.final_latitudes[sim] = rad2deg(getindex(sol_mc.u[end], 3))
    result.final_longitudes[sim] = rad2deg(getindex(sol_mc.u[end], 2))
    result.final_altitudes[sim] = getindex(sol_mc.u[end], 1) / 1e3
    result.final_velocities[sim] = getindex(sol_mc.u[end], 4) / 1e3
    result.final_heat_loads[sim] = getindex(sol_mc.u[end], 7)
    result.final_x_positions[sim] = x_km[end]
    result.final_y_positions[sim] = y_km[end]
    result.final_z_positions[sim] = z_km[end]
    result.final_cartesian_position_error_norms[sim] = norm(
        SVector(x_km[end], y_km[end], z_km[end]) .- REFERENCE_FINAL_CARTESIAN_POSITION_KM
    )
    final_velocity_cartesian_mps = local_velocity_to_cartesian_mps(
        longitudes_rad[end],
        latitudes_rad[end],
        getindex(sol_mc.u[end], 4),
        getindex(sol_mc.u[end], 5),
        getindex(sol_mc.u[end], 6),
    )
    result.final_cartesian_velocity_error_norms[sim] =
        norm(final_velocity_cartesian_mps .- REFERENCE_FINAL_CARTESIAN_VELOCITY_MPS) / 1e3
end

function run_monte_carlo_case(case::ControllerCase)
    result = MonteCarloResults(case, NUM_SIMULATIONS)
    nominal_sol = nothing
    if RUN_NOMINAL_OVERLAY
        nominal_sol, _ = run_case(case.control_function; disturbance=false, monte_carlo=false, dt=0.1)
        _collect_runtime_garbage!()
    end

    println("Running $(case.name) Monte Carlo cases")
    @showprogress for sim in 1:NUM_SIMULATIONS
        try
            sol_mc, local_saved_values = run_case(case.control_function; disturbance=true, monte_carlo=true, dt=0.5)
            store_case_output!(result, sim, sol_mc)
            sol_mc = nothing
            local_saved_values = nothing
            _collect_runtime_garbage!()
        catch e
            @warn "Monte Carlo case failed" controller=case.name sim exception=(e, catch_backtrace())
        end
    end

    _collect_runtime_garbage!()
    return build_case_summary(result, nominal_sol)
end

function _summary_case_color(controller_slug::AbstractString)
    for case in CONTROLLER_CASES
        if case.slug == controller_slug
            return case.color
        end
    end
    return :black
end

function _summary_case_name(controller_slug::AbstractString)
    for case in CONTROLLER_CASES
        if case.slug == controller_slug
            return case.name
        end
    end
    return String(controller_slug)
end

function _filter_summary_rows(summary_df::DataFrame; include_open_loop::Bool=true, run_type::Union{Nothing, String}=nothing)
    df = summary_df
    if !include_open_loop
        df = filter(row -> row.ControllerSlug != "open_loop", df)
    end
    if run_type !== nothing
        df = filter(row -> row.RunType == run_type, df)
    end
    return df
end

function load_summary_table()
    return CSV.read(joinpath(DATA_DIR, "summary.csv"), DataFrame)
end

function plot_landing_locations(
    summary_df::DataFrame;
    include_open_loop::Bool=true,
    filename::String=include_open_loop ?
        "monte_carlo_final_landing_locations.pdf" :
        "monte_carlo_final_landing_locations_without_open_loop.pdf",
    title::String=include_open_loop ?
        "Monte Carlo Simulations: Final Landing Locations" :
        "Monte Carlo Simulations: Final Landing Locations (Without Open Loop)",
)
    plt = plot(
        xlabel="Longitude (deg)",
        ylabel="Latitude (deg)",
        title=title,
        legend=true,
    )
    monte_carlo_df = _filter_summary_rows(summary_df; include_open_loop = include_open_loop, run_type = "MonteCarlo")
    nominal_df = _filter_summary_rows(summary_df; include_open_loop = include_open_loop, run_type = "Nominal")
    for controller_slug in unique(String.(monte_carlo_df.ControllerSlug))
        case_df = filter(row -> row.ControllerSlug == controller_slug, monte_carlo_df)
        color = _summary_case_color(controller_slug)
        label = _summary_case_name(controller_slug)
        plot!(
            plt,
            Float64.(case_df.FinalLongitude_deg),
            Float64.(case_df.FinalLatitude_deg),
            seriestype=:scatter,
            color=color,
            label="$(label) landings",
            alpha=0.65,
        )
        case_nominal_df = filter(row -> row.ControllerSlug == controller_slug, nominal_df)
        if !isempty(case_nominal_df)
            plot!(
                plt,
                [Float64(case_nominal_df[1, :FinalLongitude_deg])],
                [Float64(case_nominal_df[1, :FinalLatitude_deg])],
                seriestype=:scatter,
                color=color,
                markershape=:hexagon,
                markersize=8,
                label="$(label) nominal",
            )
        end
    end
    plot!(plt, [rad2deg(target_states.longitude)], [rad2deg(target_states.latitude)], seriestype=:scatter, color=:black, markershape=:diamond, markersize=8, label="Target")
    if DISPLAY_PLOTS
        display(plt)
    end
    savefig(plt, joinpath(PLOTS_DIR, filename))
end

function plot_final_cartesian_locations(
    summary_df::DataFrame;
    include_open_loop::Bool=true,
    include_open_loop_nominal::Bool=true,
    filename::String=include_open_loop_nominal ?
        "monte_carlo_final_cartesian_locations.pdf" :
        "monte_carlo_final_cartesian_locations_without_open_loop_nominal.pdf",
    title::String=include_open_loop_nominal ?
        "Monte Carlo Simulations: Final Cartesian Locations" :
        "Monte Carlo Simulations: Final Cartesian Locations (Without Open Loop Nominal)",
)
    plt = plot(
        xlabel="X (km)",
        ylabel="Y (km)",
        title=title,
        legend=true,
    )
    monte_carlo_df = _filter_summary_rows(summary_df; include_open_loop = include_open_loop, run_type = "MonteCarlo")
    nominal_df = _filter_summary_rows(summary_df; include_open_loop = include_open_loop, run_type = "Nominal")
    for controller_slug in unique(String.(monte_carlo_df.ControllerSlug))
        case_df = filter(row -> row.ControllerSlug == controller_slug, monte_carlo_df)
        color = _summary_case_color(controller_slug)
        label = _summary_case_name(controller_slug)
        plot!(
            plt,
            Float64.(case_df.FinalPositionX_km),
            Float64.(case_df.FinalPositionY_km),
            seriestype=:scatter,
            color=color,
            label="$(label) final positions",
            alpha=0.65,
        )
        case_nominal_df = filter(row -> row.ControllerSlug == controller_slug, nominal_df)
        if !isempty(case_nominal_df) && (include_open_loop_nominal || controller_slug != "open_loop")
            plot!(
                plt,
                [Float64(case_nominal_df[1, :FinalPositionX_km])],
                [Float64(case_nominal_df[1, :FinalPositionY_km])],
                seriestype=:scatter,
                color=color,
                markershape=:hexagon,
                markersize=8,
                label="$(label) nominal",
            )
        end
    end
    target_x, target_y, _ = latlonalt_to_cartesian_km(target_states.latitude, target_states.longitude, target_states.altitude, R)
    plot!(plt, [target_x], [target_y], seriestype=:scatter, color=:black, markershape=:diamond, markersize=8, label="Target")
    if DISPLAY_PLOTS
        display(plt)
    end
    savefig(plt, joinpath(PLOTS_DIR, filename))
end

function cleanup_data_dir!()
    mkpath(DATA_DIR)
    for name in readdir(DATA_DIR)
        if name != "summary.csv"
            rm(joinpath(DATA_DIR, name); recursive=true, force=true)
        end
    end
    return nothing
end

function plot_final_cartesian_velocity_error_norms(
    summary_df::DataFrame;
    include_open_loop::Bool=true,
    filename::String=include_open_loop ?
        "monte_carlo_final_cartesian_velocity_error_norm_with_open_loop.pdf" :
        "monte_carlo_final_cartesian_velocity_error_norm_without_open_loop.pdf",
    title::String=include_open_loop ?
        "Final Cartesian Velocity Error Norm" :
        "Final Cartesian Velocity Error Norm (No Open Loop)",
)
    filtered_df = _filter_summary_rows(summary_df; include_open_loop = include_open_loop, run_type = "MonteCarlo")

    plt = plot(
        xlabel="Simulation Index",
        ylabel="Final Cartesian Velocity Error Norm (km/s)",
        title=title,
        legend=true,
    )
    for controller_slug in unique(String.(filtered_df.ControllerSlug))
        case_df = filter(row -> row.ControllerSlug == controller_slug, filtered_df)
        errs = Float64.(case_df.FinalCartesianVelocityErrorNorm_kms)
        if isempty(errs)
            continue
        end
        sims = Int.(case_df.Simulation)
        color = _summary_case_color(controller_slug)
        label = _summary_case_name(controller_slug)
        plot!(
            plt,
            sims,
            errs,
            seriestype=:scatter,
            color=color,
            alpha=0.75,
            label=label,
        )
        mean_err = mean(errs)
        plot!(
            plt,
            [minimum(sims), maximum(sims)],
            [mean_err, mean_err],
            color=color,
            linestyle=:dash,
            linewidth=2,
            label="$(label) mean",
        )
    end
    if DISPLAY_PLOTS
        display(plt)
    end
    savefig(plt, joinpath(PLOTS_DIR, filename))
end

function plot_final_cartesian_position_error_norms(
    summary_df::DataFrame;
    include_open_loop::Bool=false,
    filename::String=include_open_loop ?
        "monte_carlo_final_cartesian_position_error_norm_with_open_loop.pdf" :
        "monte_carlo_final_cartesian_position_error_norm_without_open_loop.pdf",
    title::String=include_open_loop ?
        "Final Cartesian Position Error Norm" :
        "Final Cartesian Position Error Norm (No Open Loop)",
)
    filtered_df = _filter_summary_rows(summary_df; include_open_loop = include_open_loop, run_type = "MonteCarlo")

    plt = plot(
        xlabel="Simulation Index",
        ylabel="Final Cartesian Position Error Norm (km)",
        title=title,
        legend=true,
    )
    for controller_slug in unique(String.(filtered_df.ControllerSlug))
        case_df = filter(row -> row.ControllerSlug == controller_slug, filtered_df)
        errs = Float64.(case_df.FinalCartesianPositionErrorNorm_km)
        if isempty(errs)
            continue
        end
        sims = Int.(case_df.Simulation)
        color = _summary_case_color(controller_slug)
        label = _summary_case_name(controller_slug)
        plot!(
            plt,
            sims,
            errs,
            seriestype=:scatter,
            color=color,
            alpha=0.75,
            label=label,
        )
        mean_err = mean(errs)
        plot!(
            plt,
            [minimum(sims), maximum(sims)],
            [mean_err, mean_err],
            color=color,
            linestyle=:dash,
            linewidth=2,
            label="$(label) mean",
        )
    end
    if DISPLAY_PLOTS
        display(plt)
    end
    savefig(plt, joinpath(PLOTS_DIR, filename))
end

mkpath(PLOTS_DIR)
cleanup_data_dir!()
summary_dfs = [run_monte_carlo_case(case) for case in CONTROLLER_CASES]
CSV.write(joinpath(DATA_DIR, "summary.csv"), vcat(summary_dfs...))
cleanup_data_dir!()

summary_df = load_summary_table()
plot_landing_locations(summary_df; include_open_loop=true)
plot_landing_locations(summary_df; include_open_loop=false)
plot_final_cartesian_locations(summary_df; include_open_loop=true, include_open_loop_nominal=true)
plot_final_cartesian_locations(
    summary_df;
    include_open_loop=false,
    include_open_loop_nominal=false,
    filename="monte_carlo_final_cartesian_locations_without_open_loop_nominal.pdf",
    title="Monte Carlo Simulations: Final Cartesian Locations (Without Open Loop)",
)
plot_final_cartesian_position_error_norms(summary_df; include_open_loop=false)
plot_final_cartesian_velocity_error_norms(summary_df; include_open_loop=true)
plot_final_cartesian_velocity_error_norms(summary_df; include_open_loop=false)
