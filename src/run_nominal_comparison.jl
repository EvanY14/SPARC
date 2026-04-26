include("model/SimulatorModel.jl")

using .SimulatorModel
using StaticArrays
using PythonCall
using LinearAlgebra
using Statistics
ENV["GKSwstype"] = get(ENV, "GKSwstype", "100")
using Plots
using DifferentialEquations
using CSV
using DataFrames
using Interpolations
include("reference/trajectory_initialization.jl")

gr()

const OUTPUT_DIR = get(ENV, "SPARC_NOMINAL_OUTPUT_DIR", "nominal_comparison_output")
const DISPLAY_PLOTS = parse(Bool, get(ENV, "SPARC_NOMINAL_DISPLAY", "false"))
const PLOTS_DIR = joinpath(OUTPUT_DIR, "plots")
const MONTE_CARLO_LEGEND_POSITION = :outertopright
const MONTE_CARLO_PLOT_MARGIN = 12Plots.mm
const MONTE_CARLO_MAX_STATE_PROFILE_POINTS = parse(Int, get(ENV, "SPARC_NOMINAL_MAX_STATE_POINTS", "600"))
const MONTE_CARLO_MAX_CONTROL_PROFILE_POINTS = parse(Int, get(ENV, "SPARC_NOMINAL_MAX_CONTROL_POINTS", "600"))
const MONTE_CARLO_TITLE_FONTSIZE = 10
const MONTE_CARLO_GUIDE_FONTSIZE = 9
const MONTE_CARLO_TICK_FONTSIZE = 8
const MONTE_CARLO_LEGEND_FONTSIZE = 8
const IEEE_SINGLE_COLUMN_STATE_SIZE = (520, 760)

struct ControllerCase
    name::String
    slug::String
    control_function::Function
    color::Symbol
end

const CONTROLLER_CASES = (
    ControllerCase("MPG", "mpg", SimulatorModel.mpg, :green),
    ControllerCase("MPG-IA", "mpg_integral", SimulatorModel.mpg_integral, :orange),
    ControllerCase("SM-MPG", "sm_mpg", SimulatorModel.sm_mpg, :purple),
)

const GRAM_DIRECTORY = "GRAMpy/"
const GRAM_DATA_DIRECTORY = "GRAM_Data"
const GRAM_PLANET = "earth"
const GRAM_START_TIME = SimulatorModel.DateTime(2012, 8, 6, 5, 10, 46.0)
const NOMINAL_ATMOSPHERE_CACHE = Ref{Any}(nothing)

function _collect_runtime_garbage!()
    GC.gc(false)
    try
        PythonCall.GC.gc()
    catch
        nothing
    end
    return nothing
end

function _save_plot_and_cleanup!(plt, filename::String)
    if DISPLAY_PLOTS
        display(plt)
    end
    savefig(plt, joinpath(PLOTS_DIR, filename))
    closeall()
    _collect_runtime_garbage!()
    return nothing
end

function _nominal_plot_kwargs(; is_3d::Bool=false)
    return (
        legend = MONTE_CARLO_LEGEND_POSITION,
        left_margin = MONTE_CARLO_PLOT_MARGIN,
        right_margin = MONTE_CARLO_PLOT_MARGIN,
        bottom_margin = MONTE_CARLO_PLOT_MARGIN,
        titlefont = font(MONTE_CARLO_TITLE_FONTSIZE),
        guidefont = font(MONTE_CARLO_GUIDE_FONTSIZE),
        tickfont = font(MONTE_CARLO_TICK_FONTSIZE),
        legendfont = font(MONTE_CARLO_LEGEND_FONTSIZE),
        size = is_3d ? (1050, 760) : (980, 680),
    )
end

function _nominal_subplot_kwargs(; legend = MONTE_CARLO_LEGEND_POSITION)
    return (
        legend = legend,
        left_margin = 8Plots.mm,
        right_margin = 6Plots.mm,
        bottom_margin = 7Plots.mm,
        titlefont = font(MONTE_CARLO_TITLE_FONTSIZE),
        guidefont = font(MONTE_CARLO_GUIDE_FONTSIZE),
        tickfont = font(MONTE_CARLO_TICK_FONTSIZE),
        legendfont = font(MONTE_CARLO_LEGEND_FONTSIZE),
    )
end

function _reset_gram_atmosphere!(atmosphere_model::SimulatorModel.GramAtmosphere)
    atmosphere = atmosphere_model.gram_atmosphere
    gram = atmosphere_model.gram
    atmosphere.setPerturbationScales(1.5)
    atmosphere.setMinRelativeStepSize(0.5)
    atmosphere.setSeed(1001)

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

function _cached_nominal_atmosphere()
    if NOMINAL_ATMOSPHERE_CACHE[] === nothing
        NOMINAL_ATMOSPHERE_CACHE[] = SimulatorModel.GramAtmosphere(
            GRAM_DIRECTORY,
            GRAM_DATA_DIRECTORY,
            false,
            GRAM_PLANET,
            GRAM_START_TIME,
        )
    end
    return _reset_gram_atmosphere!(NOMINAL_ATMOSPHERE_CACHE[])
end

function _downsample_indices(n::Int, max_points::Int)
    if n <= max_points || max_points <= 1
        return collect(1:n)
    end
    raw = round.(Int, range(1, n; length = max_points))
    return unique(clamp.(raw, 1, n))
end

function saved_value_vectors(saved_values)
    n = length(saved_values.t)
    densities = Vector{Float64}(undef, n)
    betas = Vector{Float64}(undef, n)
    alphas = Vector{Float64}(undef, n)
    heat_rates = Vector{Float64}(undef, n)
    for i in 1:n
        densities[i] = Float64(saved_values.saveval[i][1])
        betas[i] = Float64(saved_values.saveval[i][2])
        alphas[i] = Float64(saved_values.saveval[i][3])
        heat_rates[i] = Float64(saved_values.saveval[i][4])
    end
    return densities, betas, alphas, heat_rates
end

function nominal_density_effect!(integrator)
    h = integrator.u[1]
    ϕ = integrator.u[2]
    θ = integrator.u[3]
    if !all(isfinite, integrator.u) || h <= integrator.p.target_states.altitude || integrator.p.R + h <= 0.0 || abs(θ) >= deg2rad(89.9) || !isfinite(ϕ)
        terminate!(integrator)
        return
    end

    density_function = integrator.p.atmospheric_density_function
    lat_lon_alt = (θ, ϕ, h)
    integrator.p.atmospheric_density, integrator.p.wind = density_function(lat_lon_alt, integrator.t)
end

nominal_density_callback = DiscreteCallback((u, t, integrator) -> true, nominal_density_effect!)

function latlonalt_to_cartesian_km(latitude, longitude, altitude, planet_radius)
    radius_km = (planet_radius + altitude) / 1e3
    x = radius_km * cos(latitude) * cos(longitude)
    y = radius_km * cos(latitude) * sin(longitude)
    z = radius_km * sin(latitude)
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

function build_context()
    optimal_control = CSV.read("optimal_trajectory.csv", DataFrame)
    init = reference_initial_conditions(optimal_control)
    tspan = init.tspan

    u0 = MVector{7, Float64}(init.h0, init.ϕ0, init.θ0, init.v0, init.γ0, init.ψ0, 0.0)
    mass = SimulatorModel.VEHICLE.mass
    area = SimulatorModel.VEHICLE.reference_area
    μ = 3.986004418e14
    R = 6378137.0

    interp_bank = linear_interpolation(optimal_control.Time_s, optimal_control.BankAngle_deg, extrapolation_bc = Line())
    interp_alpha = linear_interpolation(optimal_control.Time_s, optimal_control.AngleOfAttack_deg, extrapolation_bc = Line())
    β0_ref = deg2rad(interp_bank(tspan[1]))
    α0_ref = deg2rad(interp_alpha(tspan[1]))

    velocity_scale = abs(optimal_control.Velocity_1000mps[1] * 1e3 - init.v0) <= abs(optimal_control.Velocity_1000mps[1] * 1e4 - init.v0) ? 1e3 : 1e4
    interp_altitude = linear_interpolation(optimal_control.Time_s, optimal_control.Altitude_100km .* 1e5, extrapolation_bc = Line())
    interp_velocity = linear_interpolation(optimal_control.Time_s, optimal_control.Velocity_1000mps .* velocity_scale, extrapolation_bc = Line())
    interp_longitude = linear_interpolation(optimal_control.Time_s, optimal_control.Longitude_deg .* (π / 180), extrapolation_bc = Line())
    interp_latitude = linear_interpolation(optimal_control.Time_s, optimal_control.Latitude_deg .* (π / 180), extrapolation_bc = Line())
    interp_flight_path = linear_interpolation(optimal_control.Time_s, optimal_control.FlightPath_deg .* (π / 180), extrapolation_bc = Line())
    interp_azimuth = linear_interpolation(optimal_control.Time_s, optimal_control.Azimuth_deg .* (π / 180), extrapolation_bc = Line())
    nominal_trajectory = SVector{6, AbstractInterpolation}(interp_altitude, interp_longitude, interp_latitude, interp_velocity, interp_flight_path, interp_azimuth)

    target_states = SimulatorModel.TargetStates(
        altitude = Float64(optimal_control[end, :Altitude_100km]) * 1e5,
        longitude = deg2rad(Float64(optimal_control[end, :Longitude_deg])),
        latitude = deg2rad(Float64(optimal_control[end, :Latitude_deg])),
        velocity = Float64(optimal_control[end, :Velocity_1000mps]) * velocity_scale,
        flight_path_angle = deg2rad(Float64(optimal_control[end, :FlightPath_deg])),
    )

    return (;
        optimal_control = optimal_control,
        tspan = tspan,
        u0 = u0,
        mass = mass,
        area = area,
        μ = μ,
        R = R,
        β0_ref = β0_ref,
        α0_ref = α0_ref,
        nominal_trajectory = nominal_trajectory,
        target_states = target_states,
        velocity_scale = velocity_scale,
    )
end

const CONTEXT = build_context()

function build_edl_params(control_function::Function)
    atmosphere = _cached_nominal_atmosphere()
    mpc_params = SimulatorModel.MPCParams{100, 7, 8, 0.1}(
        n_horizon = SimulatorModel.mpg_default_horizon(),
        time_step = SimulatorModel.mpg_default_time_step(),
        H_SCALE = 1.0e5,
        V_SCALE = 1.0e4,
        T_SCALE = 1.0,
        n_exp = 4.512,
        m_exp = 0.82958,
        learning_rate = 0.1,
    )
    return SimulatorModel.EDLParams(
        mass = CONTEXT.mass,
        area = CONTEXT.area,
        μ = CONTEXT.μ,
        R = CONTEXT.R,
        control_function = control_function,
        β = CONTEXT.β0_ref,
        α = CONTEXT.α0_ref,
        atmospheric_density_function = (LatLonAlt, t) -> SimulatorModel.atmospheric_density(LatLonAlt, t, atmosphere, false),
        atmospheric_density = 0.0,
        wind = SVector{3, Float64}(zeros(3)),
        target_states = CONTEXT.target_states,
        optimization_states = SimulatorModel.OptimizationStates(),
        nominal_trajectory = CONTEXT.nominal_trajectory,
        cache = SimulatorModel.EDLCache(),
        mpc_params = mpc_params,
    )
end

function run_case(control_function::Function; dt::Float64=0.1)
    local_saved_values = SavedValues(Float64, Tuple{Float64, Float64, Float64, Float64})
    local_saving_callback = SavingCallback(
        (u, t, integrator) -> (
            integrator.p.atmospheric_density,
            integrator.p.β,
            integrator.p.α,
            integrator.p.cache.q_dot,
        ),
        local_saved_values,
        saveat = 0.1,
    )
    callbacks = CallbackSet(
        SimulatorModel.altitude_termination_condition,
        nominal_density_callback,
        local_saving_callback,
        SimulatorModel.control_callback,
    )
    params = build_edl_params(control_function)
    prob = ODEProblem(SimulatorModel.edl_dynamics, copy(CONTEXT.u0), CONTEXT.tspan, params, callback = callbacks)
    sol = solve(prob, Tsit5(), dt = dt, adaptive = false)
    return sol, local_saved_values
end

mutable struct NominalRunResult
    case::ControllerCase
    times::Vector{Float64}
    altitude_profiles::Vector{Float64}
    longitude_profiles::Vector{Float64}
    latitude_profiles::Vector{Float64}
    velocity_profiles::Vector{Float64}
    flight_path_profiles::Vector{Float64}
    azimuth_profiles::Vector{Float64}
    control_times::Vector{Float64}
    alpha_profiles::Vector{Float64}
    beta_profiles::Vector{Float64}
end

function build_nominal_run(case::ControllerCase)
    sol, local_saved_values = run_case(case.control_function; dt = 0.1)
    altitudes_m = getindex.(sol.u, 1)
    longitudes_rad = getindex.(sol.u, 2)
    latitudes_rad = getindex.(sol.u, 3)
    state_idx = _downsample_indices(length(sol.t), MONTE_CARLO_MAX_STATE_PROFILE_POINTS)

    _, betas, alphas, _ = saved_value_vectors(local_saved_values)
    control_idx = _downsample_indices(length(local_saved_values.t), MONTE_CARLO_MAX_CONTROL_PROFILE_POINTS)

    return NominalRunResult(
        case,
        Float64.(sol.t[state_idx]),
        Float64.(altitudes_m[state_idx] ./ 1e3),
        Float64.(rad2deg.(longitudes_rad[state_idx])),
        Float64.(rad2deg.(latitudes_rad[state_idx])),
        Float64.(getindex.(sol.u[state_idx], 4) ./ 1e3),
        Float64.(rad2deg.(getindex.(sol.u[state_idx], 5))),
        Float64.(rad2deg.(getindex.(sol.u[state_idx], 6))),
        Float64.(local_saved_values.t[control_idx]),
        Float64.(rad2deg.(alphas[control_idx])),
        Float64.(rad2deg.(betas[control_idx])),
    )
end

function _plot_state_panel(results::Vector{NominalRunResult}, profile_field::Symbol, reference_times, reference_values; ylabel::String)
    plt = plot(
        xlabel = "Time (s)",
        ylabel = ylabel,
        _nominal_subplot_kwargs()...,
    )
    plot!(plt, reference_times, reference_values, color = :black, linewidth = 2, label = "Reference")
    for result in results
        plot!(
            plt,
            getfield(result, :times),
            getfield(result, profile_field),
            color = result.case.color,
            alpha = 0.35,
            linewidth = 1.25,
            label = result.case.name,
        )
    end
    return plt
end

function _plot_control_panel(results::Vector{NominalRunResult}, profile_field::Symbol, reference_times, reference_values; ylabel::String)
    plt = plot(
        xlabel = "Time (s)",
        ylabel = ylabel,
        _nominal_subplot_kwargs(legend = MONTE_CARLO_LEGEND_POSITION)...,
    )
    plot!(plt, reference_times, reference_values, seriestype = :steppost, color = :black, linewidth = 2, label = "Reference")
    for result in results
        plot!(
            plt,
            result.control_times,
            getfield(result, profile_field),
            seriestype = :steppost,
            color = result.case.color,
            alpha = 0.35,
            linewidth = 1.25,
            label = result.case.name,
        )
    end
    return plt
end

function plot_nominal_state_profiles(results::Vector{NominalRunResult})
    state_plots = Any[
        _plot_state_panel(results, :altitude_profiles, CONTEXT.optimal_control.Time_s, CONTEXT.optimal_control.Altitude_100km .* 1e5 ./ 1e3; ylabel = "Altitude (km)"),
        _plot_state_panel(results, :longitude_profiles, CONTEXT.optimal_control.Time_s, CONTEXT.optimal_control.Longitude_deg; ylabel = "Longitude (deg)"),
        _plot_state_panel(results, :latitude_profiles, CONTEXT.optimal_control.Time_s, CONTEXT.optimal_control.Latitude_deg; ylabel = "Latitude (deg)"),
        _plot_state_panel(results, :velocity_profiles, CONTEXT.optimal_control.Time_s, CONTEXT.optimal_control.Velocity_1000mps .* CONTEXT.velocity_scale ./ 1e3; ylabel = "Velocity (km/s)"),
        _plot_state_panel(results, :flight_path_profiles, CONTEXT.optimal_control.Time_s, CONTEXT.optimal_control.FlightPath_deg; ylabel = "Flight Path Angle (deg)"),
        _plot_state_panel(results, :azimuth_profiles, CONTEXT.optimal_control.Time_s, CONTEXT.optimal_control.Azimuth_deg; ylabel = "Azimuth (deg)"),
    ]

    fig_part1 = plot(
        state_plots[1],
        state_plots[2],
        state_plots[3];
        layout = (3, 1),
        size = IEEE_SINGLE_COLUMN_STATE_SIZE,
        margin = 12Plots.mm,
        left_margin = 14Plots.mm,
        right_margin = 10Plots.mm,
        bottom_margin = 12Plots.mm,
        top_margin = 10Plots.mm,
        tickfont = font(MONTE_CARLO_TICK_FONTSIZE),
        guidefont = font(MONTE_CARLO_GUIDE_FONTSIZE),
        titlefont = font(MONTE_CARLO_TITLE_FONTSIZE),
    )
    _save_plot_and_cleanup!(fig_part1, "nominal_state_profiles_part1.pdf")

    fig_part2 = plot(
        state_plots[4],
        state_plots[5],
        state_plots[6];
        layout = (3, 1),
        size = IEEE_SINGLE_COLUMN_STATE_SIZE,
        margin = 12Plots.mm,
        left_margin = 14Plots.mm,
        right_margin = 10Plots.mm,
        bottom_margin = 12Plots.mm,
        top_margin = 10Plots.mm,
        tickfont = font(MONTE_CARLO_TICK_FONTSIZE),
        guidefont = font(MONTE_CARLO_GUIDE_FONTSIZE),
        titlefont = font(MONTE_CARLO_TITLE_FONTSIZE),
    )
    _save_plot_and_cleanup!(fig_part2, "nominal_state_profiles_part2.pdf")
end

function plot_nominal_control_profiles(results::Vector{NominalRunResult})
    control_plots = Any[
        _plot_control_panel(results, :alpha_profiles, CONTEXT.optimal_control.Time_s, CONTEXT.optimal_control.AngleOfAttack_deg; ylabel = "Angle of Attack (deg)"),
        _plot_control_panel(results, :beta_profiles, CONTEXT.optimal_control.Time_s, CONTEXT.optimal_control.BankAngle_deg; ylabel = "Bank Angle (deg)"),
    ]

    fig = plot(
        control_plots...;
        layout = (2, 1),
        size = (1300, 900),
        margin = 12Plots.mm,
        left_margin = 14Plots.mm,
        right_margin = 10Plots.mm,
        bottom_margin = 12Plots.mm,
        top_margin = 10Plots.mm,
        tickfont = font(MONTE_CARLO_TICK_FONTSIZE),
        guidefont = font(MONTE_CARLO_GUIDE_FONTSIZE),
        titlefont = font(MONTE_CARLO_TITLE_FONTSIZE),
    )
    _save_plot_and_cleanup!(fig, "nominal_control_profiles.pdf")
end

function cleanup_plots_dir!()
    mkpath(PLOTS_DIR)
    for name in readdir(PLOTS_DIR)
        rm(joinpath(PLOTS_DIR, name); recursive = true, force = true)
    end
    return nothing
end

mkpath(PLOTS_DIR)
cleanup_plots_dir!()

println("Running nominal MPG controller comparison")
results = [build_nominal_run(case) for case in CONTROLLER_CASES]
plot_nominal_state_profiles(results)
plot_nominal_control_profiles(results)
