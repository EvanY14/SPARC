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

gr()

const NUM_SIMULATIONS = parse(Int, get(ENV, "SPARC_MC_RUNS", "100"))
const OUTPUT_DIR = get(ENV, "SPARC_MC_OUTPUT_DIR", "monte_carlo_output")
const RUN_NOMINAL_OVERLAY = parse(Bool, get(ENV, "SPARC_MC_NOMINAL_OVERLAY", "true"))
const DISPLAY_PLOTS = parse(Bool, get(ENV, "SPARC_MC_DISPLAY", "false"))
const REPLOT_ONLY = parse(Bool, get(ENV, "SPARC_MC_REPLOT_ONLY", "false"))
const DATA_DIR = joinpath(OUTPUT_DIR, "data")
const PLOTS_DIR = joinpath(OUTPUT_DIR, "plots")
const MONTE_CARLO_LEGEND_POSITION = :outertopright
const MONTE_CARLO_PLOT_MARGIN = 12Plots.mm
const THREE_SIGMA = 3.0
const MONTE_CARLO_MAX_STATE_PROFILE_POINTS = parse(Int, get(ENV, "SPARC_MC_MAX_STATE_POINTS", "600"))
const MONTE_CARLO_MAX_CONTROL_PROFILE_POINTS = parse(Int, get(ENV, "SPARC_MC_MAX_CONTROL_POINTS", "600"))
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

function open_loop_control(integrator)
    t = integrator.t
    β_cmd = deg2rad(interp_optimal_control(t))
    α_cmd = deg2rad(interp_optimal_alpha(t))
    return β_cmd, α_cmd
end

const CONTROLLER_CASES = (
    ControllerCase("MPG", "mpg", SimulatorModel.mpg, :green),
    ControllerCase("MPG-IA", "mpg_integral", SimulatorModel.mpg_integral, :orange),
    ControllerCase("MPG-SM", "sm_mpg", SimulatorModel.sm_mpg, :purple),
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

function _save_plot_and_cleanup!(plt, filename::String)
    if DISPLAY_PLOTS
        display(plt)
    end
    savefig(plt, joinpath(PLOTS_DIR, filename))
    closeall()
    _collect_runtime_garbage!()
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
reference_heat_rate = 8.53e-13 .* SimulatorModel.earth_atmosphere_density.(optimal_control.Altitude_100km .* 1e5).^0.82958 .* (optimal_control.Velocity_1000mps .* velocity_scale).^4.512
interp_heat_rate = linear_interpolation(optimal_control.Time_s, reference_heat_rate, extrapolation_bc=Line())

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

function _downsample_indices(n::Int, max_points::Int)
    if n <= max_points || max_points <= 1
        return collect(1:n)
    end
    raw = round.(Int, range(1, n; length = max_points))
    return unique(clamp.(raw, 1, n))
end

function _downsample_vector(values, max_points::Int)
    idx = _downsample_indices(length(values), max_points)
    return Float64.(values[idx]), idx
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

function valid_profile_indices(result::MonteCarloResults, profile_field::Symbol; time_field::Symbol=:times)
    times = getfield(result, time_field)
    profiles = getfield(result, profile_field)
    return findall(i -> times[i] !== nothing && profiles[i] !== nothing, eachindex(times))
end

function valid_profile_indices(
    result::MonteCarloResults,
    profile_fields::Tuple{Vararg{Symbol}};
    time_field::Symbol=:times,
)
    times = getfield(result, time_field)
    return findall(eachindex(times)) do i
        time_values = times[i]
        time_values !== nothing && all(profile_field -> begin
            profile_values = getfield(result, profile_field)[i]
            profile_values !== nothing && length(profile_values) == length(time_values)
        end, profile_fields)
    end
end

function store_case_output!(result::MonteCarloResults, sim::Int, sol_mc, local_saved_values)
    altitudes_m = getindex.(sol_mc.u, 1)
    longitudes_rad = getindex.(sol_mc.u, 2)
    latitudes_rad = getindex.(sol_mc.u, 3)
    x_km, y_km, z_km = cartesian_histories_km(altitudes_m, longitudes_rad, latitudes_rad, R)
    state_idx = _downsample_indices(length(sol_mc.t), MONTE_CARLO_MAX_STATE_PROFILE_POINTS)

    result.altitude_profiles[sim] = Float64.(altitudes_m[state_idx] ./ 1e3)
    result.longitude_profiles[sim] = Float64.(rad2deg.(longitudes_rad[state_idx]))
    result.latitude_profiles[sim] = Float64.(rad2deg.(latitudes_rad[state_idx]))
    result.velocity_profiles[sim] = Float64.(getindex.(sol_mc.u[state_idx], 4) ./ 1e3)
    result.flight_path_profiles[sim] = Float64.(rad2deg.(getindex.(sol_mc.u[state_idx], 5)))
    result.azimuth_profiles[sim] = Float64.(rad2deg.(getindex.(sol_mc.u[state_idx], 6)))
    result.x_profiles[sim] = Float64.(x_km[state_idx])
    result.y_profiles[sim] = Float64.(y_km[state_idx])
    result.z_profiles[sim] = Float64.(z_km[state_idx])
    result.times[sim] = Float64.(sol_mc.t[state_idx])

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

    _, betas, alphas, heat_rates = saved_value_vectors(local_saved_values)
    control_idx = _downsample_indices(length(local_saved_values.t), MONTE_CARLO_MAX_CONTROL_PROFILE_POINTS)
    result.alpha_profiles[sim] = Float64.(rad2deg.(alphas[control_idx]))
    result.beta_profiles[sim] = Float64.(rad2deg.(betas[control_idx]))
    result.heat_rate_profiles[sim] = Float64.(heat_rates[control_idx])
    result.control_times[sim] = Float64.(local_saved_values.t[control_idx])
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
            store_case_output!(result, sim, sol_mc, local_saved_values)
            sol_mc = nothing
            local_saved_values = nothing
            _collect_runtime_garbage!()
        catch e
            @warn "Monte Carlo case failed" controller=case.name sim exception=(e, catch_backtrace())
        end
    end

    _collect_runtime_garbage!()
    return result, nominal_sol, build_case_summary(result, nominal_sol)
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

function _finite_xy(case_df::DataFrame, x_column::Symbol, y_column::Symbol)
    x = Float64[]
    y = Float64[]
    for row in eachrow(case_df)
        x_val = Float64(row[x_column])
        y_val = Float64(row[y_column])
        if isfinite(x_val) && isfinite(y_val)
            push!(x, x_val)
            push!(y, y_val)
        end
    end
    return x, y
end

function _finite_xyz(case_df::DataFrame, x_column::Symbol, y_column::Symbol, z_column::Symbol)
    x = Float64[]
    y = Float64[]
    z = Float64[]
    for row in eachrow(case_df)
        x_val = Float64(row[x_column])
        y_val = Float64(row[y_column])
        z_val = Float64(row[z_column])
        if isfinite(x_val) && isfinite(y_val) && isfinite(z_val)
            push!(x, x_val)
            push!(y, y_val)
            push!(z, z_val)
        end
    end
    return x, y, z
end

function _mean_xy(xs::AbstractVector{<:Real}, ys::AbstractVector{<:Real})
    isempty(xs) && return nothing
    return mean(Float64.(xs)), mean(Float64.(ys))
end

function _monte_carlo_plot_kwargs(; is_3d::Bool=false)
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

function _monte_carlo_subplot_kwargs(; legend::Bool=false)
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

function _covariance_matrix(points::AbstractMatrix{<:Real})
    n_points = size(points, 1)
    if n_points <= 1
        return zeros(size(points, 2), size(points, 2))
    end
    mean_vector = vec(mean(points; dims = 1))
    centered = points .- reshape(mean_vector, 1, :)
    return (centered' * centered) / (n_points - 1)
end

function _ellipse_outline(points::AbstractMatrix{<:Real}; n_sigma::Real=THREE_SIGMA, n_points::Int=241)
    if size(points, 1) < 2
        return nothing
    end
    mean_vector = vec(mean(points; dims = 1))
    covariance = Symmetric(_covariance_matrix(points))
    decomposition = eigen(covariance)
    scales = sqrt.(clamp.(decomposition.values, 0.0, Inf))
    transform = decomposition.vectors * Diagonal(scales)

    angles = range(0.0, 2π; length = n_points)
    circle = [cos.(angles)'; sin.(angles)']
    ellipse = reshape(mean_vector, :, 1) .+ Float64(n_sigma) .* transform * circle
    return ellipse[1, :], ellipse[2, :], mean_vector
end

function _ellipsoid_surface(points::AbstractMatrix{<:Real}; n_sigma::Real=THREE_SIGMA, n_azimuth::Int=49, n_elevation::Int=25)
    if size(points, 1) < 2
        return nothing
    end
    mean_vector = vec(mean(points; dims = 1))
    covariance = Symmetric(_covariance_matrix(points))
    decomposition = eigen(covariance)
    scales = sqrt.(clamp.(decomposition.values, 0.0, Inf))
    transform = decomposition.vectors * Diagonal(scales)

    azimuth = range(0.0, 2π; length = n_azimuth)
    elevation = range(-π / 2, π / 2; length = n_elevation)
    x_surface = Matrix{Float64}(undef, length(elevation), length(azimuth))
    y_surface = similar(x_surface)
    z_surface = similar(x_surface)

    for (i, elev) in enumerate(elevation)
        cos_elev = cos(elev)
        sin_elev = sin(elev)
        for (j, az) in enumerate(azimuth)
            unit_sphere = SVector(cos_elev * cos(az), cos_elev * sin(az), sin_elev)
            point = mean_vector .+ Float64(n_sigma) .* (transform * unit_sphere)
            x_surface[i, j] = point[1]
            y_surface[i, j] = point[2]
            z_surface[i, j] = point[3]
        end
    end

    return x_surface, y_surface, z_surface, mean_vector
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
        _monte_carlo_plot_kwargs()...,
    )
    monte_carlo_df = _filter_summary_rows(summary_df; include_open_loop = include_open_loop, run_type = "MonteCarlo")
    for controller_slug in unique(String.(monte_carlo_df.ControllerSlug))
        case_df = filter(row -> row.ControllerSlug == controller_slug, monte_carlo_df)
        longitudes_deg, latitudes_deg = _finite_xy(case_df, :FinalLongitude_deg, :FinalLatitude_deg)
        color = _summary_case_color(controller_slug)
        label = _summary_case_name(controller_slug)
        plot!(
            plt,
            longitudes_deg,
            latitudes_deg,
            seriestype=:scatter,
            color=color,
            label="$(label) landings",
            alpha=0.65,
        )
        mean_xy = _mean_xy(longitudes_deg, latitudes_deg)
        if mean_xy !== nothing
            mean_longitude_deg, mean_latitude_deg = mean_xy
            plot!(
                plt,
                [mean_longitude_deg],
                [mean_latitude_deg],
                seriestype=:scatter,
                color=color,
                markershape=:xcross,
                markersize=8,
                label="$(label) mean",
            )
        end
    end
    plot!(plt, [rad2deg(target_states.longitude)], [rad2deg(target_states.latitude)], seriestype=:scatter, color=:black, markershape=:diamond, markersize=8, label="Target")
    _save_plot_and_cleanup!(plt, filename)
end

function plot_landing_locations_with_ellipses(
    summary_df::DataFrame;
    include_open_loop::Bool=true,
    filename::String=include_open_loop ?
        "monte_carlo_final_landing_locations_3sigma_ellipses.pdf" :
        "monte_carlo_final_landing_locations_3sigma_ellipses_without_open_loop.pdf",
    title::String=include_open_loop ?
        "Monte Carlo Simulations: Final Landing Locations With 3-Sigma Ellipses" :
        "Monte Carlo Simulations: Final Landing Locations With 3-Sigma Ellipses (Without Open Loop)",
)
    plt = plot(
        xlabel="Longitude (deg)",
        ylabel="Latitude (deg)",
        _monte_carlo_plot_kwargs()...,
    )
    monte_carlo_df = _filter_summary_rows(summary_df; include_open_loop = include_open_loop, run_type = "MonteCarlo")
    for controller_slug in unique(String.(monte_carlo_df.ControllerSlug))
        case_df = filter(row -> row.ControllerSlug == controller_slug, monte_carlo_df)
        longitudes_deg, latitudes_deg = _finite_xy(case_df, :FinalLongitude_deg, :FinalLatitude_deg)
        color = _summary_case_color(controller_slug)
        label = _summary_case_name(controller_slug)
        if !isempty(longitudes_deg)
            plot!(
                plt,
                longitudes_deg,
                latitudes_deg,
                seriestype=:scatter,
                color=color,
                label="$(label) landings",
                alpha=0.45,
                markersize=3,
            )
        end
        mean_xy = _mean_xy(longitudes_deg, latitudes_deg)
        if mean_xy !== nothing
            mean_longitude_deg, mean_latitude_deg = mean_xy
            plot!(
                plt,
                [mean_longitude_deg],
                [mean_latitude_deg],
                seriestype=:scatter,
                color=color,
                markershape=:xcross,
                markersize=6,
                label="$(label) mean",
            )
        end
        if length(longitudes_deg) >= 2
            ellipse = _ellipse_outline(hcat(longitudes_deg, latitudes_deg))
            if ellipse !== nothing
                ellipse_x, ellipse_y, _ = ellipse
                plot!(
                    plt,
                    ellipse_x,
                    ellipse_y,
                    color=color,
                    linewidth=2,
                    label="$(label) 3-sigma ellipse",
                )
            end
        end
    end
    plot!(
        plt,
        [rad2deg(target_states.longitude)],
        [rad2deg(target_states.latitude)],
        seriestype=:scatter,
        color=:black,
        markershape=:diamond,
        markersize=8,
        label="Target",
    )
    _save_plot_and_cleanup!(plt, filename)
end

function plot_final_cartesian_locations(
    summary_df::DataFrame;
    include_open_loop::Bool=true,
    include_open_loop_nominal::Bool=true,
    filename::String=include_open_loop_nominal ?
        "monte_carlo_final_cartesian_locations.pdf" :
        "monte_carlo_final_cartesian_locations_without_open_loop_mean.pdf",
    title::String=include_open_loop_nominal ?
        "Monte Carlo Simulations: Final Cartesian Locations" :
        "Monte Carlo Simulations: Final Cartesian Locations (Without Open Loop Mean)",
)
    plt = plot(
        xlabel="X (km)",
        ylabel="Y (km)",
        _monte_carlo_plot_kwargs()...,
    )
    monte_carlo_df = _filter_summary_rows(summary_df; include_open_loop = include_open_loop, run_type = "MonteCarlo")
    for controller_slug in unique(String.(monte_carlo_df.ControllerSlug))
        case_df = filter(row -> row.ControllerSlug == controller_slug, monte_carlo_df)
        x_positions_km, y_positions_km = _finite_xy(case_df, :FinalPositionX_km, :FinalPositionY_km)
        color = _summary_case_color(controller_slug)
        label = _summary_case_name(controller_slug)
        plot!(
            plt,
            x_positions_km,
            y_positions_km,
            seriestype=:scatter,
            color=color,
            label="$(label) final positions",
            alpha=0.65,
        )
        mean_xy = _mean_xy(x_positions_km, y_positions_km)
        if mean_xy !== nothing && (include_open_loop_nominal || controller_slug != "open_loop")
            mean_x_km, mean_y_km = mean_xy
            plot!(
                plt,
                [mean_x_km],
                [mean_y_km],
                seriestype=:scatter,
                color=color,
                markershape=:xcross,
                markersize=8,
                label="$(label) mean",
            )
        end
    end
    target_x, target_y, _ = latlonalt_to_cartesian_km(target_states.latitude, target_states.longitude, target_states.altitude, R)
    plot!(plt, [target_x], [target_y], seriestype=:scatter, color=:black, markershape=:diamond, markersize=8, label="Target")
    _save_plot_and_cleanup!(plt, filename)
end

function plot_final_cartesian_locations_with_ellipses(
    summary_df::DataFrame;
    include_open_loop::Bool=true,
    include_open_loop_nominal::Bool=true,
    filename::String=include_open_loop ?
        "monte_carlo_final_cartesian_locations_3sigma_ellipses.pdf" :
        "monte_carlo_final_cartesian_locations_3sigma_ellipses_without_open_loop.pdf",
    title::String=include_open_loop ?
        "Monte Carlo Simulations: Final Cartesian Locations With 3-Sigma Ellipses" :
        "Monte Carlo Simulations: Final Cartesian Locations With 3-Sigma Ellipses (Without Open Loop)",
)
    plt = plot(
        xlabel="X (km)",
        ylabel="Y (km)",
        _monte_carlo_plot_kwargs()...,
    )
    monte_carlo_df = _filter_summary_rows(summary_df; include_open_loop = include_open_loop, run_type = "MonteCarlo")
    for controller_slug in unique(String.(monte_carlo_df.ControllerSlug))
        case_df = filter(row -> row.ControllerSlug == controller_slug, monte_carlo_df)
        x_positions_km, y_positions_km = _finite_xy(case_df, :FinalPositionX_km, :FinalPositionY_km)
        color = _summary_case_color(controller_slug)
        label = _summary_case_name(controller_slug)
        if !isempty(x_positions_km)
            plot!(
                plt,
                x_positions_km,
                y_positions_km,
                seriestype=:scatter,
                color=color,
                label="$(label) final positions",
                alpha=0.45,
                markersize=3,
            )
        end
        mean_xy = _mean_xy(x_positions_km, y_positions_km)
        if mean_xy !== nothing && (include_open_loop_nominal || controller_slug != "open_loop")
            mean_x_km, mean_y_km = mean_xy
            plot!(
                plt,
                [mean_x_km],
                [mean_y_km],
                seriestype=:scatter,
                color=color,
                markershape=:xcross,
                markersize=6,
                label="$(label) mean",
            )
        end
        if length(x_positions_km) >= 2
            ellipse = _ellipse_outline(hcat(x_positions_km, y_positions_km))
            if ellipse !== nothing
                ellipse_x, ellipse_y, _ = ellipse
                plot!(
                    plt,
                    ellipse_x,
                    ellipse_y,
                    color=color,
                    linewidth=2,
                    label="$(label) 3-sigma ellipse",
                )
            end
        end
    end
    target_x, target_y, _ = latlonalt_to_cartesian_km(target_states.latitude, target_states.longitude, target_states.altitude, R)
    plot!(
        plt,
        [target_x],
        [target_y],
        seriestype=:scatter,
        color=:black,
        markershape=:diamond,
        markersize=8,
        label="Target",
    )
    _save_plot_and_cleanup!(plt, filename)
end

function plot_state_profile_bundle(
    results::Vector{MonteCarloResults},
    profile_field::Symbol;
    title::String,
    ylabel::String,
    filename::String,
    reference_times::Union{Nothing, AbstractVector{<:Real}}=nothing,
    reference_values::Union{Nothing, AbstractVector{<:Real}}=nothing,
    include_open_loop::Bool=true,
)
    plt = plot(
        xlabel="Time (s)",
        ylabel=ylabel,
        _monte_carlo_plot_kwargs()...,
    )
    if reference_times !== nothing && reference_values !== nothing
        plot!(
            plt,
            Float64.(reference_times),
            Float64.(reference_values),
            seriestype = :steppost,
            color=:black,
            linewidth=2,
            label="Reference",
        )
    end
    for result in results
        if !include_open_loop && result.case.slug == "open_loop"
            continue
        end
        valid = valid_profile_indices(result, profile_field; time_field = :times)
        for (j, i) in enumerate(valid)
            plot!(
                plt,
                something(result.times[i]),
                something(getfield(result, profile_field)[i]),
                color=result.case.color,
                alpha=0.15,
                linewidth=1,
                label=j == 1 ? "$(result.case.name) MC" : false,
            )
        end
    end
    _save_plot_and_cleanup!(plt, filename)
end

function plot_control_profile_bundle(
    results::Vector{MonteCarloResults},
    profile_field::Symbol;
    title::String,
    ylabel::String,
    filename::String,
    reference_times::Union{Nothing, AbstractVector{<:Real}}=nothing,
    reference_values::Union{Nothing, AbstractVector{<:Real}}=nothing,
    include_open_loop::Bool=true,
)
    plt = plot(
        xlabel="Time (s)",
        ylabel=ylabel,
        _monte_carlo_plot_kwargs()...,
    )
    if reference_times !== nothing && reference_values !== nothing
        plot!(
            plt,
            Float64.(reference_times),
            Float64.(reference_values),
            color=:black,
            linewidth=2,
            label="Reference",
        )
    end
    for result in results
        if !include_open_loop && result.case.slug == "open_loop"
            continue
        end
        valid = valid_profile_indices(result, profile_field; time_field = :control_times)
        for (j, i) in enumerate(valid)
            plot!(
                plt,
                something(result.control_times[i]),
                something(getfield(result, profile_field)[i]),
                seriestype = :steppost,
                color=result.case.color,
                alpha=0.15,
                linewidth=1,
                label=j == 1 ? "$(result.case.name) MC" : false,
            )
        end
    end
    _save_plot_and_cleanup!(plt, filename)
end

function plot_mc_state_profiles_combined(
    results::Vector{MonteCarloResults};
    include_open_loop::Bool=false,
    filename_part1::String="monte_carlo_state_profiles_combined_without_open_loop_part1.pdf",
    filename_part2::String="monte_carlo_state_profiles_combined_without_open_loop_part2.pdf",
    title::String="Monte Carlo State Profiles (No Open Loop)",
)
    state_specs = (
        (:altitude_profiles, "Altitude (km)", Float64.(optimal_control.Time_s), Float64.(optimal_control.Altitude_100km .* 1e5 ./ 1e3), "Altitude"),
        (:longitude_profiles, "Longitude (deg)", Float64.(optimal_control.Time_s), Float64.(optimal_control.Longitude_deg), "Longitude"),
        (:latitude_profiles, "Latitude (deg)", Float64.(optimal_control.Time_s), Float64.(optimal_control.Latitude_deg), "Latitude"),
        (:velocity_profiles, "Velocity (km/s)", Float64.(optimal_control.Time_s), Float64.(optimal_control.Velocity_1000mps .* velocity_scale ./ 1e3), "Velocity"),
        (:flight_path_profiles, "Flight Path Angle (deg)", Float64.(optimal_control.Time_s), Float64.(optimal_control.FlightPath_deg), "Flight Path Angle"),
        (:azimuth_profiles, "Azimuth (deg)", Float64.(optimal_control.Time_s), Float64.(optimal_control.Azimuth_deg), "Azimuth"),
    )

    subplots = Any[]
    for (profile_field, ylabel, reference_times, reference_values, subplot_title) in state_specs
        plt = plot(
            xlabel="Time (s)",
            ylabel=ylabel,
            _monte_carlo_subplot_kwargs()...,
        )
        plot!(
            plt,
            reference_times,
            reference_values,
            seriestype = :steppost,
            color=:black,
            linewidth=2,
            label="Reference",
        )
        for result in results
            if !include_open_loop && result.case.slug == "open_loop"
                continue
            end
            valid = valid_profile_indices(result, profile_field; time_field = :times)
            for (j, i) in enumerate(valid)
                plot!(
                    plt,
                    something(result.times[i]),
                    something(getfield(result, profile_field)[i]),
                    color=result.case.color,
                    alpha=0.15,
                    linewidth=1,
                    label=j == 1 ? "$(result.case.name) MC" : false,
                )
            end
        end
        push!(subplots, plt)
    end

    fig_part1 = plot(
        subplots[1],
        subplots[2],
        subplots[3];
        layout=(3, 1),
        size=IEEE_SINGLE_COLUMN_STATE_SIZE,
        margin=10Plots.mm,
    )
    _save_plot_and_cleanup!(fig_part1, filename_part1)

    fig_part2 = plot(
        subplots[4],
        subplots[5],
        subplots[6];
        layout=(3, 1),
        size=IEEE_SINGLE_COLUMN_STATE_SIZE,
        margin=10Plots.mm,
    )
    _save_plot_and_cleanup!(fig_part2, filename_part2)
end

function plot_mc_control_profiles_combined(
    results::Vector{MonteCarloResults};
    include_open_loop::Bool=false,
    filename::String="monte_carlo_control_profiles_combined_without_open_loop.pdf",
    title::String="Monte Carlo Control Profiles (No Open Loop)",
)
    control_specs = (
        (:alpha_profiles, "Angle of Attack (deg)", Float64.(optimal_control.Time_s), Float64.(optimal_control.AngleOfAttack_deg), "Angle of Attack"),
        (:beta_profiles, "Bank Angle (deg)", Float64.(optimal_control.Time_s), Float64.(optimal_control.BankAngle_deg), "Bank Angle"),
    )

    subplots = Any[]
    for (profile_field, ylabel, reference_times, reference_values, subplot_title) in control_specs
        plt = plot(
            xlabel="Time (s)",
            ylabel=ylabel,
            _monte_carlo_subplot_kwargs(legend = true)...,
        )
        plot!(
            plt,
            reference_times,
            reference_values,
            color=:black,
            linewidth=2,
            label="Reference",
        )
        for result in results
            if !include_open_loop && result.case.slug == "open_loop"
                continue
            end
            valid = valid_profile_indices(result, profile_field; time_field = :control_times)
            for (j, i) in enumerate(valid)
                plot!(
                    plt,
                    something(result.control_times[i]),
                    something(getfield(result, profile_field)[i]),
                    seriestype = :steppost,
                    color=result.case.color,
                    alpha=0.15,
                    linewidth=1,
                    label=j == 1 ? "$(result.case.name) MC" : false,
                )
            end
        end
        push!(subplots, plt)
    end

    fig = plot(
        subplots...;
        layout=(2, 1),
        size=(1220, 980),
        margin=10Plots.mm,
    )
    _save_plot_and_cleanup!(fig, filename)
end

function plot_cartesian_ground_track_profiles(
    results::Vector{MonteCarloResults};
    include_reference::Bool=true,
    include_open_loop::Bool=true,
    filename::String=include_reference ?
        "monte_carlo_cartesian_ground_track.pdf" :
        (include_open_loop ?
            "monte_carlo_cartesian_ground_track_no_reference.pdf" :
            "monte_carlo_cartesian_ground_track_no_reference_without_open_loop.pdf"),
    title::String=include_reference ?
        "Monte Carlo Simulations: Cartesian Ground Track" :
        (include_open_loop ?
            "Monte Carlo Simulations: Cartesian Ground Track (No Reference)" :
            "Monte Carlo Simulations: Cartesian Ground Track (No Reference, No Open Loop)"),
)
    plt = plot(
        xlabel="X (km)",
        ylabel="Y (km)",
        _monte_carlo_plot_kwargs()...,
    )
    if include_reference
        plot!(
            plt,
            optimal_ref_x_km,
            optimal_ref_y_km,
            color=:black,
            linewidth=2,
            label="Reference",
        )
    end
    for result in results
        if !include_open_loop && result.case.slug == "open_loop"
            continue
        end
        valid = valid_profile_indices(result, :x_profiles; time_field = :times)
        for (j, i) in enumerate(valid)
            plot!(
                plt,
                something(result.x_profiles[i]),
                something(result.y_profiles[i]),
                color=result.case.color,
                alpha=0.15,
                linewidth=1,
                label=j == 1 ? "$(result.case.name) MC" : false,
            )
        end
    end
    target_x, target_y, _ = latlonalt_to_cartesian_km(target_states.latitude, target_states.longitude, target_states.altitude, R)
    plot!(
        plt,
        [target_x],
        [target_y],
        seriestype=:scatter,
        color=:black,
        markershape=:diamond,
        markersize=7,
        label="Target",
    )
    _save_plot_and_cleanup!(plt, filename)
end

function plot_mc_state_errors(results::Vector{MonteCarloResults})
    plt_alt = plot(xlabel="Time (s)", ylabel="Error (km)"; _monte_carlo_subplot_kwargs()...)
    plt_lon = plot(xlabel="Time (s)", ylabel="Error (deg)"; _monte_carlo_subplot_kwargs()...)
    plt_lat = plot(xlabel="Time (s)", ylabel="Error (deg)"; _monte_carlo_subplot_kwargs()...)
    plt_vel = plot(xlabel="Time (s)", ylabel="Error (km/s)"; _monte_carlo_subplot_kwargs()...)
    plt_fpa = plot(xlabel="Time (s)", ylabel="Error (deg)"; _monte_carlo_subplot_kwargs()...)
    plt_azi = plot(xlabel="Time (s)", ylabel="Error (deg)"; _monte_carlo_subplot_kwargs()...)

    for result in results
        valid = valid_profile_indices(
            result,
            (
                :altitude_profiles,
                :longitude_profiles,
                :latitude_profiles,
                :velocity_profiles,
                :flight_path_profiles,
                :azimuth_profiles,
            );
            time_field = :times,
        )
        for i in valid
            t = something(result.times[i])
            plot!(plt_alt, t, something(result.altitude_profiles[i]) .- (interp_altitude.(t) ./ 1e3), color=result.case.color, alpha=0.15, linewidth=1, label=false)
            plot!(plt_lon, t, something(result.longitude_profiles[i]) .- rad2deg.(interp_longitude.(t)), color=result.case.color, alpha=0.15, linewidth=1, label=false)
            plot!(plt_lat, t, something(result.latitude_profiles[i]) .- rad2deg.(interp_latitude.(t)), color=result.case.color, alpha=0.15, linewidth=1, label=false)
            plot!(plt_vel, t, something(result.velocity_profiles[i]) .- (interp_velocity.(t) ./ 1e3), color=result.case.color, alpha=0.15, linewidth=1, label=false)
            plot!(plt_fpa, t, something(result.flight_path_profiles[i]) .- rad2deg.(interp_flight_path.(t)), color=result.case.color, alpha=0.15, linewidth=1, label=false)
            plot!(plt_azi, t, something(result.azimuth_profiles[i]) .- rad2deg.(interp_azimuth.(t)), color=result.case.color, alpha=0.15, linewidth=1, label=false)
        end
    end

    fig_part1 = plot(
        plt_alt,
        plt_lon,
        plt_lat,
        layout=(3, 1),
        size=IEEE_SINGLE_COLUMN_STATE_SIZE,
        margin=5Plots.mm,
    )
    _save_plot_and_cleanup!(fig_part1, "monte_carlo_state_errors_part1.pdf")

    fig_part2 = plot(
        plt_vel,
        plt_fpa,
        plt_azi,
        layout=(3, 1),
        size=IEEE_SINGLE_COLUMN_STATE_SIZE,
        margin=5Plots.mm,
    )
    _save_plot_and_cleanup!(fig_part2, "monte_carlo_state_errors_part2.pdf")
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

function cleanup_plots_dir!()
    mkpath(PLOTS_DIR)
    for name in readdir(PLOTS_DIR)
        rm(joinpath(PLOTS_DIR, name); recursive=true, force=true)
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
        _monte_carlo_plot_kwargs()...,
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
    _save_plot_and_cleanup!(plt, filename)
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
        _monte_carlo_plot_kwargs()...,
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
    _save_plot_and_cleanup!(plt, filename)
end

function generate_all_monte_carlo_plots(summary_df::DataFrame)
    plot_landing_locations(summary_df; include_open_loop=true)
    plot_landing_locations(summary_df; include_open_loop=false)
    plot_landing_locations_with_ellipses(summary_df; include_open_loop=true)
    plot_landing_locations_with_ellipses(summary_df; include_open_loop=false)
    plot_final_cartesian_locations(summary_df; include_open_loop=true, include_open_loop_nominal=true)
    plot_final_cartesian_locations(
        summary_df;
        include_open_loop=false,
        include_open_loop_nominal=false,
        filename="monte_carlo_final_cartesian_locations_without_open_loop_mean.pdf",
        title="Monte Carlo Simulations: Final Cartesian Locations (Without Open Loop Mean)",
    )
    plot_final_cartesian_locations_with_ellipses(summary_df; include_open_loop=true, include_open_loop_nominal=true)
    plot_final_cartesian_locations_with_ellipses(
        summary_df;
        include_open_loop=false,
        include_open_loop_nominal=false,
        filename="monte_carlo_final_cartesian_locations_3sigma_ellipses_without_open_loop.pdf",
        title="Monte Carlo Simulations: Final Cartesian Locations With 3-Sigma Ellipses (Without Open Loop)",
    )
    plot_final_cartesian_position_error_norms(summary_df; include_open_loop=false)
    plot_final_cartesian_velocity_error_norms(summary_df; include_open_loop=true)
    plot_final_cartesian_velocity_error_norms(summary_df; include_open_loop=false)
    return nothing
end

function generate_all_profile_plots(results::Vector{MonteCarloResults})
    plot_state_profile_bundle(
        results,
        :altitude_profiles;
        title="Monte Carlo Simulations: Altitude Profiles",
        ylabel="Altitude (km)",
        filename="monte_carlo_altitude_profiles.pdf",
        reference_times=optimal_control.Time_s,
        reference_values=optimal_control.Altitude_100km .* 1e5 ./ 1e3,
    )
    plot_state_profile_bundle(
        results,
        :velocity_profiles;
        title="Monte Carlo Simulations: Velocity Profiles",
        ylabel="Velocity (km/s)",
        filename="monte_carlo_velocity_profiles.pdf",
        reference_times=optimal_control.Time_s,
        reference_values=optimal_control.Velocity_1000mps .* velocity_scale ./ 1e3,
    )
    plot_cartesian_ground_track_profiles(results; include_reference=true, include_open_loop=true)
    plot_cartesian_ground_track_profiles(results; include_reference=false, include_open_loop=true)
    plot_cartesian_ground_track_profiles(results; include_reference=false, include_open_loop=false)
    plot_control_profile_bundle(
        results,
        :alpha_profiles;
        title="Monte Carlo Simulations: Angle of Attack Profiles",
        ylabel="Angle of Attack (deg)",
        filename="monte_carlo_angle_of_attack_profiles.pdf",
        reference_times=optimal_control.Time_s,
        reference_values=optimal_control.AngleOfAttack_deg,
    )
    plot_control_profile_bundle(
        results,
        :beta_profiles;
        title="Monte Carlo Simulations: Bank Angle Profiles",
        ylabel="Bank Angle (deg)",
        filename="monte_carlo_bank_angle_profiles.pdf",
        reference_times=optimal_control.Time_s,
        reference_values=optimal_control.BankAngle_deg,
    )
    plot_control_profile_bundle(
        results,
        :heat_rate_profiles;
        title="Monte Carlo Simulations: Heat Rate Profiles",
        ylabel="Heat Rate (W/m^2)",
        filename="monte_carlo_heat_rate_profiles.pdf",
        reference_times=optimal_control.Time_s,
        reference_values=reference_heat_rate,
    )
    plot_mc_state_profiles_combined(results; include_open_loop=false)
    plot_mc_control_profiles_combined(results; include_open_loop=false)
    plot_mc_state_errors(results)
    return nothing
end

mkpath(PLOTS_DIR)
cleanup_plots_dir!()
summary_csv_path = joinpath(DATA_DIR, "summary.csv")

if !REPLOT_ONLY
    cleanup_data_dir!()
    case_outputs = [run_monte_carlo_case(case) for case in CONTROLLER_CASES]
    results = first.(case_outputs)
    nominal_solutions = getindex.(case_outputs, 2)
    summary_dfs = getindex.(case_outputs, 3)
    CSV.write(summary_csv_path, vcat(summary_dfs...))
    cleanup_data_dir!()
elseif !isfile(summary_csv_path)
    error("SPARC_MC_REPLOT_ONLY=true but $summary_csv_path does not exist")
end

summary_df = load_summary_table()
generate_all_monte_carlo_plots(summary_df)
if !REPLOT_ONLY
    generate_all_profile_plots(results)
else
    println("Skipping profile plots in replot-only mode because summary.csv does not contain trajectory histories.")
end
