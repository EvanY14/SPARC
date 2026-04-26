include("model/SimulatorModel.jl")

using .SimulatorModel
using StaticArrays
using DifferentialEquations
using CSV
using DataFrames
using Interpolations
using LinearAlgebra
using Printf

include("reference/trajectory_initialization.jl")

const TUNE_SIM_DT = parse(Float64, get(ENV, "SPARC_TUNE_SIM_DT", "0.2"))
const SCORE_VELOCITY_WEIGHT = parse(Float64, get(ENV, "SPARC_TUNE_VEL_WEIGHT", "4.0"))

struct TuneContext
    u0::MVector{7, Float64}
    tspan::Tuple{Float64, Float64}
    mass::Float64
    area::Float64
    μ::Float64
    R::Float64
    β0_ref::Float64
    α0_ref::Float64
    target_states::SimulatorModel.TargetStates
    nominal_trajectory::SVector{6, AbstractInterpolation}
    ref_final_state::SVector{6, Float64}
    ref_final_cartesian_position_km::SVector{3, Float64}
    ref_final_cartesian_velocity_mps::SVector{3, Float64}
end

@kwdef struct TerminalMetrics
    score::Float64
    position_error_km::Float64
    velocity_error_kms::Float64
    altitude_error_m::Float64
    longitude_error_deg::Float64
    latitude_error_deg::Float64
    speed_error_mps::Float64
    flight_path_error_deg::Float64
    azimuth_error_deg::Float64
    terminal_time_s::Float64
end

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

wrap_angle_rad(angle) = mod(angle + π, 2π) - π

function build_context()
    optimal_control = CSV.read("optimal_trajectory.csv", DataFrame)
    init = reference_initial_conditions(optimal_control)
    tspan = init.tspan

    u0 = MVector{7, Float64}(init.h0, init.ϕ0, init.θ0, init.v0, init.γ0, init.ψ0, 0.0)
    mass = SimulatorModel.VEHICLE.mass
    area = SimulatorModel.VEHICLE.reference_area
    μ = 3.986004418e14
    R = 6378137.0

    interp_optimal_control = linear_interpolation(optimal_control.Time_s, optimal_control.BankAngle_deg, extrapolation_bc = Line())
    interp_optimal_alpha = linear_interpolation(optimal_control.Time_s, optimal_control.AngleOfAttack_deg, extrapolation_bc = Line())
    β0_ref = deg2rad(interp_optimal_control(tspan[1]))
    α0_ref = deg2rad(interp_optimal_alpha(tspan[1]))

    velocity_scale = abs(optimal_control.Velocity_1000mps[1] * 1e3 - init.v0) <= abs(optimal_control.Velocity_1000mps[1] * 1e4 - init.v0) ? 1e3 : 1e4
    interp_altitude = linear_interpolation(optimal_control.Time_s, optimal_control.Altitude_100km .* 1e5, extrapolation_bc = Line())
    interp_velocity = linear_interpolation(optimal_control.Time_s, optimal_control.Velocity_1000mps .* velocity_scale, extrapolation_bc = Line())
    interp_longitude = linear_interpolation(optimal_control.Time_s, optimal_control.Longitude_deg .* (π / 180), extrapolation_bc = Line())
    interp_latitude = linear_interpolation(optimal_control.Time_s, optimal_control.Latitude_deg .* (π / 180), extrapolation_bc = Line())
    interp_flight_path = linear_interpolation(optimal_control.Time_s, optimal_control.FlightPath_deg .* (π / 180), extrapolation_bc = Line())
    interp_azimuth = linear_interpolation(optimal_control.Time_s, optimal_control.Azimuth_deg .* (π / 180), extrapolation_bc = Line())
    nominal_trajectory = SVector{6, AbstractInterpolation}(interp_altitude, interp_longitude, interp_latitude, interp_velocity, interp_flight_path, interp_azimuth)

    ref_final_state = SVector(
        Float64(optimal_control[end, :Altitude_100km]) * 1e5,
        deg2rad(Float64(optimal_control[end, :Longitude_deg])),
        deg2rad(Float64(optimal_control[end, :Latitude_deg])),
        Float64(optimal_control[end, :Velocity_1000mps]) * velocity_scale,
        deg2rad(Float64(optimal_control[end, :FlightPath_deg])),
        deg2rad(Float64(optimal_control[end, :Azimuth_deg])),
    )
    ref_final_cartesian_position_km = SVector(
        latlonalt_to_cartesian_km(ref_final_state[3], ref_final_state[2], ref_final_state[1], R)...,
    )
    ref_final_cartesian_velocity_mps = local_velocity_to_cartesian_mps(
        ref_final_state[2],
        ref_final_state[3],
        ref_final_state[4],
        ref_final_state[5],
        ref_final_state[6],
    )

    target_states = SimulatorModel.TargetStates(
        altitude = ref_final_state[1],
        longitude = ref_final_state[2],
        latitude = ref_final_state[3],
        velocity = ref_final_state[4],
        flight_path_angle = ref_final_state[5],
    )

    return TuneContext(
        u0,
        tspan,
        mass,
        area,
        μ,
        R,
        β0_ref,
        α0_ref,
        target_states,
        nominal_trajectory,
        ref_final_state,
        ref_final_cartesian_position_km,
        ref_final_cartesian_velocity_mps,
    )
end

function deterministic_density(LatLonAlt, _t)
    return SimulatorModel.earth_atmosphere_density(LatLonAlt[3]), SVector{3, Float64}(0.0, 0.0, 0.0)
end

function make_edl_params(ctx::TuneContext, control_function::Function)
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
        mass = ctx.mass,
        area = ctx.area,
        μ = ctx.μ,
        R = ctx.R,
        control_function = control_function,
        β = ctx.β0_ref,
        α = ctx.α0_ref,
        atmospheric_density_function = deterministic_density,
        atmospheric_density = 0.0,
        wind = SVector{3, Float64}(0.0, 0.0, 0.0),
        target_states = ctx.target_states,
        optimization_states = SimulatorModel.OptimizationStates(),
        nominal_trajectory = ctx.nominal_trajectory,
        cache = SimulatorModel.EDLCache(),
        mpc_params = mpc_params,
    )
end

function run_terminal_metrics(ctx::TuneContext, control_function::Function; sim_dt::Float64=TUNE_SIM_DT)
    params = make_edl_params(ctx, control_function)
    callbacks = CallbackSet(
        SimulatorModel.altitude_termination_condition,
        SimulatorModel.atmospheric_density_callback,
        SimulatorModel.control_callback,
    )
    prob = ODEProblem(SimulatorModel.edl_dynamics, copy(ctx.u0), ctx.tspan, params, callback = callbacks)
    sol = solve(prob, Tsit5(), dt = sim_dt, adaptive = false)

    x_final = SVector{6, Float64}(sol.u[end][1:6]...)
    position_final_km = SVector(
        latlonalt_to_cartesian_km(x_final[3], x_final[2], x_final[1], ctx.R)...,
    )
    velocity_final_mps = local_velocity_to_cartesian_mps(x_final[2], x_final[3], x_final[4], x_final[5], x_final[6])

    position_error_km = norm(position_final_km .- ctx.ref_final_cartesian_position_km)
    velocity_error_kms = norm(velocity_final_mps .- ctx.ref_final_cartesian_velocity_mps) / 1e3
    score = position_error_km + SCORE_VELOCITY_WEIGHT * velocity_error_kms

    return TerminalMetrics(
        score = score,
        position_error_km = position_error_km,
        velocity_error_kms = velocity_error_kms,
        altitude_error_m = abs(x_final[1] - ctx.ref_final_state[1]),
        longitude_error_deg = abs(rad2deg(wrap_angle_rad(x_final[2] - ctx.ref_final_state[2]))),
        latitude_error_deg = abs(rad2deg(wrap_angle_rad(x_final[3] - ctx.ref_final_state[3]))),
        speed_error_mps = abs(x_final[4] - ctx.ref_final_state[4]),
        flight_path_error_deg = abs(rad2deg(wrap_angle_rad(x_final[5] - ctx.ref_final_state[5]))),
        azimuth_error_deg = abs(rad2deg(wrap_angle_rad(x_final[6] - ctx.ref_final_state[6]))),
        terminal_time_s = sol.t[end],
    )
end

function print_metrics(label::AbstractString, metrics::TerminalMetrics)
    @printf(
        "%s: score=%.6f pos_err=%.6f km vel_err=%.6f km/s dh=%.3f m dlon=%.6f deg dlat=%.6f deg dv=%.6f m/s dγ=%.6f deg dψ=%.6f deg tf=%.3f s\n",
        label,
        metrics.score,
        metrics.position_error_km,
        metrics.velocity_error_kms,
        metrics.altitude_error_m,
        metrics.longitude_error_deg,
        metrics.latitude_error_deg,
        metrics.speed_error_mps,
        metrics.flight_path_error_deg,
        metrics.azimuth_error_deg,
        metrics.terminal_time_s,
    )
end

function evaluate_shared!(ctx::TuneContext, snapshot)
    SimulatorModel.set_mpg_tracking_tuning!(
        state_scales = snapshot.state_scales,
        n_horizon = snapshot.n_horizon,
        time_step = snapshot.time_step,
    )
    return run_terminal_metrics(ctx, SimulatorModel.mpg)
end

function evaluate_integral!(ctx::TuneContext, snapshot)
    SimulatorModel.set_mpg_integral_tuning!(
        state_normalized_weights = snapshot.state_normalized_weights,
        increment_weights = snapshot.increment_weights,
        state_gain = snapshot.state_gain,
        increment_gain = snapshot.increment_gain,
    )
    return run_terminal_metrics(ctx, SimulatorModel.mpg_integral)
end

function evaluate_sm!(ctx::TuneContext, snapshot)
    SimulatorModel.set_sm_mpg_tuning!(
        lambda = snapshot.lambda,
        sliding_gain = snapshot.sliding_gain,
        increment_gain = snapshot.increment_gain,
    )
    return run_terminal_metrics(ctx, SimulatorModel.sm_mpg)
end

function tune_shared!(ctx::TuneContext)
    current = SimulatorModel.mpg_tracking_tuning_snapshot()
    best_snapshot = (
        state_scales = Float64.(current.state_scales),
        n_horizon = current.n_horizon,
        time_step = current.time_step,
    )
    best_metrics = evaluate_shared!(ctx, best_snapshot)

    println("Tuning shared MPG parameters")
    print_metrics("Shared initial", best_metrics)

    coarse_horizons = unique(sort(Int.([30, 40, 50, 60, best_snapshot.n_horizon])))
    coarse_time_steps = unique(sort(Float64.([0.4, 0.5, 0.6, 0.7, best_snapshot.time_step])))
    for n_horizon in coarse_horizons, time_step in coarse_time_steps
        candidate = (
            state_scales = best_snapshot.state_scales,
            n_horizon = n_horizon,
            time_step = time_step,
        )
        metrics = evaluate_shared!(ctx, candidate)
        if metrics.score < best_metrics.score
            best_snapshot = candidate
            best_metrics = metrics
            print_metrics("Shared coarse best", best_metrics)
        end
    end

    for factors in ([0.7, 1.0, 1.3], [0.85, 1.0, 1.15])
        for idx in eachindex(best_snapshot.state_scales)
            local_best = best_snapshot
            local_metrics = best_metrics
            for factor in factors
                trial_scales = copy(best_snapshot.state_scales)
                trial_scales[idx] = best_snapshot.state_scales[idx] * factor
                candidate = (
                    state_scales = trial_scales,
                    n_horizon = best_snapshot.n_horizon,
                    time_step = best_snapshot.time_step,
                )
                metrics = evaluate_shared!(ctx, candidate)
                if metrics.score < local_metrics.score
                    local_best = candidate
                    local_metrics = metrics
                end
            end
            if local_metrics.score < best_metrics.score
                best_snapshot = local_best
                best_metrics = local_metrics
                print_metrics("Shared scale best", best_metrics)
            end
        end
    end

    fine_horizons = unique(sort(Int.([max(20, best_snapshot.n_horizon - 5), best_snapshot.n_horizon, best_snapshot.n_horizon + 5])))
    fine_time_steps = unique(sort(Float64.([max(0.3, best_snapshot.time_step - 0.1), best_snapshot.time_step, best_snapshot.time_step + 0.1])))
    for n_horizon in fine_horizons, time_step in fine_time_steps
        candidate = (
            state_scales = best_snapshot.state_scales,
            n_horizon = n_horizon,
            time_step = time_step,
        )
        metrics = evaluate_shared!(ctx, candidate)
        if metrics.score < best_metrics.score
            best_snapshot = candidate
            best_metrics = metrics
            print_metrics("Shared fine best", best_metrics)
        end
    end

    SimulatorModel.set_mpg_tracking_tuning!(
        state_scales = best_snapshot.state_scales,
        n_horizon = best_snapshot.n_horizon,
        time_step = best_snapshot.time_step,
    )
    return best_snapshot, best_metrics
end

function tune_integral!(ctx::TuneContext)
    stage_weights = vec(diag(SimulatorModel._mpg_stage_state_normalized_weight_matrix()))
    control_weights = vec(diag(SimulatorModel._mpg_control_normalized_weight_matrix()))
    current = SimulatorModel.mpg_integral_tuning_snapshot()
    best_snapshot = (
        state_normalized_weights = Float64.(stage_weights),
        increment_weights = Float64.(control_weights),
        state_gain = current.state_gain,
        increment_gain = current.increment_gain,
    )
    best_metrics = evaluate_integral!(ctx, best_snapshot)

    println("Tuning MPG-IA parameters")
    print_metrics("MPG-IA initial", best_metrics)

    for state_gain in [0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08]
        candidate = (
            state_normalized_weights = best_snapshot.state_normalized_weights,
            increment_weights = best_snapshot.increment_weights,
            state_gain = state_gain,
            increment_gain = best_snapshot.increment_gain,
        )
        metrics = evaluate_integral!(ctx, candidate)
        if metrics.score < best_metrics.score
            best_snapshot = candidate
            best_metrics = metrics
            print_metrics("MPG-IA state-gain best", best_metrics)
        end
    end

    for increment_gain in [0.0, 0.001, 0.002, 0.005, 0.01, 0.02]
        candidate = (
            state_normalized_weights = best_snapshot.state_normalized_weights,
            increment_weights = best_snapshot.increment_weights,
            state_gain = best_snapshot.state_gain,
            increment_gain = increment_gain,
        )
        metrics = evaluate_integral!(ctx, candidate)
        if metrics.score < best_metrics.score
            best_snapshot = candidate
            best_metrics = metrics
            print_metrics("MPG-IA increment-gain best", best_metrics)
        end
    end

    for state_gain in best_snapshot.state_gain .* [0.5, 1.0, 1.5]
        candidate = (
            state_normalized_weights = best_snapshot.state_normalized_weights,
            increment_weights = best_snapshot.increment_weights,
            state_gain = max(0.0, state_gain),
            increment_gain = best_snapshot.increment_gain,
        )
        metrics = evaluate_integral!(ctx, candidate)
        if metrics.score < best_metrics.score
            best_snapshot = candidate
            best_metrics = metrics
            print_metrics("MPG-IA state fine best", best_metrics)
        end
    end

    for increment_gain in best_snapshot.increment_gain .* [0.5, 1.0, 1.5]
        candidate = (
            state_normalized_weights = best_snapshot.state_normalized_weights,
            increment_weights = best_snapshot.increment_weights,
            state_gain = best_snapshot.state_gain,
            increment_gain = max(0.0, increment_gain),
        )
        metrics = evaluate_integral!(ctx, candidate)
        if metrics.score < best_metrics.score
            best_snapshot = candidate
            best_metrics = metrics
            print_metrics("MPG-IA increment fine best", best_metrics)
        end
    end

    SimulatorModel.set_mpg_integral_tuning!(
        state_normalized_weights = best_snapshot.state_normalized_weights,
        increment_weights = best_snapshot.increment_weights,
        state_gain = best_snapshot.state_gain,
        increment_gain = best_snapshot.increment_gain,
    )
    return best_snapshot, best_metrics
end

function tune_sm!(ctx::TuneContext)
    current = SimulatorModel.sm_mpg_tuning_snapshot()
    best_snapshot = (
        lambda = current.lambda,
        sliding_gain = current.sliding_gain,
        increment_gain = current.increment_gain,
    )
    best_metrics = evaluate_sm!(ctx, best_snapshot)

    println("Tuning MPG-SM parameters")
    print_metrics("MPG-SM initial", best_metrics)

    for lambda in [0.0, 0.1, 0.2, 0.3, 0.4, 0.6]
        candidate = (
            lambda = lambda,
            sliding_gain = best_snapshot.sliding_gain,
            increment_gain = best_snapshot.increment_gain,
        )
        metrics = evaluate_sm!(ctx, candidate)
        if metrics.score < best_metrics.score
            best_snapshot = candidate
            best_metrics = metrics
            print_metrics("MPG-SM lambda best", best_metrics)
        end
    end

    for sliding_gain in [0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08]
        candidate = (
            lambda = best_snapshot.lambda,
            sliding_gain = sliding_gain,
            increment_gain = best_snapshot.increment_gain,
        )
        metrics = evaluate_sm!(ctx, candidate)
        if metrics.score < best_metrics.score
            best_snapshot = candidate
            best_metrics = metrics
            print_metrics("MPG-SM slide best", best_metrics)
        end
    end

    for increment_gain in [0.0, 0.001, 0.002, 0.005, 0.01, 0.02]
        candidate = (
            lambda = best_snapshot.lambda,
            sliding_gain = best_snapshot.sliding_gain,
            increment_gain = increment_gain,
        )
        metrics = evaluate_sm!(ctx, candidate)
        if metrics.score < best_metrics.score
            best_snapshot = candidate
            best_metrics = metrics
            print_metrics("MPG-SM increment best", best_metrics)
        end
    end

    for lambda in clamp.(best_snapshot.lambda .+ [-0.05, 0.0, 0.05], 0.0, 1.5)
        candidate = (
            lambda = lambda,
            sliding_gain = best_snapshot.sliding_gain,
            increment_gain = best_snapshot.increment_gain,
        )
        metrics = evaluate_sm!(ctx, candidate)
        if metrics.score < best_metrics.score
            best_snapshot = candidate
            best_metrics = metrics
            print_metrics("MPG-SM lambda fine best", best_metrics)
        end
    end

    for sliding_gain in max.(0.0, best_snapshot.sliding_gain .* [0.5, 1.0, 1.5])
        candidate = (
            lambda = best_snapshot.lambda,
            sliding_gain = sliding_gain,
            increment_gain = best_snapshot.increment_gain,
        )
        metrics = evaluate_sm!(ctx, candidate)
        if metrics.score < best_metrics.score
            best_snapshot = candidate
            best_metrics = metrics
            print_metrics("MPG-SM slide fine best", best_metrics)
        end
    end

    for increment_gain in max.(0.0, best_snapshot.increment_gain .* [0.5, 1.0, 1.5])
        candidate = (
            lambda = best_snapshot.lambda,
            sliding_gain = best_snapshot.sliding_gain,
            increment_gain = increment_gain,
        )
        metrics = evaluate_sm!(ctx, candidate)
        if metrics.score < best_metrics.score
            best_snapshot = candidate
            best_metrics = metrics
            print_metrics("MPG-SM increment fine best", best_metrics)
        end
    end

    SimulatorModel.set_sm_mpg_tuning!(
        lambda = best_snapshot.lambda,
        sliding_gain = best_snapshot.sliding_gain,
        increment_gain = best_snapshot.increment_gain,
    )
    return best_snapshot, best_metrics
end

ctx = build_context()

println("Baseline controller metrics")
print_metrics("MPG baseline", run_terminal_metrics(ctx, SimulatorModel.mpg))
print_metrics("MPG-IA baseline", run_terminal_metrics(ctx, SimulatorModel.mpg_integral))
print_metrics("MPG-SM baseline", run_terminal_metrics(ctx, SimulatorModel.sm_mpg))

shared_snapshot, shared_metrics = tune_shared!(ctx)
integral_snapshot, integral_metrics = tune_integral!(ctx)
sm_snapshot, sm_metrics = tune_sm!(ctx)

println("\nBest snapshots")
println("Shared MPG = ", shared_snapshot)
println("MPG-IA = ", integral_snapshot)
println("MPG-SM = ", sm_snapshot)

println("\nFinal tuned metrics")
print_metrics("MPG tuned", run_terminal_metrics(ctx, SimulatorModel.mpg))
print_metrics("MPG-IA tuned", run_terminal_metrics(ctx, SimulatorModel.mpg_integral))
print_metrics("MPG-SM tuned", run_terminal_metrics(ctx, SimulatorModel.sm_mpg))
