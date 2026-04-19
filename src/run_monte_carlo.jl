include("model/SimulatorModel.jl")

using .SimulatorModel
using StaticArrays
using Plots
using DifferentialEquations
using ProgressMeter
using CSV
using DataFrames
using Interpolations

gr()

const NUM_SIMULATIONS = parse(Int, get(ENV, "SPARC_MC_RUNS", "100"))
const OUTPUT_DIR = get(ENV, "SPARC_MC_OUTPUT_DIR", ".")
const RUN_NOMINAL_OVERLAY = parse(Bool, get(ENV, "SPARC_MC_NOMINAL_OVERLAY", "true"))

h0 = 125000.0
ϕ0 = deg2rad(126.7)
θ0 = deg2rad(-3.93)
v0 = 5845.39
γ0 = deg2rad(-15.49)
ψ0 = deg2rad(90.0)

u0 = MVector{7, Float64}(h0, ϕ0, θ0, v0, γ0, ψ0, 0.0)
tspan = (0.0, 500.0)

mass = 3257.0
area = 15.904
const μ = 3.986004418e14
R = 6378137.0

optimal_control = CSV.read("optimal_trajectory.csv", DataFrame)
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

target_altitude = 11848.0
target_velocity = 500.0
target_γ = deg2rad(-5.0)
target_states = TargetStates(altitude=target_altitude, longitude=deg2rad(137.4), latitude=deg2rad(-4.5), velocity=target_velocity, flight_path_angle=target_γ)

function build_edl_params(; disturbance::Bool, monte_carlo::Bool)
    atmosphere = GramAtmosphere("GRAMpy/", "GRAM_Data", monte_carlo, "earth", DateTime(2012, 8, 6, 5, 10, 46.0))
    mpc_params = MPCParams{100, 7, 8, 0.1}(
        n_horizon=100,
        time_step=0.75,
        H_SCALE=1.0e5,
        V_SCALE=1.0e4,
        T_SCALE=1.0,
        n_exp=4.512,
        m_exp=0.82958,
        learning_rate=0.1,
    )
    return EDLParams(
        mass=mass,
        area=area,
        μ=μ,
        R=R,
        control_function=trackingmpc_shrinking,
        β=deg2rad(interp_optimal_control(tspan[1])),
        α=deg2rad(interp_optimal_alpha(tspan[1])),
        atmospheric_density_function=(LatLonAlt, t) -> atmospheric_density(LatLonAlt, t, atmosphere, disturbance),
        atmospheric_density=0.0,
        wind=SVector{3, Float64}(zeros(3)),
        target_states=target_states,
        optimization_states=OptimizationStates(),
        nominal_trajectory=optimal_trajectory,
        cache=EDLCache(),
        mpc_params=mpc_params,
    )
end

function run_case(case_index::Int; disturbance::Bool, monte_carlo::Bool, dt::Float64)
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
    callbacks = CallbackSet(altitude_termination_condition, atmospheric_density_callback, local_saving_callback, control_callback)
    params = build_edl_params(disturbance=disturbance, monte_carlo=monte_carlo)
    prob = ODEProblem(edl_dynamics, copy(u0), tspan, params, callback=callbacks)
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

mkpath(OUTPUT_DIR)

nominal_sol = nothing
if RUN_NOMINAL_OVERLAY
    nominal_sol, _ = run_case(0; disturbance=false, monte_carlo=false, dt=0.5)
end

final_latitudes = fill(NaN, NUM_SIMULATIONS)
final_longitudes = fill(NaN, NUM_SIMULATIONS)
altitude_profiles = Vector{Union{Nothing, Vector{Float64}}}(undef, NUM_SIMULATIONS)
velocity_profiles = Vector{Union{Nothing, Vector{Float64}}}(undef, NUM_SIMULATIONS)
times = Vector{Union{Nothing, Vector{Float64}}}(undef, NUM_SIMULATIONS)
fill!(altitude_profiles, nothing)
fill!(velocity_profiles, nothing)
fill!(times, nothing)

@showprogress for sim in 1:NUM_SIMULATIONS
    try
        sol_mc, local_saved_values = run_case(sim; disturbance=true, monte_carlo=true, dt=0.5)
        altitude_profiles[sim] = getindex.(sol_mc.u, 1) ./ 1e3
        velocity_profiles[sim] = getindex.(sol_mc.u, 4) ./ 1e3
        times[sim] = sol_mc.t
        final_latitudes[sim] = rad2deg(getindex(sol_mc.u[end], 3))
        final_longitudes[sim] = rad2deg(getindex(sol_mc.u[end], 2))

        _, betas, alphas, heat_rates = saved_value_vectors(local_saved_values)

        trajectory_df = DataFrame(
            Time_s=sol_mc.t,
            Altitude_100km=altitude_profiles[sim],
            Longitude_fin_deg=rad2deg.(getindex.(sol_mc.u, 2)),
            Latitude_fin_deg=rad2deg.(getindex.(sol_mc.u, 3)),
            Velocity_1000mps=velocity_profiles[sim],
            FlightPath_deg=rad2deg.(getindex.(sol_mc.u, 5)),
            Azimuth_deg=rad2deg.(getindex.(sol_mc.u, 6)),
            HeatLoad_Jm2=getindex.(sol_mc.u, 7),
        )
        CSV.write(joinpath(OUTPUT_DIR, "optimal_trajectory_mpc_$(sim).csv"), trajectory_df)

        control_df = DataFrame(
            Time_s=local_saved_values.t,
            AngleOfAttack_deg=rad2deg.(alphas),
            BankAngle_deg=rad2deg.(betas),
            HeatRate_Wm2=heat_rates,
        )
        CSV.write(joinpath(OUTPUT_DIR, "optimal_trajectory_mpc_$(sim)_control.csv"), control_df)
    catch e
        @warn "Monte Carlo case failed" sim exception=(e, catch_backtrace())
    end
end

valid = findall(i -> times[i] !== nothing && altitude_profiles[i] !== nothing && velocity_profiles[i] !== nothing, 1:NUM_SIMULATIONS)

plt1 = plot(title="Monte Carlo Simulations: Altitude Profiles", xlabel="Time (s)", ylabel="Altitude (km)", legend=true)
for i in valid
    plot!(plt1, something(times[i]), something(altitude_profiles[i]), color=:grey, alpha=0.3, label=false)
end
if nominal_sol !== nothing
    plot!(plt1, nominal_sol.t, getindex.(nominal_sol.u, 1) ./ 1e3, color=:red, label="Nominal Trajectory")
end
display(plt1)
savefig(plt1, joinpath(OUTPUT_DIR, "monte_carlo_altitude_profiles.pdf"))

plt2 = plot(title="Monte Carlo Simulations: Velocity Profiles", xlabel="Time (s)", ylabel="Velocity (km/s)", legend=true)
for i in valid
    plot!(plt2, something(times[i]), something(velocity_profiles[i]), color=:grey, alpha=0.3, label=false)
end
if nominal_sol !== nothing
    plot!(plt2, nominal_sol.t, getindex.(nominal_sol.u, 4) ./ 1e3, color=:red, label="Nominal Trajectory")
end
display(plt2)
savefig(plt2, joinpath(OUTPUT_DIR, "monte_carlo_velocity_profiles.pdf"))

plt3 = plot(
    final_longitudes[valid],
    final_latitudes[valid],
    seriestype=:scatter,
    xlabel="Longitude (deg)",
    ylabel="Latitude (deg)",
    title="Monte Carlo Simulations: Final Landing Locations",
    legend=true,
    label="Landing Locations",
    alpha=0.6,
)
if nominal_sol !== nothing
    plot!(
        plt3,
        [rad2deg(getindex(nominal_sol.u[end], 2))],
        [rad2deg(getindex(nominal_sol.u[end], 3))],
        seriestype=:scatter,
        color=:red,
        markershape=:star5,
        markersize=8,
        label="Nominal Landing Location",
    )
end
plot!(plt3, [rad2deg(target_states.longitude)], [rad2deg(target_states.latitude)], seriestype=:scatter, color=:blue, markershape=:diamond, markersize=8, label="Target Location")
display(plt3)
savefig(plt3, joinpath(OUTPUT_DIR, "monte_carlo_final_landing_locations.pdf"))
