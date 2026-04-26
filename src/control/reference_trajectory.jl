using JuMP
import Interpolations
import Ipopt
using CSV
using DataFrames
using Plots
const MOI = JuMP.MOI
include("../model/earth_atmosphere_polyfit.jl")
include("../model/vehicle.jl")
include("control_limits.jl")
plotly()
const DISPLAY_PLOTS = haskey(ENV, "DISPLAY") && !isempty(ENV["DISPLAY"]) && Sys.which("xdg-open") !== nothing

function maybe_display(plt)
    if DISPLAY_PLOTS
        display(plt)
    end
    return plt
end

# Global variables
const m = VEHICLE.mass    # mass (kg)

# Aerodynamic and atmospheric forces on the vehicle
const Rₑ = 6378137.0 # Radius of Earth (m, WGS84 equatorial)
const μ = 3.986004418e14 # Earth gravitational parameter (m^3/sec^2)
const S = VEHICLE.reference_area # Reference area (m^2)
# const cD = 1.46 # Drag coefficient
# const cL = 0.24 * cD # Lift coefficient

a₀ = -0.20704
a₁ = 0.029244
b₀ = 0.07854
b₁ = -0.61592e-2
b₂ = 0.621408e-3

const C1 = 8.53e-13 # Convective heat-rate coefficient for ρ in kg/m^3 and v in m/s
const n_exp = 0.82958 # Exponent for convective heat rate calculation
const m_exp = 4.512 # Exponent for convective heat rate calculation

# Initial conditions
const H_SCALE = 1e5  # Altitude scaling factor
const V_SCALE = 1e4  # Velocity scaling factor
const h_s = 125.0e3 / H_SCALE          # altitude (m) / 1e5
const ϕ_s = deg2rad(0.0)   # longitude (rad)
const θ_s = deg2rad(0.0)   # latitude (rad)
const v_s = 7200.0 / V_SCALE         # velocity (m/sec) / 1e4
const γ_s = deg2rad(-1.0)  # flight path angle (rad)
const ψ_s = deg2rad(90.0)  # azimuth (rad)
const α_s = deg2rad(0)   # angle of attack (rad)
const β_s = deg2rad(0)   # bank angle (rad)
const t_s = 0.33         # time step (sec)

# Final conditions, the so-called Terminal Area Energy Management (TAEM)
const h_t = 25000.0 / H_SCALE          # altitude (m) / 1e5
const v_t = 700.0 / V_SCALE         # velocity (m/sec) / 1e4
const γ_t = deg2rad(-5.0)  # flight path angle (rad)
const ϕ_t = deg2rad(25.0)  # target longitude (rad)
const θ_t = deg2rad(3.0)   # target latitude (rad)
const INITIAL_GUESS_PATH = joinpath(@__DIR__, "..", "..", "optimal_trajectory.csv")

# Number of mesh points (knots) to be used
const n = 503

# Integration scheme to be used for the dynamics
const integration_rule = "trapezoidal"  # "rectangular", "trapezoidal", or "rk4"
maybe_display(plot(
    earth_atmosphere_density.(1.0e3:1.0e3:100.0e3),
    1.0:1.0:100.0,
    title = "Earth Density vs Altitude",
    ylabel = "Altitude (km)",
    xlabel = "Density",
    xscale = :log10,
    linewidth = 2,
    size = (700, 500),
))

function load_initial_guess_from_csv(path::AbstractString, n_nodes::Integer)
    if !isfile(path)
        return nothing
    end

    df = CSV.read(path, DataFrame)
    n_rows = nrow(df)
    if n_rows < 2
        return nothing
    end

    sample_points = collect(range(1.0, n_rows, length = n_nodes))
    dt_samples = diff(Float64.(df.Time_s))
    dt_last = isempty(dt_samples) ? t_s : dt_samples[end]
    dt_source = vcat(dt_samples, dt_last)

    interp_column(values) = Interpolations.LinearInterpolation(1:n_rows, Float64.(values), extrapolation_bc = Interpolations.Line())

    altitude_guess = interp_column(df.Altitude_100km).(sample_points)
    longitude_guess = deg2rad.(interp_column(df.Longitude_deg).(sample_points))
    latitude_guess = deg2rad.(interp_column(df.Latitude_deg).(sample_points))
    velocity_guess = (interp_column(df.Velocity_1000mps).(sample_points) .* 1.0e3) ./ V_SCALE
    flight_path_guess = deg2rad.(interp_column(df.FlightPath_deg).(sample_points))
    azimuth_guess = deg2rad.(interp_column(df.Azimuth_deg).(sample_points))
    alpha_guess = deg2rad.(interp_column(df.AngleOfAttack_deg).(sample_points))
    beta_guess = deg2rad.(interp_column(df.BankAngle_deg).(sample_points))
    dt_guess = clamp.(interp_column(dt_source).(sample_points), 0.1, 1.0)

    return hcat(
        altitude_guess,
        longitude_guess,
        latitude_guess,
        velocity_guess,
        flight_path_guess,
        azimuth_guess,
        alpha_guess,
        beta_guess,
        dt_guess,
    )
end
# Uncomment the lines below to pass user options to the solver
user_options = (
    "tol" => 1e-9,
    "acceptable_tol" => 1e-8,
    "constr_viol_tol" => 1e-9,
    "compl_inf_tol" => 1e-9,
    "dual_inf_tol" => 1e-9,
    "bound_relax_factor" => 0.0,
    "honor_original_bounds" => "yes",
    "max_iter" => 5000,
)

# Create JuMP model, using Ipopt as the solver
model = Model(optimizer_with_attributes(Ipopt.Optimizer, user_options...))
@operator(model, earth_density_op, 1, earth_atmosphere_density)

@variables(model, begin
    0 ≤ scaled_h[1:n]                # altitude (m) / 1e5
    ϕ[1:n]                # longitude (rad)
    deg2rad(-89) ≤ θ[1:n] ≤ deg2rad(89)  # latitude (rad)
    1e-4 ≤ scaled_v[1:n]                # velocity (m/sec) / 1e4
    deg2rad(-89) ≤ γ[1:n] ≤ deg2rad(89)  # flight path angle (rad)
    ψ[1:n]                # azimuth (rad)
    deg2rad(-90) ≤ α[1:n] ≤ deg2rad(90)  # angle of attack (rad)
    deg2rad(-89) ≤ β[1:n] ≤ deg2rad(89)  # bank angle (rad)
    0.1 ≤       Δt[1:n] ≤ 1.0          # time step (sec)
    # 0.0 <= q_dot[1:n] <= 269.0               # heat rate (W/m^2)
    # 0.0 <= q[1:n] <= 6200.0                  # heat load (J/m^2)
    # Δt[1:n] == 4.0         # time step (sec)
end);

# Fix initial conditions
fix(scaled_h[1], h_s; force = true)
fix(ϕ[1], ϕ_s; force = true)
fix(θ[1], θ_s; force = true)
fix(scaled_v[1], v_s; force = true)
fix(γ[1], γ_s; force = true)
fix(ψ[1], ψ_s; force = true)
fix(α[1], α_s; force = true)
fix(β[1], β_s; force = true)
# fix(q_dot[1], 0.0; force = true)
# fix(q[1], 0.0; force = true)

# Hard terminal conditions: pin the final state to the desired target.
fix(scaled_h[n], h_t; force = true)
fix(ϕ[n], ϕ_t; force = true)
fix(θ[n], θ_t; force = true)
fix(scaled_v[n], v_t; force = true)
fix(γ[n], γ_t; force = true)

# Initial guess: linear interpolation between boundary conditions
x_s = [h_s, ϕ_s, θ_s, v_s, γ_s, ψ_s, α_s, β_s, t_s]  # Initial state and control at the first knot
x_t = [h_t, ϕ_t, θ_t, v_t, γ_t, ψ_s, α_s, β_s, t_s]
interp_linear = Interpolations.LinearInterpolation([1, n], [x_s, x_t])
linear_initial_guess = mapreduce(transpose, vcat, interp_linear.(1:n))
csv_initial_guess = load_initial_guess_from_csv(INITIAL_GUESS_PATH, n)
initial_guess = something(csv_initial_guess, linear_initial_guess)
set_start_value.(scaled_h, initial_guess[:, 1])
set_start_value.(ϕ, initial_guess[:, 2])
set_start_value.(θ, initial_guess[:, 3])
set_start_value.(scaled_v, initial_guess[:, 4])
set_start_value.(γ, initial_guess[:, 5])
set_start_value.(ψ, initial_guess[:, 6])
set_start_value.(α, initial_guess[:, 7])
set_start_value.(β, initial_guess[:, 8])
set_start_value.(Δt, initial_guess[:, 9])
fix(Δt[n], initial_guess[end, 9]; force = true)

# Functions to restore `h` and `v` to their true scale
@expression(model, h[j=1:n], scaled_h[j] * H_SCALE)
@expression(model, v[j=1:n], scaled_v[j] * V_SCALE)

# Helper functions
@expression(model, cL[j=1:n], a₀ + a₁ * rad2deg(α[j]))
@expression(model, cD[j=1:n], b₀ + b₁ * rad2deg(α[j]) + b₂ * rad2deg(α[j])^2)
@expression(model, ρ[j=1:n], earth_density_op(h[j]))
@expression(model, D[j=1:n], 0.5 * cD[j] * S * ρ[j] * v[j]^2)
@expression(model, L[j=1:n], 0.5 * cL[j] * S * ρ[j] * v[j]^2)
@expression(model, r[j=1:n], Rₑ + h[j])
@expression(model, g[j=1:n], μ / r[j]^2)
@expression(model, q_dot[j=1:n], C1 * ρ[j]^n_exp * v[j]^m_exp)

# Motion of the vehicle as a differential-algebraic system of equations (DAEs)
@expression(model, δh[j=1:n], v[j] * sin(γ[j]))
@expression(model, δϕ[j=1:n], (v[j] / r[j]) * cos(γ[j]) * sin(ψ[j]) / cos(θ[j]))
@expression(model, δθ[j=1:n], (v[j] / r[j]) * cos(γ[j]) * cos(ψ[j]))
@expression(model, δv[j=1:n], -(D[j] / m) - g[j] * sin(γ[j]))
@expression(
    model,
    δγ[j=1:n],
    (L[j] / (m * v[j])) * cos(β[j]) +
    cos(γ[j]) * ((v[j] / r[j]) - (g[j] / v[j]))
)
@expression(
    model,
    δψ[j=1:n],
    (1 / (m * v[j] * cos(γ[j]))) * L[j] * sin(β[j]) +
    (v[j] / (r[j] * cos(θ[j]))) * cos(γ[j]) * sin(ψ[j]) * sin(θ[j])
)

function reentry_rhs_expressions(model, hq, ϕq, θq, vq, γq, ψq, αq, βq)
    cLq = @expression(model, a₀ + a₁ * rad2deg(αq))
    cDq = @expression(model, b₀ + b₁ * rad2deg(αq) + b₂ * rad2deg(αq)^2)
    ρq = @expression(model, earth_density_op(hq))
    Dq = @expression(model, 0.5 * cDq * S * ρq * vq^2)
    Lq = @expression(model, 0.5 * cLq * S * ρq * vq^2)
    rq = @expression(model, Rₑ + hq)
    gq = @expression(model, μ / rq^2)

    dh = @expression(model, vq * sin(γq))
    dϕ = @expression(model, (vq / rq) * cos(γq) * sin(ψq) / cos(θq))
    dθ = @expression(model, (vq / rq) * cos(γq) * cos(ψq))
    dv = @expression(model, -(Dq / m) - gq * sin(γq))
    dγ = @expression(
        model,
        (Lq / (m * vq)) * cos(βq) +
        cos(γq) * ((vq / rq) - (gq / vq))
    )
    dψ = @expression(
        model,
        (1 / (m * vq * cos(γq))) * Lq * sin(βq) +
        (vq / (rq * cos(θq))) * cos(γq) * sin(ψq) * sin(θq)
    )

    return dh, dϕ, dθ, dv, dγ, dψ
end

# @constraint(model, q_dot .<= 269.0)  # Heat rate constraint
@constraint(model, [j=2:n], α[j] - α[j - 1] <= _CONTROL_RATE_LIMIT_RAD_PER_SEC[1] * Δt[j - 1])
@constraint(model, [j=2:n], α[j - 1] - α[j] <= _CONTROL_RATE_LIMIT_RAD_PER_SEC[1] * Δt[j - 1])
@constraint(model, [j=2:n], β[j] - β[j - 1] <= _CONTROL_RATE_LIMIT_RAD_PER_SEC[2] * Δt[j - 1])
@constraint(model, [j=2:n], β[j - 1] - β[j] <= _CONTROL_RATE_LIMIT_RAD_PER_SEC[2] * Δt[j - 1])

# Dynamics constraints
for j in 2:n
    i = j - 1  # index of previous knot

    if integration_rule == "rectangular"
        # Rectangular integration
        @constraint(model, h[j] == h[i] + Δt[i] * δh[i])
        @constraint(model, ϕ[j] == ϕ[i] + Δt[i] * δϕ[i])
        @constraint(model, θ[j] == θ[i] + Δt[i] * δθ[i])
        @constraint(model, v[j] == v[i] + Δt[i] * δv[i])
        @constraint(model, γ[j] == γ[i] + Δt[i] * δγ[i])
        @constraint(model, ψ[j] == ψ[i] + Δt[i] * δψ[i])
    elseif integration_rule == "trapezoidal"
        # Trapezoidal integration
        @constraint(model, h[j] == h[i] + 0.5 * Δt[i] * (δh[j] + δh[i]))
        @constraint(model, ϕ[j] == ϕ[i] + 0.5 * Δt[i] * (δϕ[j] + δϕ[i]))
        @constraint(model, θ[j] == θ[i] + 0.5 * Δt[i] * (δθ[j] + δθ[i]))
        @constraint(model, v[j] == v[i] + 0.5 * Δt[i] * (δv[j] + δv[i]))
        @constraint(model, γ[j] == γ[i] + 0.5 * Δt[i] * (δγ[j] + δγ[i]))
        @constraint(model, ψ[j] == ψ[i] + 0.5 * Δt[i] * (δψ[j] + δψ[i]))
        
    elseif integration_rule == "rk4"
        # RK4 with linearly interpolated controls over the interval [i, j].
        α_mid = @expression(model, 0.5 * (α[i] + α[j]))
        β_mid = @expression(model, 0.5 * (β[i] + β[j]))

        k1_h, k1_ϕ, k1_θ, k1_v, k1_γ, k1_ψ =
            reentry_rhs_expressions(model, h[i], ϕ[i], θ[i], v[i], γ[i], ψ[i], α[i], β[i])

        h2 = @expression(model, h[i] + 0.5 * Δt[i] * k1_h)
        ϕ2 = @expression(model, ϕ[i] + 0.5 * Δt[i] * k1_ϕ)
        θ2 = @expression(model, θ[i] + 0.5 * Δt[i] * k1_θ)
        v2 = @expression(model, v[i] + 0.5 * Δt[i] * k1_v)
        γ2 = @expression(model, γ[i] + 0.5 * Δt[i] * k1_γ)
        ψ2 = @expression(model, ψ[i] + 0.5 * Δt[i] * k1_ψ)
        k2_h, k2_ϕ, k2_θ, k2_v, k2_γ, k2_ψ =
            reentry_rhs_expressions(model, h2, ϕ2, θ2, v2, γ2, ψ2, α_mid, β_mid)

        h3 = @expression(model, h[i] + 0.5 * Δt[i] * k2_h)
        ϕ3 = @expression(model, ϕ[i] + 0.5 * Δt[i] * k2_ϕ)
        θ3 = @expression(model, θ[i] + 0.5 * Δt[i] * k2_θ)
        v3 = @expression(model, v[i] + 0.5 * Δt[i] * k2_v)
        γ3 = @expression(model, γ[i] + 0.5 * Δt[i] * k2_γ)
        ψ3 = @expression(model, ψ[i] + 0.5 * Δt[i] * k2_ψ)
        k3_h, k3_ϕ, k3_θ, k3_v, k3_γ, k3_ψ =
            reentry_rhs_expressions(model, h3, ϕ3, θ3, v3, γ3, ψ3, α_mid, β_mid)

        h4 = @expression(model, h[i] + Δt[i] * k3_h)
        ϕ4 = @expression(model, ϕ[i] + Δt[i] * k3_ϕ)
        θ4 = @expression(model, θ[i] + Δt[i] * k3_θ)
        v4 = @expression(model, v[i] + Δt[i] * k3_v)
        γ4 = @expression(model, γ[i] + Δt[i] * k3_γ)
        ψ4 = @expression(model, ψ[i] + Δt[i] * k3_ψ)
        k4_h, k4_ϕ, k4_θ, k4_v, k4_γ, k4_ψ =
            reentry_rhs_expressions(model, h4, ϕ4, θ4, v4, γ4, ψ4, α[j], β[j])

        @constraint(model, h[j] == h[i] + (Δt[i] / 6) * (k1_h + 2 * k2_h + 2 * k3_h + k4_h))
        @constraint(model, ϕ[j] == ϕ[i] + (Δt[i] / 6) * (k1_ϕ + 2 * k2_ϕ + 2 * k3_ϕ + k4_ϕ))
        @constraint(model, θ[j] == θ[i] + (Δt[i] / 6) * (k1_θ + 2 * k2_θ + 2 * k3_θ + k4_θ))
        @constraint(model, v[j] == v[i] + (Δt[i] / 6) * (k1_v + 2 * k2_v + 2 * k3_v + k4_v))
        @constraint(model, γ[j] == γ[i] + (Δt[i] / 6) * (k1_γ + 2 * k2_γ + 2 * k3_γ + k4_γ))
        @constraint(model, ψ[j] == ψ[i] + (Δt[i] / 6) * (k1_ψ + 2 * k2_ψ + 2 * k3_ψ + k4_ψ))
    else
        @error "Unexpected integration rule '$(integration_rule)'"
    end
    # @constraint(model, q[j] == q[i] + q_dot[i]*Δt[i])
end

# Heating constraints
@constraint(model, scaled_h[1:(n-1)] .>= h_t)  # Hard constraint on altitude to ensure we don't end up underground
@expression(model, total_time, sum(Δt[j] for j in 1:(n - 1)))
@expression(model, control_smoothing, sum((α[j] - α[j - 1])^2 + (β[j] - β[j - 1])^2 for j in 2:n))
@expression(model, control_effort, sum(α[j]^2 + β[j]^2 for j in 1:n))
@objective(
    model,
    Min,
    1.0e3 * total_time +
    1.0e5 * control_smoothing +
    1.0e3 * control_effort,
)

# set_silent(model)  # Hide solver's verbose output
optimize!(model)  # Solve for the control and state
term_status = termination_status(model)
primal_stat = primal_status(model)
acceptable_term_statuses = (
    MOI.OPTIMAL,
    MOI.LOCALLY_SOLVED,
    MOI.ALMOST_OPTIMAL,
    MOI.ALMOST_LOCALLY_SOLVED,
)
acceptable_primal_statuses = (
    MOI.FEASIBLE_POINT,
    MOI.NEARLY_FEASIBLE_POINT,
)
if !(term_status in acceptable_term_statuses && primal_stat in acceptable_primal_statuses)
    error(
        "Reference trajectory solve failed. Here is the output of `solution_summary` to help debug why this happened:\n\n" *
        sprint(show, solution_summary(model; verbose = false)),
    )
end

# Show final cross-range of the solution
println(
    "Final latitude θ = ",
    round(value(θ[n]) |> rad2deg; digits = 2),
    "°",
    " at longitude ϕ = ",
    round(value(ϕ[n]) |> rad2deg; digits = 2),
    "°",
    " and altitude h = ",
    round(value(h[n]); digits = 2),
    " m",
    " and velocity v = ",
    round(value(v[n]); digits = 2),
    " m/sec",
    " and flight path angle γ = ",
    round(value(γ[n]) |> rad2deg; digits = 2),
    "°",
)
println(
    "Terminal errors: Δh = ",
    value(h[n]) - h_t * H_SCALE,
    " m, Δlon = ",
    rad2deg(value(ϕ[n]) - ϕ_t),
    " deg, Δlat = ",
    rad2deg(value(θ[n]) - θ_t),
    " deg, Δv = ",
    value(v[n]) - v_t * V_SCALE,
    " m/sec, Δγ = ",
    rad2deg(value(γ[n]) - γ_t),
    " deg",
)

using Plots
ts = cumsum([0; value.(Δt)])[1:(end-1)]
plt_altitude = plot(
    ts,
    value.(h);
    legend = nothing,
    title = "Altitude (m)",
)
plt_longitude =
    plot(ts, rad2deg.(value.(ϕ)); legend = nothing, title = "Longitude (deg)")
plt_latitude =
    plot(ts, rad2deg.(value.(θ)); legend = nothing, title = "Latitude (deg)")
plt_velocity = plot(
    ts,
    value.(v);
    legend = nothing,
    title = "Velocity (m/sec)",
)
plt_flight_path =
    plot(ts, rad2deg.(value.(γ)); legend = nothing, title = "Flight Path (deg)")
plt_azimuth =
    plot(ts, rad2deg.(value.(ψ)); legend = nothing, title = "Azimuth (deg)")

maybe_display(plot(
    plt_altitude,
    plt_velocity,
    plt_longitude,
    plt_flight_path,
    plt_latitude,
    plt_azimuth;
    layout = grid(3, 2),
    linewidth = 2,
    size = (700, 700),
))

q_dots = earth_atmosphere_density.(value.(h)).^n_exp .* (value.(v)).^m_exp .* C1
maybe_display(plot(
    ts,
    q_dots;
    legend = nothing,
    title = "Heating Rate (W/m^2)",
    linewidth = 2,
    size = (700, 500),
))
# println(polyfit)
# function q_dot_calc(h, v)
#     q = C1 * earth_atmosphere_density(h)^n * v^m_exp
#     return q
# end

plt_attack_angle = plot(
    ts[1:(end-1)],
    rad2deg.(value.(α)[1:(end-1)]);
    legend = nothing,
    title = "Angle of Attack (deg)",
)
plt_bank_angle = plot(
    ts[1:(end-1)],
    rad2deg.(value.(β)[1:(end-1)]);
    legend = nothing,
    title = "Bank Angle (deg)",
)
plt_heat_rate = plot(
    ts,
    q_dots;
    legend = nothing,
    title = "Heating (W/m^2)",
)

# plt_heat_load = plot(
#     ts,
#     value.(q);
#     legend = nothing,
#     title = "Heat Load (J/m^2)",
# )
maybe_display(plot(
    plt_attack_angle,
    plt_bank_angle,
    # plt_heat_rate,
    # plt_heat_load;
    layout = grid(2, 1),
    linewidth = 2,
    size = (700, 700),
))

maybe_display(plot(
    rad2deg.(value.(ϕ)),
    rad2deg.(value.(θ)),
    value.(scaled_h);
    linewidth = 2,
    legend = nothing,
    title = "Earth EDL Reference Trajectory",
    xlabel = "Longitude (deg)",
    ylabel = "Latitude (deg)",
    zlabel = "Altitude (100 km)",
))

# Save optimal trajectory data to a CSV file

df = DataFrame(
    Time_s = ts,
    Altitude_100km = value.(scaled_h),
    Longitude_deg = rad2deg.(value.(ϕ)),
    Latitude_deg = rad2deg.(value.(θ)),
    Velocity_1000mps = value.(v) ./ 1e3,
    FlightPath_deg = rad2deg.(value.(γ)),
    Azimuth_deg = rad2deg.(value.(ψ)),
    AngleOfAttack_deg = rad2deg.(value.(α)),
    BankAngle_deg = rad2deg.(value.(β)),
    # HeatRate_Wm2 = value.(q_dots),
    # HeatLoad_Jm2 = value.(q),
)
CSV.write("optimal_trajectory.csv", df)
