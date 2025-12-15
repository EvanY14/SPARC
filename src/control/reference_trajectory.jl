using JuMP
import Interpolations
import Ipopt
using CSV
using DataFrames
# Global variables
const w = 203000.0  # weight (lb)
const g₀ = 32.174    # acceleration (ft/sec^2)
const m = 3257.0    # mass (slug)

# Aerodynamic and atmospheric forces on the vehicle
const ρ₀ = 7e-3 # Density at surface on Mars (kg/m^3)
const hᵣ = 11.1e3 # Scale height (m)
const Rₑ = 3396.2e3 # Radius of Mars (m)
const μ = 4.2828372e13 # Gravitational parameter (m^3/sec^2)
const S = 15.904 # Reference area (m^2)
# const cD = 1.46 # Drag coefficient
# const cL = 0.24 * cD # Lift coefficient

a₀ = -0.20704
a₁ = 0.029244
b₀ = 0.07854
b₁ = -0.61592e-2
b₂ = 0.621408e-3

const C1 = 8.53e-13 # Constant for convective heat rate calculation
const n_exp = 0.82958 # Exponent for convective heat rate calculation
const m_exp = 4.512 # Exponent for convective heat rate calculation

# Initial conditions
const H_SCALE = 1e5  # Altitude scaling factor
const V_SCALE = 1e4  # Velocity scaling factor
const h_s = 125.0e3 / H_SCALE          # altitude (m) / 1e5
const ϕ_s = deg2rad(126.7)   # longitude (rad)
const θ_s = deg2rad(-3.93)   # latitude (rad)
const v_s = 5845.39 / V_SCALE         # velocity (m/sec) / 1e4
const γ_s = deg2rad(-15.49)  # flight path angle (rad)
const ψ_s = deg2rad(90.0)  # azimuth (rad)
const α_s = deg2rad(0)   # angle of attack (rad)
const β_s = deg2rad(0)   # bank angle (rad)
const t_s = 0.33         # time step (sec)

# Final conditions, the so-called Terminal Area Energy Management (TAEM)
const h_t = 11848.0 / H_SCALE          # altitude (ft) / 1e5
const v_t = 700.0 / V_SCALE         # velocity (ft/sec) / 1e4
const γ_t = deg2rad(-5.0)  # flight path angle (rad)

# Number of mesh points (knots) to be used
const n = 503

# Integration scheme to be used for the dynamics
const integration_rule = "trapezoidal"  # "rectangular", "trapezoidal", or "rk4"
const polyfit_coefficients = [-8.278592174668491e-43, 1.2598495030132498e-38, -8.634065871212132e-35, 3.5185552646901455e-31, -9.480197229347404e-28, 1.7753104600795092e-24, -2.3622107295909874e-21, 2.2393603867716714e-18, -1.487031340144351e-15, 6.592111911218399e-13, -1.714014789283248e-10, 1.3556252797088945e-08, 5.196239221937857e-06, -0.0012393556758398866, -0.0500835105059738, -4.213431227716942]
polyfit_exponent = (h) -> polyfit_coefficients[1] * h^15 + polyfit_coefficients[2] * h^14 + polyfit_coefficients[3] * h^13 +
    polyfit_coefficients[4] * h^12 + polyfit_coefficients[5] * h^11 + polyfit_coefficients[6] * h^10 +
    polyfit_coefficients[7] * h^9 + polyfit_coefficients[8] * h^8 + polyfit_coefficients[9] * h^7 +
    polyfit_coefficients[10] * h^6 + polyfit_coefficients[11] * h^5 + polyfit_coefficients[12] * h^4 +
    polyfit_coefficients[13] * h^3 + polyfit_coefficients[14] * h^2 + polyfit_coefficients[15] * h^1 + polyfit_coefficients[16]
display(plot(
    exp.(polyfit_exponent.(1.0:1.0:100.0)),
    1.0:1.0:100.0,
    title = "Polyfit Exponent vs Altitude",
    ylabel = "Altitude (km)",
    xlabel = "Density",
    xscale = :log10,
    linewidth = 2,
    size = (700, 500),
))
# Uncomment the lines below to pass user options to the solver
user_options = (
# "mu_strategy" => "monotone",
# "linear_solver" => "ma27",
)

# Create JuMP model, using Ipopt as the solver
model = Model(optimizer_with_attributes(Ipopt.Optimizer, user_options...))

@variables(model, begin
    0 ≤ scaled_h[1:n]                # altitude (ft) / 1e5
    ϕ[1:n]                # longitude (rad)
    deg2rad(-89) ≤ θ[1:n] ≤ deg2rad(89)  # latitude (rad)
    1e-4 ≤ scaled_v[1:n]                # velocity (ft/sec) / 1e4
    deg2rad(-89) ≤ γ[1:n] ≤ deg2rad(89)  # flight path angle (rad)
    ψ[1:n]                # azimuth (rad)
    deg2rad(-90) ≤ α[1:n] ≤ deg2rad(90)  # angle of attack (rad)
    deg2rad(-89) ≤ β[1:n] ≤ deg2rad(89)  # bank angle (rad)
    0.1 ≤       Δt[1:n] ≤ 1.0          # time step (sec)
    # 0.0 <= q_dot[1:n] <= 269.0               # heat rate (W/m^2)
    0.0 <= q[1:n] <= 6200.0                  # heat load (J/m^2)
    # Δt[1:n] == 4.0         # time step (sec)
end);

# Fix initial conditions
fix(scaled_h[1], h_s; force = true)
fix(ϕ[1], ϕ_s; force = true)
fix(θ[1], θ_s; force = true)
fix(scaled_v[1], v_s; force = true)
fix(γ[1], γ_s; force = true)
fix(ψ[1], ψ_s; force = true)
# fix(q_dot[1], 0.0; force = true)
fix(q[1], 0.0; force = true)

# Fix final conditions
fix(scaled_h[n], h_t; force = true)
# fix(scaled_v[n], v_t; force = true)
# fix(γ[n], γ_t; force = true)
# fix(θ[n], deg2rad(-4.5); force = true)  # Target latitude in radians
# fix(ϕ[n], deg2rad(137.4); force = true)  # Target longitude in radians

# Initial guess: linear interpolation between boundary conditions
x_s = [h_s, ϕ_s, θ_s, v_s, γ_s, ψ_s, α_s, β_s, t_s, 0.0]
x_t = [h_t, ϕ_s, θ_s, v_t, γ_t, ψ_s, α_s, β_s, t_s, 6200.0]
interp_linear = Interpolations.LinearInterpolation([1, n], [x_s, x_t])
initial_guess = mapreduce(transpose, vcat, interp_linear.(1:n))
set_start_value.(all_variables(model), vec(initial_guess))

# Functions to restore `h` and `v` to their true scale
@expression(model, h[j=1:n], scaled_h[j] * H_SCALE)
@expression(model, v[j=1:n], scaled_v[j] * V_SCALE)

# Helper functions
@expression(model, cL[j=1:n], a₀ + a₁ * rad2deg(α[j]))
@expression(model, cD[j=1:n], b₀ + b₁ * rad2deg(α[j]) + b₂ * rad2deg(α[j])^2)
@expression(model, ρ[j=1:n], exp(polyfit_exponent(h[j]*1e-3)))  # Convert altitude to km
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

# System dynamics
if integration_rule == "rk4"
    # Precompute RK4 k-values for all knots
    @expression(model, k1_dh[j=1:n], δh[j])
    @expression(model, k1_dϕ[j=1:n], δϕ[j])
    @expression(model, k1_dθ[j=1:n], δθ[j])
    @expression(model, k1_dv[j=1:n], δv[j])
    @expression(model, k1_dγ[j=1:n], δγ[j])
    @expression(model, k1_dψ[j=1:n], δψ[j])

    @expression(
        model,
        k2_dh[j=1:n],
        δh[j] + 0.5 * Δt[j] * k1_dh[j]
    )
    @expression(
        model,
        k2_dϕ[j=1:n],
        δϕ[j] + 0.5 * Δt[j] * k1_dϕ[j]
    )
    @expression(
        model,
        k2_dθ[j=1:n],
        δθ[j] + 0.5 * Δt[j] * k1_dθ[j]
    )
    @expression(
        model,
        k2_dv[j=1:n],
        δv[j] + 0.5 * Δt[j] * k1_dv[j]
    )
    @expression(
        model,
        k2_dγ[j=1:n],
        δγ[j] + 0.5 * Δt[j] * k1_dγ[j]
    )
    @expression(
        model,
        k2_dψ[j=1:n],
        δψ[j] + 0.5 * Δt[j] * k1_dψ[j]
    )

    @expression(
        model,
        k3_dh[j=1:n],
        δh[j] + 0.5 * Δt[j] * k2_dh[j]
    )
    @expression(
        model,
        k3_dϕ[j=1:n],
        δϕ[j] + 0.5 * Δt[j] * k2_dϕ[j]
    )
    @expression(
        model,
        k3_dθ[j=1:n],
        δθ[j] + 0.5 * Δt[j] * k2_dθ[j]
    )
    @expression(
        model,
        k3_dv[j=1:n],
        δv[j] + 0.5 * Δt[j] * k2_dv[j]
    )
    @expression(
        model,
        k3_dγ[j=1:n],
        δγ[j] + 0.5 * Δt[j] * k2_dγ[j]
    )
    @expression(
        model,
        k3_dψ[j=1:n],
        δψ[j] + 0.5 * Δt[j] * k2_dψ[j]
    )
    @expression(
        model,
        k4_dh[j=1:n],
        δh[j] + Δt[j] * k3_dh[j]
    )
    @expression(
        model,
        k4_dϕ[j=1:n],
        δϕ[j] + Δt[j] * k3_dϕ[j]
    )
    @expression(
        model,
        k4_dθ[j=1:n],
        δθ[j] + Δt[j] * k3_dθ[j]
    )
    @expression(
        model,
        k4_dv[j=1:n],
        δv[j] + Δt[j] * k3_dv[j]
    )
    @expression(
        model,
        k4_dγ[j=1:n],
        δγ[j] + Δt[j] * k3_dγ[j]
    )
    @expression(
        model,
        k4_dψ[j=1:n],
        δψ[j] + Δt[j] * k3_dψ[j]
    )
end
@constraint(model, q_dot .<= 269.0)  # Heat rate constraint
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
        # Runge-Kutta 4th order integration from step i to j
        @constraint(model, h[j] == h[i] + (Δt[i] / 6) * (k1_dh[i] + 2 * k2_dh[i] + 2 * k3_dh[i] + k4_dh[i]))
        @constraint(model, ϕ[j] == ϕ[i] + (Δt[i] / 6) * (k1_dϕ[i] + 2 * k2_dϕ[i] + 2 * k3_dϕ[i] + k4_dϕ[i]))
        @constraint(model, θ[j] == θ[i] + (Δt[i] / 6) * (k1_dθ[i] + 2 * k2_dθ[i] + 2 * k3_dθ[i] + k4_dθ[i]))
        @constraint(model, v[j] == v[i] + (Δt[i] / 6) * (k1_dv[i] + 2 * k2_dv[i] + 2 * k3_dv[i] + k4_dv[i]))
        @constraint(model, γ[j] == γ[i] + (Δt[i] / 6) * (k1_dγ[i] + 2 * k2_dγ[i] + 2 * k3_dγ[i] + k4_dγ[i]))
        @constraint(model, ψ[j] == ψ[i] + (Δt[i] / 6) * (k1_dψ[i] + 2 * k2_dψ[i] + 2 * k3_dψ[i] + k4_dψ[i]))
    else
        @error "Unexpected integration rule '$(integration_rule)'"
    end
    @constraint(model, q[j] == q[i] + q_dot[i]*Δt[i])
end

# Heating constraints
# Objective: Reach target latitude and longitude
target_latitude = deg2rad(-4.5)  # Target latitude in radians
@constraint(model,target_latitude - deg2rad(0.1) <= θ[n] <= target_latitude + deg2rad(0.1))
@expression(model, latitude_error, θ[n] - target_latitude)
target_longitude = deg2rad(137.4)  # Target longitude in radians
@constraint(model,target_longitude - deg2rad(0.1) <= ϕ[n] <= target_longitude + deg2rad(0.1))
# @constraint(model, γ[n] >= deg2rad(-6.0))
@expression(model, longitude_error, ϕ[n] - target_longitude)
@expression(model, altitude_error, h[n] / H_SCALE - h_t)  # Target altitude in meters
@expression(model, velocity_error, v[n] / V_SCALE - v_t) # Target velocity in m/s
@expression(model, flight_path_angle_error, γ[n] - γ_t) # Target flight path angle in radians
@objective(model, Min, sum(Δt))

# set_silent(model)  # Hide solver's verbose output
set_attribute(model, "tol", 1e-6)  # Set solver tolerance
optimize!(model)  # Solve for the control and state
assert_is_solved_and_feasible(model)

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

display(plot(
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

q_dots = exp.(polyfit_exponent.(value.(h) * 1e-3)).^n_exp .* (value.(v)).^m_exp .* C1
display(plot(
    ts,
    exp.(polyfit_exponent.(value.(h) * 1e-3)).^n_exp;
    legend = nothing,
    title = "Heating Rate (W/m^2)",
    linewidth = 2,
    size = (700, 500),
))
# println(polyfit)
# function q_dot_calc(h, v)
#     ρ(h) = exp.(polyfit_exponent.(h*1e-3))
#     q = C1 * ρ(h)^n * v^m_exp
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

plt_heat_load = plot(
    ts,
    value.(q);
    legend = nothing,
    title = "Heat Load (J/m^2)",
)
display(plot(
    plt_attack_angle,
    plt_bank_angle,
    plt_heat_rate,
    plt_heat_load;
    layout = grid(4, 1),
    linewidth = 2,
    size = (700, 700),
))

display(plot(
    rad2deg.(value.(ϕ)),
    rad2deg.(value.(θ)),
    value.(scaled_h);
    linewidth = 2,
    legend = nothing,
    title = "Space Shuttle Reentry Trajectory",
    xlabel = "Longitude (deg)",
    ylabel = "Latitude (deg)",
    zlabel = "Altitude (100,000 ft)",
))

# Save optimal trajectory data to a CSV file

df = DataFrame(
    Time_s = ts,
    Altitude_100km = value.(scaled_h),
    Longitude_deg = rad2deg.(value.(ϕ)),
    Latitude_deg = rad2deg.(value.(θ)),
    Velocity_1000mps = value.(scaled_v),
    FlightPath_deg = rad2deg.(value.(γ)),
    Azimuth_deg = rad2deg.(value.(ψ)),
    AngleOfAttack_deg = rad2deg.(value.(α)),
    BankAngle_deg = rad2deg.(value.(β)),
    HeatRate_Wm2 = value.(q_dots),
    HeatLoad_Jm2 = value.(q),
)
CSV.write("optimal_trajectory.csv", df)