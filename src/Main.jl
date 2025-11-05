using Revise
includet("model/SimulatorModel.jl")

using .SimulatorModel
using StaticArrays
using Plots
using DifferentialEquations
plotly()
# Define initial conditions and parameters
h0 = 100000.0       # Initial altitude in meters
ϕ0 = 0.0            # Initial latitude in radians
θ0 = 0.0            # Initial longitude in radians
v0 = 7800.0         # Initial velocity in m/s
γ0 = -5.0 * (π / 180) # Initial flight path angle in radians
ψ0 = 90.0 * (π / 180) # Initial azimuth angle in radians

u0 = @MVector [h0, ϕ0, θ0, v0, γ0, ψ0]
tspan = (0.0, 500.0) # Time span for the simulation
# Define EDL parameters
mass = 2000.0 # kg
drag_coefficient = 2.2
lift_coefficient = 0.3
area = 15.0 # m^2
μ = 4.2828372e13 # m^3/s^2 for Mars
R = 3389500.0 # m for Mars
β = 0.0 # Bank angle in radians

polyfit_coeffs = [-3.691310097181554e-58, 5.819173546214448e-54, -3.9285937578286423e-50, 1.4222601230188116e-46, -2.606951392190571e-43, 3.2943551967480965e-41, 9.394166176413728e-37, -1.7651753457891617e-33, -5.79069281873952e-31, 8.639557954110502e-27, -1.991207114225621e-23, 2.7207390647640917e-20, -2.5611296697872007e-17, 1.7386922029136165e-14, -8.619727907575625e-12, 3.1040218147963276e-09, -7.949080301839893e-07, 0.00013834108975291533, -0.014729001168514675, 0.6707044510751348, -19.414578139119545]
polyfit_atmosphere = PolyfitAtmosphere{length(polyfit_coeffs)}(SVector{length(polyfit_coeffs), Float64}(polyfit_coeffs))
edl_params = EDLParams(mass, drag_coefficient, lift_coefficient, area, μ, R, β, h -> atmospheric_density(h, polyfit_atmosphere))
# Define the ODE problem
prob = ODEProblem(edl_dynamics, u0, tspan, edl_params)
# Solve the ODE problem
sol = solve(prob, Tsit5(), reltol=1e-9, abstol=1e-11, dtmax=0.01)
# The solution `sol` now contains the state of the system over time
display(plot(sol.t, getindex.(sol.u, 1) ./ 1000, xlabel="Time (s)", ylabel="Altitude (km)", title="EDL Simulation: Altitude vs Time", legend=false))
display(plot(sol.t, getindex.(sol.u, 4) ./ 1000, xlabel="Time (s)", ylabel="Velocity (km/s)", title="EDL Simulation: Velocity vs Time", legend=false))
display(plot(getindex.(sol.u, 2) .* (180 / π), getindex.(sol.u, 3) .* (180 / π), xlabel="Latitude (deg)", ylabel="Longitude (deg)", title="EDL Simulation: Ground Track", legend=false))

