using Revise
includet("model/SimulatorModel.jl")

using .SimulatorModel
using StaticArrays
using Plots
using DifferentialEquations
plotly()
# Define initial conditions and parameters
h0 = 125000.0      # Initial altitude in meters
ϕ0 = deg2rad(126.8)            # Initial longitude in radians
θ0 = deg2rad(-3.9)            # Initial latitude in radians
v0 = 5845.39         # Initial velocity in m/s
γ0 = deg2rad(-15.49) # Initial flight path angle in radians
ψ0 = deg2rad(91.0) # Initial azimuth angle in radians

u0 = MVector{6, Float64}(h0, ϕ0, θ0, v0, γ0, ψ0)
tspan = (0.0, 500.0) # Time span for the simulation
# Define EDL parameters
mass = 3257.0 # kg
drag_coefficient = 1.46
lift_coefficient = 0.24*drag_coefficient
area = 15.904 # m^2
μ = 4.2828372e13 # m^3/s^2 for Mars
R = 3389500.0 # m for Mars
β = deg2rad(0.0) # Bank angle in radians
target_altitude = 11848.0 # Termination altitude in meters

polyfit_coeffs = Float64[2.484093267854419e-35, -3.432059129183589e-32, 2.0998712380197567e-29, -7.374629031680772e-27, 1.5792723271745155e-24, -1.8603802534535614e-22, 1.1824450144926489e-21, 3.944724626716538e-18, -8.193458848294376e-16, 9.735891059182661e-14, -7.897816207129188e-12, 4.5807414555856416e-10, -1.9161056559474318e-08, 5.713547101023083e-07, -1.1780507222866087e-05, 0.00015839694888627217, -0.0012270664089332438, 0.0035825645308133545, 0.012231321466518718, -0.1691661107577747, -4.32384932627002]
polyfit_atmosphere = PolyfitAtmosphere{length(polyfit_coeffs)}(SVector{length(polyfit_coeffs), Float64}(polyfit_coeffs))
# exponential_atmosphere = ExponentialAtmosphere(0.02, 11.1) # surface density in kg/m^3, scale height in km
gram_atmosphere = GramAtmosphere("Gram_Data/", false, "mars", DateTime(2024, 1, 1, 0, 0, 0.0))
edl_params = EDLParams(mass, drag_coefficient, lift_coefficient, area, μ, R, β, target_altitude, h -> atmospheric_density(h, gram_atmosphere))

# Define the ODE problem
prob = ODEProblem(edl_dynamics, u0, tspan, edl_params, callback=altitude_termination_condition)
# Solve the ODE problem
sol = solve(prob, Tsit5(), reltol=1e-10, abstol=1e-12, dtmax=0.1)
# The solution `sol` now contains the state of the system over time
display(plot(sol.t, getindex.(sol.u, 1) ./ 1e3 , xlabel="Time (s)", ylabel="Altitude (km)", title="EDL Simulation: Altitude vs Time", legend=false))
display(plot(sol.t, getindex.(sol.u, 4) ./ 1e3 , xlabel="Time (s)", ylabel="Velocity (km/s)", title="EDL Simulation: Velocity vs Time", legend=false))
display(plot(getindex.(sol.u, 2) .* (180 / π), getindex.(sol.u, 3) .* (180 / π), xlabel="Longitude (deg)", ylabel="Latitude (deg)", title="EDL Simulation: Ground Track", legend=false))
densities = zeros(length(sol.u))
for i in 1:length(sol.u)
    h = getindex(sol.u[i], 1)
    densities[i] = atmospheric_density(h, polyfit_atmosphere)
end
display(plot(getindex.(sol.u, 1) ./ 1e3, densities, xlabel="Altitude (km)", ylabel="Atmospheric Density (kg/m³)", title="Atmospheric Density Profile", legend=false, yscale=:log10))

