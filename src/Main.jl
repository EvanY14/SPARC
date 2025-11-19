using Revise
includet("model/SimulatorModel.jl")

using .SimulatorModel
using StaticArrays
using Plots
using DifferentialEquations
using ProgressMeter
gr()
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
β = (u, p, t) -> deg2rad(0.0) # Constant bank angle in radians
# β = (u, p, t) -> deg2rad(45*sin(t*pi/250)) # Bank angle function in radians
# β = () -> deg2rad(rand() * 90.0 - 45.0) # Bank angle in radians
target_altitude = 11848.0 # Termination altitude in meters

polyfit_coeffs = Float64[2.484093267854419e-35, -3.432059129183589e-32, 2.0998712380197567e-29, -7.374629031680772e-27, 1.5792723271745155e-24, -1.8603802534535614e-22, 1.1824450144926489e-21, 3.944724626716538e-18, -8.193458848294376e-16, 9.735891059182661e-14, -7.897816207129188e-12, 4.5807414555856416e-10, -1.9161056559474318e-08, 5.713547101023083e-07, -1.1780507222866087e-05, 0.00015839694888627217, -0.0012270664089332438, 0.0035825645308133545, 0.012231321466518718, -0.1691661107577747, -4.32384932627002]
# polyfit_atmosphere = PolyfitAtmosphere{length(polyfit_coeffs)}(SVector{length(polyfit_coeffs), Float64}(polyfit_coeffs))
# exponential_atmosphere = ExponentialAtmosphere(0.02, 11.1) # surface density in kg/m^3, scale height in km
gram_atmosphere = GramAtmosphere("GRAMpy/", "GRAM_Data", false, "mars", DateTime(2024, 1, 1, 0, 0, 0.0))
edl_cache = EDLCache()
edl_params = EDLParams(mass, drag_coefficient, lift_coefficient, area, μ, R, β, 0.0, target_altitude, integrator -> atmospheric_density(integrator, gram_atmosphere, false), 0.0, SVector{3, Float64}(zeros(3)), edl_cache)

callbacks = CallbackSet(altitude_termination_condition, atmospheric_density_callback, saving_callback)
# Define the ODE problem
prob = ODEProblem(edl_dynamics, u0, tspan, edl_params, callback=callbacks)
# Solve the ODE problem
sol = solve(prob, Tsit5(), reltol=1e-10, abstol=1e-12, dtmax=1.0)
# The solution `sol` now contains the state of the system over time
display(plot(sol.t, getindex.(sol.u, 1) ./ 1e3 , xlabel="Time (s)", ylabel="Altitude (km)", title="EDL Simulation: Altitude vs Time", legend=false))
display(plot(sol.t, getindex.(sol.u, 4) ./ 1e3 , xlabel="Time (s)", ylabel="Velocity (km/s)", title="EDL Simulation: Velocity vs Time", legend=false))
display(plot(getindex.(sol.u, 2) .* (180 / π), getindex.(sol.u, 3) .* (180 / π), xlabel="Longitude (deg)", ylabel="Latitude (deg)", title="EDL Simulation: Ground Track", legend=false))

# Get saved data
saved_data = saved_values.saveval
densities = zeros(length(saved_data))
betas = zeros(length(saved_data))
for i in 1:length(saved_data)
    densities[i] = saved_data[i][1]
    betas[i] = saved_data[i][2]
end
println(size(betas))
println(size(saved_values.t))
# display(plot(saved_values.t, densities, xlabel="Time (s)", ylabel="Atmospheric Density (kg/m³)", title="Atmospheric Density Profile", legend=false, yscale=:log10))
display(plot(saved_values.t, betas .* (180 / π), xlabel="Time (s)", ylabel="Bank Angle (deg)", title="Bank Angle Profile", legend=false))
# Monte Carlo to test atmospheric disturbances
num_simulations = 1000
final_latitudes = zeros(num_simulations)
final_longitudes = zeros(num_simulations)
altitude_profiles = Vector{Vector{Float64}}(undef, num_simulations)
velocity_profiles = Vector{Vector{Float64}}(undef, num_simulations)
times = Vector{Vector{Float64}}(undef, num_simulations)
gram_atmosphere = GramAtmosphere("GRAMpy/", "GRAM_Data", false, "mars", DateTime(2024, 1, 1, 0, 0, 0.0))
edl_params.atmospheric_density_function = integrator -> atmospheric_density(integrator, gram_atmosphere, true)
@showprogress for sim in 1:num_simulations
    prob_mc = ODEProblem(edl_dynamics, u0, tspan, edl_params, callback=callbacks)
    sol_mc = solve(prob_mc, Tsit5(), reltol=1e-10, abstol=1e-12, dtmax=1.0)
    altitude_profiles[sim] = getindex.(sol_mc.u, 1) ./ 1e3
    velocity_profiles[sim] = getindex.(sol_mc.u, 4) ./ 1e3
    times[sim] = sol_mc.t
    final_latitudes[sim] = getindex(sol_mc.u[end], 3) * (180 / π)
    final_longitudes[sim] = getindex(sol_mc.u[end], 2) * (180 / π)
end

# Plot MC results
# Altitude profiles
plt1 = plot(title="Monte Carlo Simulations: Altitude Profiles", xlabel="Time (s)", ylabel="Altitude (km)", legend=true)
for i in 1:num_simulations
    plot!(plt1, times[i], altitude_profiles[i], color=:grey, alpha=0.3, label=false)
end
# Plot nominal trajectory
plot!(plt1, sol.t, getindex.(sol.u, 1) ./ 1e3, color=:red, label="Nominal Trajectory")
display(plt1)
savefig(plt1, "monte_carlo_altitude_profiles.pdf")

# Velocity profiles
plt2 = plot(title="Monte Carlo Simulations: Velocity Profiles", xlabel="Time (s)", ylabel="Velocity (km/s)", legend=true)
for i in 1:num_simulations
    plot!(plt2, times[i], velocity_profiles[i], color=:grey, alpha=0.3, label=false)
end
# Plot nominal trajectory
plot!(plt2, sol.t, getindex.(sol.u, 4) ./ 1e3, color=:red, label="Nominal Trajectory")
display(plt2)
savefig(plt2, "monte_carlo_velocity_profiles.pdf")

# Final landing locations
plt3 = plot(final_longitudes, final_latitudes, seriestype=:scatter, xlabel="Longitude (deg)", ylabel="Latitude (deg)", title="Monte Carlo Simulations: Final Landing Locations", legend=true, label="Landing Locations", alpha=0.6)
plot!(plt3, [getindex(sol.u[end], 2) * (180 / π)], [getindex(sol.u[end], 3) * (180 / π)], seriestype=:scatter, color=:red, markershape=:star5, markersize=8, label="Nominal Landing Location")
display(plt3)
savefig(plt3, "monte_carlo_final_landing_locations.pdf")