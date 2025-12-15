using CSV
using Plots
using DataFrames
plotly()

# Create plot instances
plt_alt = plot(title="Monte Carlo Simulations: Altitude Profiles", xlabel="Time (s)", ylabel="Altitude (km)", legend=true)
plt_vel = plot(title="Monte Carlo Simulations: Velocity Profiles", xlabel="Time (s)", ylabel="Velocity (km/s)", legend=true)
plt_landing_locations = plot(title="Monte Carlo Simulations: Landing Locations", xlabel="Longitude (deg)", ylabel="Latitude (deg)", legend=true)
plt_betas = plot(title="Monte Carlo Simulations: Bank Angle Profiles", xlabel="Time (s)", ylabel="Bank angle (deg)", legend=true)
plt_alphas = plot(title="Monte Carlo Simulations: Angle of Attack Profiles", xlabel="Time (s)", ylabel="Angle of attack (deg)", legend=true)
plt_heat_rates = plot(title="Monte Carlo Simulations: Heat Rate Profiles", xlabel="Time (s)", ylabel="Heat rate (W/m^2)", legend=true)
# Load data from csv files
data_directory = "MPC_results/"
lons = Float64[]
lats = Float64[]
i = 1
for file in readdir(data_directory)
    global i
    data = CSV.read(data_directory * file, DataFrame)
    times = data.Time_s
    if !contains(file, "control")
        alts = data.Altitude_100km
        vels = data.Velocity_1000mps
        append!(lons, data.Longitude_fin_deg[end])
        append!(lats, data.Latitude_fin_deg[end])
        i += 1
        plot!(plt_alt, times, alts, color=:grey, alpha=0.3, label=false)
        plot!(plt_vel, times, vels, color=:grey, alpha=0.3, label=false)
    else
        alphas = data.AngleOfAttack_deg
        betas = data.BankAngle_deg
        heat_rates = data.HeatRate_Wm2
        plot!(plt_alphas, times, alphas, color=:grey, alpha=0.3, label=false)
        plot!(plt_betas, times, betas, color=:grey, alpha=0.3, label=false)
        plot!(plt_heat_rates, times, heat_rates, color=:grey, alpha=0.3, label=false)
    end
end
plot!(plt_landing_locations, lons, lats, seriestype=:scatter, label=false)

# Plot state MC results
display(plt_alt)
display(plt_vel)
display(plt_landing_locations)

# Coplot control MC results
display(plot(plt_alphas, plt_betas, plt_heat_rates))
# display(plt_betas)
# display(plt_alphas)
# display(plt_heat_rates)