using CSV
using Plots
using DataFrames
using StatsPlots
using Distributions
gr()

# Create plot instances
plt_alt = plot(xlabel="Time (s)", ylabel="Altitude (km)", legend=true)
plt_vel = plot(xlabel="Time (s)", ylabel="Velocity (km/s)", legend=true)
plt_landing_locations = plot(xlabel="Longitude (deg)", ylabel="Latitude (deg)", legend=true)
plt_betas = plot(xlabel="Time (s)", ylabel="Bank angle (deg)", legend=true)
plt_alphas = plot(xlabel="Time (s)", ylabel="Angle of attack (deg)", legend=true)
plt_heat_rates = plot(xlabel="Time (s)", ylabel="Heat rate (W/m^2)", legend=true)
target_lon_lat = (137.4, -4.5)
# Load data from csv files
data_directory = "MPC_results/"
# data_directory = "ssimpc_data_57/ssimpc_data/"
optimal_trajectory_file = "optimal_trajectory.csv"
lons = Float64[]
lats = Float64[]
i = 1
total_alts = Vector{Vector{Float64}}()
total_vels = Vector{Vector{Float64}}()
total_lats = Vector{Vector{Float64}}()
total_lons = Vector{Vector{Float64}}()
total_alphas = Vector{Vector{Float64}}()
total_betas = Vector{Vector{Float64}}()
total_heat_rates = Vector{Vector{Float64}}()
max_timesteps = 0
for file in readdir(data_directory)
    global i
    global max_timesteps
    data = CSV.read(data_directory * file, DataFrame)
    times = data.Time_s
    if !contains(file, "control")
        alts = data.Altitude_100km
        vels = data.Velocity_1000mps
        max_timesteps = max(max_timesteps, length(times))
        append!(total_alts, [alts])
        append!(total_vels, [vels])
        append!(total_lats, [data.Latitude_fin_deg])
        append!(total_lons, [data.Longitude_fin_deg])
        append!(lons, data.Longitude_fin_deg[end])
        append!(lats, data.Latitude_fin_deg[end])
        i += 1
        plot!(plt_alt, times, alts, color=:grey, alpha=0.3, label=false)
        plot!(plt_vel, times, vels, color=:grey, alpha=0.3, label=false)
    else
        alphas = data.AngleOfAttack_deg
        betas = data.BankAngle_deg
        heat_rates = data.HeatRate_Wm2
        append!(total_alphas, [alphas])
        append!(total_betas, [betas])
        append!(total_heat_rates, [heat_rates])
        plot!(plt_alphas, times, alphas, color=:grey, alpha=0.3, label=false)
        plot!(plt_betas, times, betas, color=:grey, alpha=0.3, label=false)
        plot!(plt_heat_rates, times, heat_rates, color=:grey, alpha=0.3, label=false)
    end
end
plot!(plt_landing_locations, lons, lats, seriestype=:scatter, label=false)

# Calculate means of states and controls
println(max_timesteps)
mean_alts = zeros(max_timesteps)
step_data_counter = zeros(Int, max_timesteps)
for alt_vec in total_alts
    for t in 1:length(alt_vec)
        mean_alts[t] += alt_vec[t]
        step_data_counter[t] += 1
    end
end
for t in 1:max_timesteps
    if step_data_counter[t] > 0
        mean_alts[t] /= step_data_counter[t]
    end
end
plot!(plt_alt, cumsum(ones(max_timesteps)) * 0.5, mean_alts, color=:blue, linewidth=2, label="Mean Trajectory")
# Plot reference optimal trajectory
optimal_data = CSV.read(optimal_trajectory_file, DataFrame)
optimal_times = optimal_data.Time_s
optimal_alts = optimal_data.Altitude_100km * 1e2
optimal_vels = optimal_data.Velocity_1000mps * 10
optimal_lons = optimal_data.Longitude_deg
optimal_lats = optimal_data.Latitude_deg
optimal_alphas = optimal_data.AngleOfAttack_deg
optimal_betas = optimal_data.BankAngle_deg
optimal_heat_rates = optimal_data.HeatRate_Wm2
plot!(plt_alt, optimal_times, optimal_alts, color=:red, linewidth=2, label="Optimal Trajectory")
plot!(plt_vel, optimal_times, optimal_vels, color=:red, linewidth=2, label="Optimal Trajectory")
plot!(plt_alphas, optimal_times, optimal_alphas, color=:red, linewidth=2, label="Optimal Control")
plot!(plt_betas, optimal_times, optimal_betas, color=:red, linewidth=2, label="Optimal Control")
plot!(plt_heat_rates, optimal_times, optimal_heat_rates, color=:red, linewidth=2, label="Optimal Control")
# Plot state MC results
display(plt_alt)
display(plt_vel)
println(cov(lons, lats))
mean_lon = mean(lons)
mean_lat = mean(lats)
println("Mean Landing Location: Longitude = $mean_lon deg, Latitude = $mean_lat deg")
cov_XY = cov(lons, lats)
var_lon = var(lons)
var_lat = var(lats)
covellipse!(plt_landing_locations, [mean(lons), mean(lats)], [var_lon cov_XY; cov_XY var_lat], nstd=3, linecolor=:red, label="3σ Covariance Ellipse")
scatter!(plt_landing_locations, [target_lon_lat[1]], [target_lon_lat[2]], markershape=:star, markersize=10, markercolor=:red, label="Target Landing Site")
display(plt_landing_locations)

# display(StatsPlots.scatter(lons, lats, title="Monte Carlo Simulations: Landing Locations", xlabel="Longitude (deg)", ylabel="Latitude (deg)", legend=false, size=(800,600)))
# Coplot control MC results
# control_plot = plot(plt_alphas, plt_betas, plt_heat_rates, layout=(3,1), size=(1000,1600))
display(plt_alphas)
display(plt_betas)
display(plt_heat_rates)
display(control_plot)
# display(plt_betas)
# display(plt_alphas)
# display(plt_heat_rates)
savefig(plt_alt, "altitude_profiles.pdf")
savefig(plt_vel, "velocity_profiles.pdf")
savefig(plt_landing_locations, "landing_locations.pdf")
# savefig(control_plot, "control_profiles.pdf")
savefig(plt_betas, "bank_angle_profiles.pdf")
savefig(plt_alphas, "angle_of_attack_profiles.pdf")
savefig(plt_heat_rates, "heat_rate_profiles.pdf")
