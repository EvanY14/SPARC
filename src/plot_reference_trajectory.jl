using CSV
using DataFrames
using Plots

gr()

const REFERENCE_TRAJECTORY_PATH = get(ENV, "SPARC_REFERENCE_TRAJECTORY_FILE", "optimal_trajectory.csv")
const OUTPUT_DIR = get(ENV, "SPARC_REFERENCE_PLOTS_DIR", "reference_trajectory_output")
const DISPLAY_PLOTS = parse(Bool, get(ENV, "SPARC_REF_DISPLAY", "false"))
const PLOT_MARGIN = 8Plots.mm
const GUIDE_FONT = font(10)
const TICK_FONT = font(8)
const TITLE_FONT = font(11)
const TICKS_FONT_SMALL = font(8)
const IEEE_SINGLE_COLUMN_SIZE = (520, 760)

function base_plot_kwargs()
    return (
        legend = false,
        grid = true,
        foreground_color_grid = :lightgray,
        gridalpha = 0.45,
        gridlinewidth = 1.0,
        background_color = :white,
        background_color_inside = :white,
        background_color_outside = :white,
        guidefont = GUIDE_FONT,
        tickfont = TICK_FONT,
        titlefont = TITLE_FONT,
        left_margin = PLOT_MARGIN,
        right_margin = PLOT_MARGIN,
        bottom_margin = PLOT_MARGIN,
        top_margin = PLOT_MARGIN,
    )
end

function save_reference_plot(plt, basename)
    try
        pdf_path = joinpath(OUTPUT_DIR, basename * ".pdf")
        savefig(plt, pdf_path)
    catch err
        @warn "Unable to save PDF version of $basename" exception = (err, catch_backtrace())
    end

    if DISPLAY_PLOTS
        display(plt)
    end

    return nothing
end

mkpath(OUTPUT_DIR)

optimal_reference = CSV.read(REFERENCE_TRAJECTORY_PATH, DataFrame)

times_s = Float64.(optimal_reference.Time_s)
altitude_km = Float64.(optimal_reference.Altitude_100km) .* 100.0
longitude_deg = Float64.(optimal_reference.Longitude_deg)
latitude_deg = Float64.(optimal_reference.Latitude_deg)
velocity_mps = Float64.(optimal_reference.Velocity_1000mps) .* 1.0e3
flight_path_deg = Float64.(optimal_reference.FlightPath_deg)
azimuth_deg = Float64.(optimal_reference.Azimuth_deg)
angle_of_attack_deg = Float64.(optimal_reference.AngleOfAttack_deg)
bank_angle_deg = Float64.(optimal_reference.BankAngle_deg)

reference_trajectory_3d_plot = plot(
    longitude_deg,
    latitude_deg,
    altitude_km;
    seriestype = :path3d,
    xlabel = "Longitude (deg)",
    ylabel = "Latitude (deg)",
    zlabel = "Altitude (km)",
    legend = false,
    grid = true,
    foreground_color_grid = :lightgray,
    background_color = :white,
    background_color_inside = :white,
    background_color_outside = :white,
    guidefont = GUIDE_FONT,
    tickfont = TICK_FONT,
    titlefont = TITLE_FONT,
    left_margin = 12Plots.mm,
    right_margin = 12Plots.mm,
    bottom_margin = 12Plots.mm,
    top_margin = 12Plots.mm,
    linewidth = 4,
    size = (1000, 720),
)

altitude_plot = plot(
    times_s,
    altitude_km;
    xlabel = "Time (s)",
    ylabel = "Altitude (km)",
    base_plot_kwargs()...,
    linewidth = 2,
)

longitude_plot = plot(
    times_s,
    longitude_deg;
    xlabel = "Time (s)",
    ylabel = "Longitude (deg)",
    base_plot_kwargs()...,
    linewidth = 2,
)

latitude_plot = plot(
    times_s,
    latitude_deg;
    xlabel = "Time (s)",
    ylabel = "Latitude (deg)",
    base_plot_kwargs()...,
    linewidth = 2,
)

velocity_plot = plot(
    times_s,
    velocity_mps;
    xlabel = "Time (s)",
    ylabel = "Velocity (m/s)",
    base_plot_kwargs()...,
    linewidth = 2,
)

flight_path_plot = plot(
    times_s,
    flight_path_deg;
    xlabel = "Time (s)",
    ylabel = "Flight Path Angle (deg)",
    base_plot_kwargs()...,
    linewidth = 2,
)

azimuth_plot = plot(
    times_s,
    azimuth_deg;
    xlabel = "Time (s)",
    ylabel = "Azimuth (deg)",
    base_plot_kwargs()...,
    linewidth = 2,
)

reference_state_plot_part1 = plot(
    altitude_plot,
    longitude_plot,
    latitude_plot,
    layout = (3, 1),
    size = IEEE_SINGLE_COLUMN_SIZE,
    legend = false,
    margin = 10Plots.mm,
    left_margin = 16Plots.mm,
    right_margin = 10Plots.mm,
    bottom_margin = 14Plots.mm,
    top_margin = 12Plots.mm,
    tickfont = TICKS_FONT_SMALL,
    guidefont = GUIDE_FONT,
    titlefont = TITLE_FONT,
)

reference_state_plot_part2 = plot(
    velocity_plot,
    flight_path_plot,
    azimuth_plot;
    layout = (3, 1),
    size = IEEE_SINGLE_COLUMN_SIZE,
    legend = false,
    margin = 10Plots.mm,
    left_margin = 16Plots.mm,
    right_margin = 10Plots.mm,
    bottom_margin = 14Plots.mm,
    top_margin = 12Plots.mm,
    tickfont = TICKS_FONT_SMALL,
    guidefont = GUIDE_FONT,
    titlefont = TITLE_FONT,
)

angle_of_attack_plot = plot(
    times_s,
    angle_of_attack_deg;
    seriestype = :steppost,
    xlabel = "Time (s)",
    ylabel = "Angle of Attack (deg)",
    base_plot_kwargs()...,
    linewidth = 2,
)

bank_angle_plot = plot(
    times_s,
    bank_angle_deg;
    seriestype = :steppost,
    xlabel = "Time (s)",
    ylabel = "Bank Angle (deg)",
    base_plot_kwargs()...,
    linewidth = 2,
)

reference_control_plot = plot(
    angle_of_attack_plot,
    bank_angle_plot;
    layout = (2, 1),
    size = (1000, 720),
    legend = false,
    margin = 10Plots.mm,
    left_margin = 16Plots.mm,
    right_margin = 10Plots.mm,
    bottom_margin = 14Plots.mm,
    top_margin = 12Plots.mm,
    tickfont = TICKS_FONT_SMALL,
    guidefont = GUIDE_FONT,
    titlefont = TITLE_FONT,
)

save_reference_plot(reference_trajectory_3d_plot, "reference_trajectory_3d_state")
save_reference_plot(reference_state_plot_part1, "reference_trajectory_states_part1")
save_reference_plot(reference_state_plot_part2, "reference_trajectory_states_part2")
save_reference_plot(reference_control_plot, "reference_trajectory_controls")
