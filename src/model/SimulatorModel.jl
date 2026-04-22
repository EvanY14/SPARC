module SimulatorModel
    using StaticArrays
    using Reexport
    using DifferentialEquations
    using PythonCall
    using JuMP
    using Ipopt
    using Interpolations
    using Statistics

    export edl_dynamics 
    export atmospheric_density
    export altitude_termination_condition, atmospheric_density_callback, control_callback
    export saving_callback, saved_values
    export mpc, ssimpc, trackingmpc, trackingmpc_shrinking_horizon, trackingmpc_shrinking, model_predictive_guidance, mpg, sm_mpg, openloopcontrol, openloop
    export VehicleDefinition, VEHICLE, VEHICLE_MASS, VEHICLE_REFERENCE_AREA
    include("vehicle.jl")
    include("types.jl")
    @reexport using .ModelTypes
    include("earth_atmosphere_polyfit.jl")

    # Simulator models
    include("../simulation/simulator.jl")
    include("atmosphere_models.jl")

    # Control strategies
    include("../control/control_limits.jl")
    include("../control/regular_mpc.jl")
    include("../control/ssi_mpc.jl")
    include("../control/tracking_mpc.jl")
    include("../control/tracking_mpc_shrinking_horizon.jl")
    include("../control/model_predictive_guidance.jl")
    include("../control/sm_mpg.jl")
    include("../control/open_loop_control.jl")

    # Integration callbacks
    include("../simulation/callbacks.jl")
end # module SimulatorModel
