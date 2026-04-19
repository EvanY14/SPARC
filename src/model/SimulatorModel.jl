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
    export mpc, ssimpc, trackingmpc, trackingmpc_shrinking_horizon, trackingmpc_shrinking, openloopcontrol, openloop
    include("types.jl")
    @reexport using .ModelTypes

    # Simulator models
    include("../simulation/simulator.jl")
    include("atmosphere_models.jl")

    # Control strategies
    include("../control/regular_mpc.jl")
    include("../control/ssi_mpc.jl")
    include("../control/tracking_mpc.jl")
    include("../control/tracking_mpc_shrinking_horizon.jl")
    include("../control/open_loop_control.jl")

    # Integration callbacks
    include("../simulation/callbacks.jl")
end # module SimulatorModel
