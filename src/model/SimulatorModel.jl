module SimulatorModel
    using StaticArrays
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
    export mpc, ssimpc, trackingmpc, trackingmpc_shrinking_horizon, trackingmpc_shrinking, model_predictive_guidance, mpg, mpg_integral_tracking, mpg_integral, sm_mpg_tracking, sm_mpg, openloopcontrol, openloop
    export mpg_state_scales, mpg_default_horizon, mpg_default_time_step, mpg_tracking_tuning_snapshot, set_mpg_tracking_tuning!
    export mpg_integral_tuning_snapshot, set_mpg_integral_tuning!, sm_mpg_tuning_snapshot, set_sm_mpg_tuning!
    export VehicleDefinition, VEHICLE, VEHICLE_MASS, VEHICLE_REFERENCE_AREA
    export EDLParams, MPCParams, EDLCache, TargetStates, OptimizationStates
    export PolyfitAtmosphere, ExponentialAtmosphere, GramAtmosphere, DateTime
    include("vehicle.jl")
    include("types.jl")
    using .ModelTypes: EDLParams,
                       MPCParams,
                       EDLCache,
                       TargetStates,
                       OptimizationStates,
                       PolyfitAtmosphere,
                       ExponentialAtmosphere,
                       GramAtmosphere,
                       DateTime
    include("earth_atmosphere_polyfit.jl")

    # Simulator models
    include("../simulation/simulator.jl")
    include("atmosphere_models.jl")

    # Control strategies
    include("../control/control_limits.jl")
    include("../control/regular_mpc.jl")
    include("../control/ssi_mpc.jl")
    include("../control/mpg_shared_tuning.jl")
    include("../control/tracking_mpc.jl")
    include("../control/tracking_mpc_shrinking_horizon.jl")
    include("../control/model_predictive_guidance.jl")
    include("../control/mpg_integral_tracking.jl")
    include("../control/sm_mpg.jl")
    include("../control/open_loop_control.jl")

    # Integration callbacks
    include("../simulation/callbacks.jl")
end # module SimulatorModel
