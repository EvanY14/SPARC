module SimulatorModel
    using StaticArrays
    using Reexport
    using DifferentialEquations
    using PythonCall

    export edl_dynamics 
    export atmospheric_density
    export altitude_termination_condition, atmospheric_density_callback
    export saving_callback, saved_values

    include("types.jl")
    @reexport using .ModelTypes

    include("../simulation/simulator.jl")
    include("atmosphere_models.jl")
    include("../simulation/callbacks.jl")
end # module SimulatorModel