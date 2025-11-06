module SimulatorModel
    using StaticArrays
    using Reexport

    export edl_dynamics, atmospheric_density, altitude_termination_condition

    include("types.jl")
    @reexport using .ModelTypes

    include("../simulation/simulator.jl")
end # module SimulatorModel