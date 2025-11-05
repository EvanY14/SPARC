module SimulatorModel
    using StaticArrays
    using Reexport

    export edl_dynamics, atmospheric_density

    include("types.jl")
    @reexport using .ModelTypes

    include("../simulation/simulator.jl")
end # module SimulatorModel