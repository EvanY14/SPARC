module ModelTypes
    using StaticArrays

    export EDLParams, PolyfitAtmosphere

    @kwdef struct EDLParams
        mass::Float64 = 0.0
        Cd::Float64 = 0.0
        Cl::Float64 = 0.0
        area::Float64 = 0.0
        μ::Float64 = 0.0          # Gravitational parameter
        R::Float64 = 0.0          # Planetary radius
        β::Float64 = 0.0          # Bank angle, radians, from control input
        atmospheric_density::Function = (h) -> 0.0  # Function of altitude
    end

    @kwdef struct PolyfitAtmosphere{N}
        polyfit_coefficients::SVector{N, Float64} = SVector{N, Float64}(zeros(N))
    end
end # module ModelTypes