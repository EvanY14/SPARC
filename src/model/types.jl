module ModelTypes
    using StaticArrays
    using PythonCall

    export EDLParams, EDLCache
    export PolyfitAtmosphere, ExponentialAtmosphere, GramAtmosphere 
    export DateTime

    @kwdef struct DateTime
        year::Int64 = 2024
        month::Int64 = 1
        day::Int64 = 1
        hours::Int64 = 0
        minutes::Int64 = 0
        secs::Float64 = 0.0
    end

    @kwdef mutable struct EDLCache
        atmospheric_density::Float64 = 0.0
        β::Float64 = 0.0
    end

    @kwdef mutable struct EDLParams
        mass::Float64 = 0.0
        Cd::Float64 = 0.0
        Cl::Float64 = 0.0
        area::Float64 = 0.0
        μ::Float64 = 0.0          # Gravitational parameter
        R::Float64 = 0.0          # Planetary radius
        β_function::Function = (u, p, t) -> 0.0          # Bank angle, radians, from control input
        β::Float64 = 0.0          # Current bank angle
        target_altitude::Float64 = 0.0  # Altitude to terminate simulation
        atmospheric_density_function::Function = (h) -> 0.0  # Function of altitude
        atmospheric_density::Float64 = 0.0  # Current atmospheric density
        wind::SVector{3, Float64} = SVector{3, Float64}(0.0, 0.0, 0.0)  # Current wind vector
        cache::EDLCache = EDLCache()
    end

    @kwdef struct PolyfitAtmosphere{N}
        polyfit_coefficients::SVector{N, Float64} = SVector{N, Float64}(zeros(N))
    end

    @kwdef struct ExponentialAtmosphere
        surface_density::Float64 = 0.0
        scale_height::Float64 = 0.0
    end

    struct GramAtmosphere
        gram::Any
        gram_atmosphere::Any
    end

    function GramAtmosphere(gram_directory::String, gram_data_directory::String, monte_carlo::Bool, planet_name::String, date::DateTime)
        sys = pyimport("sys")
        os = pyimport("os")
        if !(gram_directory in pyconvert(Vector{String}, sys.path))
            sys.path.append(gram_directory)
        end
        gram = pyimport("gram")
        inputParameters = Dict("earth" => gram.EarthInputParameters(),
                            "mars" => gram.MarsInputParameters(),
                            "venus" => gram.VenusInputParameters(),
                            "titan" => gram.TitanInputParameters())
        
        namelistReaders = Dict("earth" => gram.EarthNamelistReader(),
                            "mars" => gram.MarsNamelistReader(),
                            "venus" => gram.VenusNamelistReader(),
                            "titan" => gram.TitanNamelistReader())
            
        atmospheres = Dict("earth" => gram.EarthAtmosphere(),
                        "mars" => gram.MarsAtmosphere(),
                        "venus" => gram.VenusAtmosphere(),
                        "titan" => gram.TitanAtmosphere())

        input_parameters = inputParameters[planet_name]

        # Mars has some weird specific parameters, so this line is just to check to make sure the it doesn't do it for the other planets
        if planet_name == "mars"
            # input_parameters.dataPath = os.path.join(os.path.dirname(os.path.abspath(@__FILE__)),"..", "GRAM_Data", "Mars", "data", "")
            input_parameters.dataPath = gram_data_directory * "/Mars/data/"
            if !Bool(os.path.exists(input_parameters.dataPath))
                throw(ArgumentError("GRAM data path not found: " * input_parameters.dataPath))
            end
        end

        if planet_name == "earth"
            # input_parameters.dataPath = os.path.join(os.path.dirname(os.path.abspath(@__FILE__)),"..", "GRAM_Data", "Mars", "data", "")
            input_parameters.dataPath = gram_data_directory * "/Earth/data/"
            if !Bool(os.path.exists(input_parameters.dataPath))
                throw(ArgumentError("GRAM data path not found: " * input_parameters.dataPath))
            end
        end

        reader = namelistReaders[planet_name]
        reader.tryGetSpicePath(input_parameters)

        gram_atmosphere = atmospheres[planet_name]
        gram_atmosphere.setInputParameters(input_parameters)
        
        if planet_name == "earth"
            gram_atmosphere.setMERRA2Parameters(0, -90.0, 90.0, 0.0, 359.99999)
        end

        gram_atmosphere.setPerturbationScales(1.5)
        gram_atmosphere.setMinRelativeStepSize(0.5)
        if monte_carlo
            gram_atmosphere.setSeed(Int(round(rand()*10000)))
        else
            gram_atmosphere.setSeed(1001)
            # gram_atmosphere.setSeed(Int(round(rand()*10000)))
        end

        if planet_name == "mars"
            gram_atmosphere.setMOLAHeights(false)
        end

        ttime = gram.GramTime()
        ttime.setStartTime(date.year, date.month, date.day, date.hours, date.minutes, date.secs, gram.UTC, gram.PET)
        gram_atmosphere.setStartTime(ttime)

        return GramAtmosphere(gram, gram_atmosphere)
    end
end # module ModelTypes