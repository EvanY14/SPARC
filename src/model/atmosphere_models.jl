# using PythonCall

function atmospheric_density(LatLonAlt::Tuple, t::Real, atmosphere_model::PolyfitAtmosphere, disturbance::Bool=false)
    polyfit_coefficients = atmosphere_model.polyfit_coefficients
    # power = zeros(Real, length(polyfit_coefficients))
    exponent = 0.0
    # Convert height from meters to kilometers
    h = LatLonAlt[3] * 1e-3
    # Calculate the polynomial value at height h
    for i in eachindex(polyfit_coefficients)
        exponent += polyfit_coefficients[i] * (h)^(length(polyfit_coefficients)-i)
    end
    # Calculate the density
    ρ = exp(exponent)
    return ρ, SVector{3, Float64}(0.0, 0.0, 0.0)
end

function atmospheric_density(LatLonAlt::Tuple, t::Float64, atmosphere_model::ExponentialAtmosphere, disturbance::Bool=false)
    h = LatLonAlt[3]  # Altitude in meters
    ρ = atmosphere_model.surface_density * exp(-h*1e-3/atmosphere_model.scale_height)
    if disturbance
        # Apply a random disturbance of up to ±5%
        disturbance_factor = 1.0 + (rand() * 0.1 - 0.05)
        ρ *= disturbance_factor
    end
    return ρ, SVector{3, Float64}(0.0, 0.0, 0.0)
end

function atmospheric_density(LatLonAlt::Tuple, t::Float64, atmosphere_model::GramAtmosphere, disturbance::Bool=false)
    gram = atmosphere_model.gram
    atmosphere = atmosphere_model.gram_atmosphere
    alt = Float64(LatLonAlt[3] * 1e-3)  # Convert altitude to km
    lat = rad2deg(Float64(LatLonAlt[1]))  # Convert latitude to degrees
    lon = rad2deg(Float64(LatLonAlt[2]))  # Convert longitude to degrees
    position = gram.Position()
    position.height = alt
    position.latitude = lat
    position.longitude = lon

    position.elapsedTime = t # Time since start in s
    atmosphere.setPosition(position)
    atmosphere.update()
    atmos = atmosphere.getAtmosphereState()
    rho = pyconvert(Float64, disturbance ? atmos.perturbedDensity : atmos.density)
    # T = pyconvert(Float64, atmos.temperature)
    wind = SVector{3, Float64}([pyconvert(Float64, disturbance ? atmos.perturbedEWWind : atmos.ewWind),
            pyconvert(Float64, disturbance ? atmos.perturbedNSWind : atmos.nsWind),
            pyconvert(Float64, atmos.verticalWind)])
    return rho, wind
end