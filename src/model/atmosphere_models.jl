# using PythonCall

function atmospheric_density(h::Float64, atmosphere_model::PolyfitAtmosphere, disturbance::Bool=false)
    polyfit_coefficients = atmosphere_model.polyfit_coefficients
    power = zeros(length(polyfit_coefficients))
    # Convert height from meters to kilometers
    h = h * 1e-3
    # Calculate the polynomial value at height h
    for i=1:length(polyfit_coefficients)
        power[i] = (h)^(length(polyfit_coefficients)-i)
    end
    # Calculate the exponent term of the density using the polynomial coefficients
    exponent = sum(polyfit_coefficients .* power)
    # Calculate the density
    ρ = exp(exponent)
    if disturbance
        # Apply a random disturbance of up to ±10%
        disturbance_factor = 1.0 + (randn() * 0.1)
        ρ *= disturbance_factor
    end
    return ρ
end

function atmospheric_density(h::Float64, atmosphere_model::ExponentialAtmosphere, disturbance::Bool=false)
    ρ = atmosphere_model.surface_density * exp(-h*1e-3/atmosphere_model.scale_height)
    if disturbance
        # Apply a random disturbance of up to ±5%
        disturbance_factor = 1.0 + (rand() * 0.1 - 0.05)
        ρ *= disturbance_factor
    end
    return ρ
end

function atmospheric_density(integrator, atmosphere_model::GramAtmosphere, disturbance::Bool=false)
    gram = atmosphere_model.gram
    atmosphere = atmosphere_model.gram_atmosphere
    u = integrator.u
    t = integrator.t
    alt = u[1] * 1e-3  # Convert altitude to km
    lat = rad2deg(u[3])  # Convert latitude to degrees
    lon = rad2deg(u[2])  # Convert longitude to degrees
    position = gram.Position()
    position.height = u[1] * 1e-3
    position.latitude = lat
    position.longitude = lon

    position.elapsedTime = t # Time since start in s
    atmosphere.setPosition(position)
    atmosphere.update()
    atmos = atmosphere.getAtmosphereState()
    rho = pyconvert(Float64, atmos.density)
    T = pyconvert(Float64, atmos.temperature)
    wind = SVector{3, Float64}([pyconvert(Float64, disturbance ? atmos.perturbedEWWind : atmos.ewWind),
            pyconvert(Float64, disturbance ? atmos.perturbedNSWind : atmos.nsWind),
            pyconvert(Float64, atmos.verticalWind)])
    return rho
end