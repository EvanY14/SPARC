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