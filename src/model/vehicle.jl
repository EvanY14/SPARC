Base.@kwdef struct VehicleDefinition
    mass::Float64
    reference_area::Float64
end

const VEHICLE = VehicleDefinition(
    mass = 92000.0,
    reference_area = 249.9,
)

const VEHICLE_MASS = VEHICLE.mass
const VEHICLE_REFERENCE_AREA = VEHICLE.reference_area
