function _earth_atmosphere_log_density_derivative(h::Real)
	h_km = Float64(h) * 1.0e-3
	coeffs = earth_atmosphere_polyfit_coefficients(h)
	degree = length(coeffs) - 1
	derivative = 0.0
	for (idx, c) in enumerate(coeffs[1:end - 1])
		derivative = derivative * h_km + (degree - idx + 1) * c
	end
	return derivative * 1.0e-3
end

function _analytical_continuous_linearization_si(
	x_ref::AbstractVector{<:Real},
	u_ref::AbstractVector{<:Real};
	mass::Real = VEHICLE_MASS,
	area::Real = VEHICLE_REFERENCE_AREA,
	μ::Real = 3.986004418e14,
	R::Real = 6378137.0,
)
	h = Float64(x_ref[1])
	θ = Float64(x_ref[3])
	v = Float64(x_ref[4])
	γ = Float64(x_ref[5])
	ψ = Float64(x_ref[6])
	α = Float64(u_ref[1])
	β = Float64(u_ref[2])

	m = Float64(mass)
	S = Float64(area)
	μ_si = Float64(μ)
	R_si = Float64(R)
	r = R_si + h
	r2 = r^2
	r3 = r^3

	α_deg = rad2deg(α)
	a0 = -0.20704
	a1 = 0.029244
	b0 = 0.07854
	b1 = -0.61592e-2
	b2 = 0.621408e-3
	cL = a0 + a1 * α_deg
	cD = b0 + b1 * α_deg + b2 * α_deg^2
	dcL_dα = a1 * 180.0 / π
	dcD_dα = (b1 + 2.0 * b2 * α_deg) * 180.0 / π

	ρ = earth_atmosphere_density(h)
	dρ_dh = ρ * _earth_atmosphere_log_density_derivative(h)
	K = 0.5 * S / m
	g = μ_si / r2

	sinγ = sin(γ)
	cosγ = cos(γ)
	tanγ = tan(γ)
	secγ = 1.0 / cosγ
	sinψ = sin(ψ)
	cosψ = cos(ψ)
	tanθ = tan(θ)
	secθ = 1.0 / cos(θ)
	sinβ = sin(β)
	cosβ = cos(β)

	Ac = zeros(6, 6)
	Bc = zeros(6, 2)

	# hdot = v sin(γ)
	Ac[1, 4] = sinγ
	Ac[1, 5] = v * cosγ

	# ϕdot = v/r cos(γ) sin(ψ) sec(θ)
	Ac[2, 1] = -v * cosγ * sinψ * secθ / r2
	Ac[2, 3] = v * cosγ * sinψ * secθ * tanθ / r
	Ac[2, 4] = cosγ * sinψ * secθ / r
	Ac[2, 5] = -v * sinγ * sinψ * secθ / r
	Ac[2, 6] = v * cosγ * cosψ * secθ / r

	# θdot = v/r cos(γ) cos(ψ)
	Ac[3, 1] = -v * cosγ * cosψ / r2
	Ac[3, 4] = cosγ * cosψ / r
	Ac[3, 5] = -v * sinγ * cosψ / r
	Ac[3, 6] = -v * cosγ * sinψ / r

	# vdot = -K cD ρ v^2 - g sin(γ)
	Ac[4, 1] = -K * cD * dρ_dh * v^2 + 2.0 * μ_si * sinγ / r3
	Ac[4, 4] = -2.0 * K * cD * ρ * v
	Ac[4, 5] = -g * cosγ
	Bc[4, 1] = -K * dcD_dα * ρ * v^2

	# γdot = K cL ρ v cos(β) + cos(γ) * (v/r - g/v)
	w = v / r - g / v
	Ac[5, 1] = K * cL * dρ_dh * v * cosβ + cosγ * (-v / r2 + 2.0 * μ_si / (r3 * v))
	Ac[5, 4] = K * cL * ρ * cosβ + cosγ * (1.0 / r + g / v^2)
	Ac[5, 5] = -sinγ * w
	Bc[5, 1] = K * dcL_dα * ρ * v * cosβ
	Bc[5, 2] = -K * cL * ρ * v * sinβ

	# ψdot = K cL ρ v sin(β) sec(γ) + v/r cos(γ) sin(ψ) tan(θ)
	Ac[6, 1] = K * cL * dρ_dh * v * sinβ * secγ - v * cosγ * sinψ * tanθ / r2
	Ac[6, 3] = v * cosγ * sinψ * secθ^2 / r
	Ac[6, 4] = K * cL * ρ * sinβ * secγ + cosγ * sinψ * tanθ / r
	Ac[6, 5] = K * cL * ρ * v * sinβ * secγ * tanγ - v * sinγ * sinψ * tanθ / r
	Ac[6, 6] = v * cosγ * cosψ * tanθ / r
	Bc[6, 1] = K * dcL_dα * ρ * v * sinβ * secγ
	Bc[6, 2] = K * cL * ρ * v * cosβ * secγ

	return Ac, Bc
end
