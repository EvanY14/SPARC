function openloopcontrol(integrator)
	current_control_fallback = [Float64(integrator.p.α), Float64(integrator.p.β)]
	α_ref, β_ref = _reference_control_at(integrator.t, current_control_fallback)
	return β_ref, α_ref
end

const openloop = openloopcontrol
