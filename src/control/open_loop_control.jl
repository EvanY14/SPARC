function openloopcontrol(integrator)
	current_control_fallback = [Float64(integrator.p.α), Float64(integrator.p.β)]
	α_ref, β_ref = _reference_control_at(integrator.t, current_control_fallback)
	α_cmd, β_cmd = _rate_limited_control_from_integrator(integrator, [α_ref, β_ref])
	return β_cmd, α_cmd
end

const openloop = openloopcontrol
