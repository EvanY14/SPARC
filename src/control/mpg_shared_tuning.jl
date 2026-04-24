using LinearAlgebra

const _MPG_STATE_SCALES = Ref([1.0e5, 0.2, 0.2, 5.0e3, 0.3, 0.5])
const _MPG_STAGE_STATE_NORMALIZED_WEIGHTS = Diagonal([1200.0, 3500.0, 3500.0, 1200.0, 700.0, 1200.0])
const _MPG_TERMINAL_STATE_NORMALIZED_WEIGHTS = Diagonal([2500.0, 180000.0, 180000.0, 4500.0, 2500.0, 4000.0])
const _MPG_CONTROL_NORMALIZED_WEIGHTS = Diagonal([1.0, 1.0])
const _MPG_DEFAULT_HORIZON = Ref(40)
const _MPG_DEFAULT_TIME_STEP = Ref(0.6)

function mpg_state_scales()
    return Float64.(_MPG_STATE_SCALES[])
end

function _mpg_state_scale_matrix()
    return Diagonal(1.0 ./ mpg_state_scales())
end

function _mpg_stage_state_normalized_weight_matrix()
    return Matrix{Float64}(_MPG_STAGE_STATE_NORMALIZED_WEIGHTS)
end

function _mpg_terminal_state_normalized_weight_matrix()
    return Matrix{Float64}(_MPG_TERMINAL_STATE_NORMALIZED_WEIGHTS)
end

function _mpg_control_normalized_weight_matrix()
    return Matrix{Float64}(_MPG_CONTROL_NORMALIZED_WEIGHTS)
end

function _mpg_stage_state_cost_matrix()
    G = _mpg_state_scale_matrix()
    return Matrix{Float64}(G' * _MPG_STAGE_STATE_NORMALIZED_WEIGHTS * G)
end

function _mpg_terminal_state_cost_matrix()
    G = _mpg_state_scale_matrix()
    return Matrix{Float64}(G' * _MPG_TERMINAL_STATE_NORMALIZED_WEIGHTS * G)
end

function _mpg_control_cost_matrix()
    return Matrix{Float64}(_MPG_CONTROL_NORMALIZED_WEIGHTS)
end

function mpg_default_horizon()
    return _MPG_DEFAULT_HORIZON[]
end

function mpg_default_time_step()
    return _MPG_DEFAULT_TIME_STEP[]
end

function mpg_tracking_tuning_snapshot()
    return (
        state_scales = mpg_state_scales(),
        n_horizon = mpg_default_horizon(),
        time_step = mpg_default_time_step(),
    )
end

function set_mpg_tracking_tuning!(; state_scales=nothing, n_horizon=nothing, time_step=nothing)
    if state_scales !== nothing
        new_scales = Float64.(collect(state_scales))
        length(new_scales) == 6 || throw(ArgumentError("state_scales must have length 6"))
        all(isfinite, new_scales) || throw(ArgumentError("state_scales must be finite"))
        all(>(0.0), new_scales) || throw(ArgumentError("state_scales must be positive"))
        _MPG_STATE_SCALES[] = new_scales
    end

    if n_horizon !== nothing
        new_horizon = Int(n_horizon)
        new_horizon >= 2 || throw(ArgumentError("n_horizon must be at least 2"))
        _MPG_DEFAULT_HORIZON[] = new_horizon
    end

    if time_step !== nothing
        new_time_step = Float64(time_step)
        (isfinite(new_time_step) && new_time_step > 0.0) || throw(ArgumentError("time_step must be positive and finite"))
        _MPG_DEFAULT_TIME_STEP[] = new_time_step
    end

    return mpg_tracking_tuning_snapshot()
end
