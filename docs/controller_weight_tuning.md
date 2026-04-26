# Controller Weight Tuning Method

This note documents how the weighting parameters for `MPG`, `MPG-IA`, and `MPG-SM` are tuned in the current codebase.

## Scope

The tuning workflow is implemented in:

- `src/tune_mpg_controllers.jl`

The tuned parameters are stored in:

- `src/control/mpg_shared_tuning.jl`
- `src/control/mpg_integral_tracking.jl`
- `src/control/sm_mpg.jl`

## State Ordering

Whenever a 6-state weight or scale vector is used, the state order is:

1. Altitude `h`
2. Longitude `ϕ`
3. Latitude `θ`
4. Velocity `v`
5. Flight-path angle `γ`
6. Azimuth `ψ`

## Evaluation Environment Used During Tuning

The tuning script evaluates candidate parameter sets in a deterministic environment, not in Monte Carlo.

The evaluation setup is:

- reference trajectory read from `optimal_trajectory.csv`
- same initial condition and target used by the reference initialization
- deterministic atmospheric density model:
  - `SimulatorModel.earth_atmosphere_density(h)`
- zero wind:
  - `SVector(0.0, 0.0, 0.0)`
- fixed-step simulation:
  - `TUNE_SIM_DT = 0.2 s` by default

This means the tuning is aimed at nominal tracking performance, not robustness to atmospheric uncertainty.

## Tuning Objective

Each candidate parameter set is evaluated by running the closed-loop trajectory and computing terminal metrics.

The scalar objective minimized by the tuning script is:

```math
\mathrm{score} = e_{pos,\mathrm{km}} + w_v \, e_{vel,\mathrm{km/s}}
```

where:

- `e_pos,km` is the final Cartesian position error norm in km
- `e_vel,km/s` is the final Cartesian velocity error norm in km/s
- `w_v = 4.0` by default

So the exact score used in code is:

```julia
score = position_error_km + 4.0 * velocity_error_kms
```

The script also reports:

- altitude error
- longitude error
- latitude error
- speed error
- flight-path-angle error
- azimuth error
- terminal time

but those are diagnostic outputs, not the optimization objective.

## Shared MPG Tuning

The shared MPG tuning step modifies:

- `state_scales`
- `n_horizon`
- `time_step`

These parameters are applied through:

- `SimulatorModel.set_mpg_tracking_tuning!`

### Search Procedure

The search is a staged coordinate/grid search:

1. Start from the current snapshot in `mpg_shared_tuning.jl`
2. Coarse search over:
   - `n_horizon ∈ {30, 40, 50, 60, current}`
   - `time_step ∈ {0.4, 0.5, 0.6, 0.7, current}`
3. Coordinate search on each element of `state_scales` using multiplicative factors:
   - first pass: `[0.7, 1.0, 1.3]`
   - second pass: `[0.85, 1.0, 1.15]`
4. Fine search around the best horizon and time step:
   - `n_horizon ∈ {best-5, best, best+5}`, clipped to at least `20`
   - `time_step ∈ {best-0.1, best, best+0.1}`, clipped to at least `0.3`

### Current Shared Defaults

The current tuned defaults in `src/control/mpg_shared_tuning.jl` are:

- `state_scales = [8.05e4, 0.26, 0.26, 5.0e3, 0.21, 0.5]`
- `n_horizon = 30`
- `time_step = 0.4`

The fixed normalized base weight matrices are:

- stage state weights:
  - `diag([1200, 3500, 3500, 1200, 700, 1200])`
- terminal state weights:
  - `diag([2500, 180000, 180000, 4500, 2500, 4000])`
- control weights:
  - `diag([1, 1])`

## MPG-IA Tuning

The integral-action tuning step modifies only the controller-specific add-on terms:

- integral-state normalized weights
- control-increment normalized weights
- `state_gain`
- `increment_gain`

These are applied through:

- `SimulatorModel.set_mpg_integral_tuning!`

### Search Procedure

The search is staged as follows:

1. Initialize:
   - `state_normalized_weights` from the base MPG stage-state normalized weights
   - `increment_weights` from the base MPG control weights
   - gains from the current snapshot
2. Sweep `state_gain` over:
   - `[0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08]`
3. Sweep `increment_gain` over:
   - `[0.0, 0.001, 0.002, 0.005, 0.01, 0.02]`
4. Local refinement:
   - `state_gain *= [0.5, 1.0, 1.5]`
   - `increment_gain *= [0.5, 1.0, 1.5]`

### Current MPG-IA Defaults

The current tuned defaults in `src/control/mpg_integral_tracking.jl` are:

- `state_normalized_weights = [1200, 3500, 3500, 1200, 700, 1200]`
- `increment_weights = [1.0, 1.0]`
- `state_gain = 0.0`
- `increment_gain = 0.0`

So at the moment, the tuned integral add-on is effectively inactive under the nominal tuning objective.

## MPG-SM Tuning

The sliding-mode MPG tuning step modifies:

- `lambda`
- `sliding_gain`
- `increment_gain`

These are applied through:

- `SimulatorModel.set_sm_mpg_tuning!`

### Search Procedure

The search is staged as follows:

1. Start from the current snapshot in `sm_mpg.jl`
2. Sweep `lambda` over:
   - `[0.0, 0.1, 0.2, 0.3, 0.4, 0.6]`
3. Sweep `sliding_gain` over:
   - `[0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08]`
4. Sweep `increment_gain` over:
   - `[0.0, 0.001, 0.002, 0.005, 0.01, 0.02]`
5. Local refinement:
   - `lambda = best_lambda + [-0.05, 0.0, 0.05]`, clipped to `[0.0, 1.5]`
   - `sliding_gain *= [0.5, 1.0, 1.5]`
   - `increment_gain *= [0.5, 1.0, 1.5]`

### Current MPG-SM Defaults

The current tuned defaults in `src/control/sm_mpg.jl` are:

- `lambda = 0.4`
- `sliding_gain = 0.03`
- `increment_gain = 0.01`

## Important Interpretation Note

The current tuning script does **not** optimize Monte Carlo robustness directly. It tunes the controllers against a deterministic atmosphere with zero wind and minimizes terminal tracking error only.

That means:

- the tuned values are best interpreted as nominal-performance weights
- a separate robustness-oriented tuning pass would be needed if the paper wants Monte Carlo-optimal weights

## How To Rerun The Tuning

Run:

```bash
julia --project=.SPARC src/tune_mpg_controllers.jl
```

Optional environment variables:

- `SPARC_TUNE_SIM_DT`
- `SPARC_TUNE_VEL_WEIGHT`

## Suggested Paper Summary

A concise description suitable for the paper is:

> The controller weights were tuned using a deterministic closed-loop simulation of the reference entry trajectory. A staged grid/coordinate search was used to minimize a terminal score equal to the final Cartesian position error norm plus four times the final Cartesian velocity error norm. First, the shared MPG parameters (state scaling, horizon length, and guidance sample time) were tuned. Then, the controller-specific augmentation terms for MPG with integral action and MPG with sliding-mode augmentation were tuned while holding the shared MPG parameters fixed. Monte Carlo atmospheric uncertainty was not included in this tuning stage; therefore, the resulting weights should be interpreted as nominal-performance tuning values.
