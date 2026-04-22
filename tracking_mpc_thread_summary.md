# Tracking MPC Debugging Summary

Date: 2026-04-22

This note summarizes the debugging thread around the tracking MPC implementation, why it was tracking worse than the MPG controller, and what changes were made. It is intended as future context before continuing work on `tracking_mpc.jl`, `tracking_mpc_shrinking_horizon.jl`, or the reference trajectory generation.

## Original Problem

The tracking MPC formulations matched the reference reasonably well early in flight, then began saturating the commanded bank/angle-of-attack controls while the MPG algorithm tracked the same reference much better.

The main symptoms were:

- tracking error grew late in the trajectory,
- tracking MPC commands saturated or chattered near the end,
- MPG remained more stable against the same broad reference,
- some attempted changes improved one behavior while worsening another.

## Important Files

- `src/control/tracking_mpc.jl`: standard tracking MPC controller.
- `src/control/tracking_mpc_shrinking_horizon.jl`: shrinking-horizon tracking MPC controller.
- `src/control/model_predictive_guidance.jl`: MPG controller used as the stronger baseline.
- `src/control/reference_trajectory.jl`: reference trajectory generation.
- `src/simulation/callbacks.jl`: simulation callback that applies controls and rate limits them.
- `src/control/control_limits.jl`: shared control bounds/rate limiting helpers.
- `src/model/vehicle.jl`: centralized vehicle constants.
- `src/reference/reentry_jacobians_si.jl`: analytical SI-unit Jacobian implementation.

## Vehicle Model Mismatch

One major early issue was that vehicle parameters were scattered through the codebase. The tracking MPC path used hardcoded values that did not match the vehicle used by the simulator/reference generation.

The mismatch included values such as:

- mass: `3257` in one tracking path versus `92000` elsewhere,
- reference area: `15.904` versus `249.9`.

This made the prediction model used by tracking MPC inconsistent with the plant/reference dynamics, which can explain good short-term agreement followed by saturation as accumulated prediction error grows.

Change made:

- added `src/model/vehicle.jl`,
- introduced `VehicleDefinition`, `VEHICLE`, `VEHICLE_MASS`, and `VEHICLE_REFERENCE_AREA`,
- updated the relevant model, MPC, reference, and Monte Carlo code to reference the centralized vehicle definition.

## Analytical Jacobians

The tracking MPC was using finite-difference Jacobians even though analytical Jacobian code existed. The older generated Jacobian file appeared to be in an incompatible unit convention, so directly swapping it in was risky.

Change made:

- added `src/reference/reentry_jacobians_si.jl`,
- implemented a parameterized analytical continuous linearization in SI units,
- routed the tracking MPC linearization through the SI analytical Jacobian helper.

This reduced one source of numerical inconsistency, though it was not by itself enough to fully fix tracking quality.

## Energy-Based Reformulation Attempt

The MPG algorithm is based around an energy-indexed formulation, while the tracking MPC formulations were time-indexed. An attempt was made to reformulate tracking MPC around the MPG-style energy domain.

Observed result:

- the code crashed initially,
- after fixes, the results were significantly worse,
- the energy reformulation was reverted back to the time-domain formulation.

Current state:

- `tracking_mpc.jl` and `tracking_mpc_shrinking_horizon.jl` are back to a time-based horizon,
- the energy-domain attempt should be considered an experiment, not the current baseline.

Likely reason the energy attempt performed badly:

- MPG is not just "time MPC with energy as the independent variable"; it depends on a consistent energy-domain reference, dynamics scaling, discretization, and constraints. A partial conversion can introduce worse mismatch than the original time formulation.

## Reference Trajectory Slack

The reference trajectory generation previously used hardcoded terminal altitude/velocity behavior. Slack variables were added for altitude and velocity to soften those constraints.

Intent:

- avoid infeasibility or over-constraining the reference trajectory,
- allow the optimizer to trade small terminal altitude/velocity deviations against the rest of the objective.

Current caution:

- reference slack can help produce a feasible reference, but if penalties are poorly tuned it can also produce a reference that is easier for the planner than for the tracking controller.

## Why MPG Still Tracks Better

The likely reasons MPG outperforms the tracking MPC are a combination of formulation and implementation details:

- MPG and the reference are more internally consistent.
- MPG uses the paper-style deviation formulation and receding-horizon structure more directly.
- Tracking MPC remains more sensitive to prediction/reference mismatch.
- Terminal behavior is delicate; when little horizon remains, terminal penalties and active constraints can dominate.
- Rate limits and previous-control memory matter a lot near the end of flight.

In short: the tracking MPC was not failing because of one single bug. It was affected by several small mismatches that become large late in the trajectory.

## Late-Trajectory Chattering

A specific chattering source was identified in the simulation callback/control memory interaction.

Before the fix:

- the MPC/MPG controller solved for a raw command,
- the controller stored that raw command in `mpc_params.prev_x`,
- the simulation callback then rate-limited the command before applying it to the vehicle,
- the next MPC solve used the raw command as the previous command, even though the vehicle actually received the limited command.

This creates an artificial mismatch in the control-rate penalty and can cause oscillatory corrections near the end of the trajectory.

Change made in `src/simulation/callbacks.jl`:

- the callback now applies the rate limiter,
- writes the limited command to `integrator.p.α` and `integrator.p.β`,
- syncs the limited/applied `[α, β]` back into `mpc_params.prev_x[7:8]` for the two-control controllers:
  - `trackingmpc`,
  - `trackingmpc_shrinking_horizon`,
  - `trackingmpc_shrinking`,
  - `model_predictive_guidance`,
  - `mpg`.

This was intentionally not applied generically to all controllers because SSI MPC uses a different `prev_x` layout.

Observed result:

- this made the results a little bit better, but did not fully solve the tracking issue.

## Current Baseline State

The current intended baseline is:

- time-domain tracking MPC, not energy-domain tracking MPC,
- centralized vehicle parameters,
- SI analytical Jacobians,
- reference trajectory with altitude/velocity slack,
- callback-level sync between applied rate-limited controls and two-control MPC previous-control memory.

Known environment issue:

- full Julia execution has been blocked in this workspace by missing packages such as `Revise`, `StaticArrays`, or `ForwardDiff` depending on the run path,
- syntax parsing checks have been used where possible, but full simulation verification still needs a complete Julia environment.

## Remaining Hypotheses

The remaining tracking gap may come from:

- terminal cost/constraint tuning that becomes too aggressive near the end,
- horizon becoming too short or poorly conditioned near terminal time,
- reference interpolation mismatch between the planner and tracker,
- state/control normalization still not matching physical sensitivities,
- rate limits preventing recovery from errors that the optimizer assumes are recoverable,
- tracking full state too tightly instead of prioritizing the physically important output channels,
- insufficient slack or no slack in the tracking MPC itself,
- mismatch between the discrete prediction model and the simulator integration.

## Suggested Next Steps

1. Plot raw optimizer commands and applied rate-limited commands together.
2. Plot tracking error by state with the active control bounds/rate bounds overlaid.
3. Log QP status, objective, first control increment, and constraint activity at each MPC step.
4. Compare one-step prediction error against the simulator after applying the same limited control.
5. Temporarily reduce or remove terminal weights to test whether late chattering is terminal-cost driven.
6. Temporarily increase control-rate penalties to test whether oscillation is mostly actuator-memory driven.
7. Add state/output slack directly in tracking MPC if infeasibility or active constraints are forcing saturation.
8. Revisit energy-domain tracking only after creating a fully consistent energy-domain reference, dynamics scaling, and discretization.

## Quick Takeaway

The biggest confirmed issues were model consistency and control-memory consistency. Those are now improved. The remaining problem appears to be controller formulation/tuning around terminal behavior rather than an obvious single-line implementation bug.
