# Monte Carlo Uncertainty Model

This note documents the uncertainty model used in the current Monte Carlo campaign implemented in `src/run_monte_carlo.jl`.

## Overview

The current Monte Carlo study applies **atmospheric uncertainty**, not broad vehicle-parameter uncertainty.

In particular, the Monte Carlo runs perturb:

- atmospheric density
- horizontal wind components

The current Monte Carlo runs do **not** perturb:

- initial conditions
- vehicle mass
- reference area
- aerodynamic coefficient model
- gravitational parameter
- target state
- reference trajectory
- controller weighting matrices

Therefore, the present Monte Carlo analysis should be described as an **atmospheric uncertainty study** rather than a full parametric uncertainty study.

## Atmospheric Uncertainty Source

The uncertainty model is based on the GRAM atmosphere interface for Earth, with MERRA2 data enabled in the GRAM configuration.

The Earth GRAM object is created in `ModelTypes.GramAtmosphere(...)`, where the code sets:

- Earth data path: `GRAM_Data/Earth/data/`
- MERRA2 parameters for the Earth atmosphere model
- perturbation scale
- random seed behavior

In the current implementation:

- the GRAM perturbation scale is set to `1.5`
- the nominal case uses a fixed seed of `1001`
- each Monte Carlo case uses a fresh random seed drawn from `1:10000`

This means each Monte Carlo run samples a different atmospheric realization from the same GRAM/MERRA2-based uncertainty model.

## How Monte Carlo Mode Is Enabled

In `run_monte_carlo.jl`, each Monte Carlo simulation is executed with:

- `disturbance = true`
- `monte_carlo = true`

The nominal overlay run, when enabled, is executed with:

- `disturbance = false`
- `monte_carlo = false`

So the difference between the nominal and Monte Carlo trajectories is driven by whether the atmospheric model returns the nominal or perturbed atmospheric state.

## Quantities That Are Perturbed

For the GRAM atmosphere model, the disturbed simulation uses:

- `perturbedDensity`
- `perturbedEWWind`
- `perturbedNSWind`

The current implementation uses the following wind vector:

```math
\mathbf{w} =
\begin{bmatrix}
w_{EW} \\
w_{NS} \\
w_{V}
\end{bmatrix}
```

where:

- `w_EW` is the east-west wind from GRAM
- `w_NS` is the north-south wind from GRAM
- `w_V` is the vertical wind

In the present code:

- east-west wind uses `perturbedEWWind` when `disturbance=true`
- north-south wind uses `perturbedNSWind` when `disturbance=true`
- vertical wind uses `verticalWind` directly

So the Monte Carlo perturbation explicitly applies to the horizontal wind components, while the vertical wind is taken from the GRAM vertical-wind field without a separate perturbed branch in the current implementation.

## How the Uncertainty Enters the Dynamics

The atmospheric callback updates the local atmosphere throughout the trajectory by passing the current latitude, longitude, altitude, and time into the atmosphere model.

At each callback update, the simulation stores:

- atmospheric density `ρ`
- wind vector `w`

The 6-DOF point-mass dynamics then compute the wind-relative velocity:

```math
\mathbf{v}_{rel} = \mathbf{v} - \mathbf{w}
```

and its norm:

```math
v_{rel} = \lVert \mathbf{v}_{rel} \rVert
```

This affects the aerodynamic and heating quantities through:

```math
D = \tfrac{1}{2} \rho v_{rel}^2 C_D A
```

```math
L = \tfrac{1}{2} \rho v_{rel}^2 C_L A
```

```math
\dot q = C_1 \rho^n v_{rel}^m
```

Therefore, the atmospheric uncertainty changes:

- drag
- lift
- deceleration
- flight-path evolution
- heading evolution
- convective heating rate

## Practical Interpretation

The Monte Carlo spread seen in the final landing locations and Cartesian terminal errors is produced by atmospheric realization changes alone.

That spread is driven primarily by:

1. density perturbations, which modify aerodynamic force and heat-rate levels
2. horizontal wind perturbations, which modify the wind-relative velocity and therefore the aerodynamic force direction/magnitude

## What Is Held Fixed

For clarity, the following quantities remain fixed across the Monte Carlo campaign:

- initial state from the reference initialization
- vehicle mass and reference area
- aerodynamic coefficient law as a function of angle of attack
- planetary constants
- controller structure
- controller weights and tuned gains
- target state and reference trajectory

## Recommended Wording For The Paper

The following paragraph is consistent with the current implementation and can be adapted directly into the paper:

> The Monte Carlo analysis in this work models atmospheric uncertainty through the Earth GRAM environment with MERRA2-based atmospheric data. For each Monte Carlo run, GRAM perturbations are enabled with perturbation scale 1.5 and a new random seed, producing a distinct realization of atmospheric density and horizontal wind. During trajectory propagation, the local density and wind are updated as functions of latitude, longitude, altitude, and elapsed time, and these perturbations enter the equations of motion through the wind-relative velocity, aerodynamic forces, and convective heat-rate model. No uncertainty is applied to the initial state, vehicle mass, aerodynamic model, reference trajectory, target state, or controller tuning parameters. Consequently, the reported Monte Carlo dispersion should be interpreted as an atmospheric uncertainty study rather than a full parametric uncertainty study.

## Important Limitation

If the paper intends to claim robustness to broader system uncertainty, the current Monte Carlo setup is not yet sufficient by itself. A broader uncertainty campaign would require adding perturbations to some combination of:

- initial state
- mass
- aerodynamic coefficients
- actuator behavior
- sensor noise
- target-state mismatch
- guidance/model mismatch

At present, those effects are not sampled in `src/run_monte_carlo.jl`.
