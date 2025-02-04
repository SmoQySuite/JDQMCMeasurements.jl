# API

## Correlation Measurements

- [`greens!`](@ref)
- [`density_correlation!`](@ref)
- [`spin_x_correlation!`](@ref)
- [`spin_z_correlation!`](@ref)
- [`pair_correlation!`](@ref)
- [`bond_correlation!`](@ref)
- [`current_correlation!`](@ref)

```@docs
greens!
density_correlation!
spin_x_correlation!
spin_z_correlation!
pair_correlation!
bond_correlation!
current_correlation!
```

## Scalar Measurements

- [`measure_n`](@ref)
- [`measure_double_occ`](@ref)
- [`measure_N`](@ref)
- [`measure_Nsqrd`](@ref)

```@docs
measure_n
measure_double_occ
measure_N
measure_Nsqrd
```

## Utility Functions

- [`cubic_spline_τ_to_ωn!`](@ref)
- [`fourier_transform!`](@ref)
- [`susceptibility!`](@ref)
- [`susceptibility`](@ref)
- [`jackknife`](@ref)

```@docs
cubic_spline_τ_to_ωn!
fourier_transform!
susceptibility!
susceptibility
jackknife
```

## Developer API

- [`JDQMCMeasurements.average_Gr0`](@ref)
- [`JDQMCMeasurements.average_ηGr0`](@ref)
- [`JDQMCMeasurements.contract_G00!`](@ref)
- [`JDQMCMeasurements.contract_Gr0!`](@ref)
- [`JDQMCMeasurements.contract_Grr_G00!`](@ref)
- [`JDQMCMeasurements.contract_Gr0_Gr0!`](@ref)
- [`JDQMCMeasurements.contract_G0r_Gr0!`](@ref)
- [`JDQMCMeasurements.simpson`](@ref)

```@docs
JDQMCMeasurements.average_Gr0
JDQMCMeasurements.average_ηGr0
JDQMCMeasurements.contract_G00!
JDQMCMeasurements.contract_Gr0!
JDQMCMeasurements.contract_Grr_G00!
JDQMCMeasurements.contract_Gr0_Gr0!
JDQMCMeasurements.contract_G0r_Gr0!
JDQMCMeasurements.simpson
```