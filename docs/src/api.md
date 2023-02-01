# API

## Correlation Measurements

- [`greens!`](@ref)
- [`density_correlation!`](@ref)
- [`spin_x_correlation!`](@ref)
- [`spin_z_correlation!`](@ref)
- [`pair_correlation!`](@ref)
- [`bond_correlation!`](@ref)

```@docs
greens!
density_correlation!
spin_x_correlation!
spin_z_correlation!
pair_correlation!
bond_correlation!
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

- [`fourier_transform!`](@ref)
- [`susceptibility!`](@ref)
- [`susceptibility`](@ref)

```@docs
fourier_transform!
susceptibility!
susceptibility
```

## Developer API

- [`JDQMCMeasurements.contract_Gr0!`](@ref)
- [`JDQMCMeasurements.contract_G00!`](@ref)
- [`JDQMCMeasurements.contract_δGr0!`](@ref)
- [`JDQMCMeasurements.contract_Grr_G00!`](@ref)
- [`JDQMCMeasurements.contract_G00_Grr!`](@ref)
- [`JDQMCMeasurements.contract_Gr0_Gr0!`](@ref)
- [`JDQMCMeasurements.contract_G0r_G0r!`](@ref)
- [`JDQMCMeasurements.contract_G0r_Gr0!`](@ref)
- [`JDQMCMeasurements.contract_Gr0_G0r!`](@ref)
- [`JDQMCMeasurements.simpson`](@ref)

```@docs
JDQMCMeasurements.contract_Gr0!
JDQMCMeasurements.contract_G00!
JDQMCMeasurements.contract_δGr0!
JDQMCMeasurements.contract_Grr_G00!
JDQMCMeasurements.contract_G00_Grr!
JDQMCMeasurements.contract_Gr0_Gr0!
JDQMCMeasurements.contract_G0r_G0r!
JDQMCMeasurements.contract_G0r_Gr0!
JDQMCMeasurements.contract_Gr0_G0r!
JDQMCMeasurements.simpson
```