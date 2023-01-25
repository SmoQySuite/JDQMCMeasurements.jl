module JDQMCMeasurements

using LinearAlgebra
using FFTW
using LatticeUtilities
import ShiftedArrays as sa
import OffsetArrays as oa

# implement methods no exported by the package
include("utilities.jl")

# calculate the fourier transform for correlation data
include("fourier_transform.jl")
export fourier_transform!

# calculate susceptibility from correlation function
include("susceptibility.jl")
export susceptibility, susceptibility!

# implement scalar measurements
include("scalar_measurements.jl")
export measure_N, measure_n, measure_Nsqrd, measure_double_occ

# implement Green's function measurement
include("correlation_measurements/greens.jl")
export greens!

# implement density-density correlation measurement
include("correlation_measurements/density_correlation.jl")
export density_correlation!

# implement Sx-Sx spin correlation function
include("correlation_measurements/spin_x_correlation.jl")
export spin_x_correlation!

# implement Sy-Sy spin correlation function
include("correlation_measurements/spin_y_correlation.jl")
export spin_y_correlation!

# implement Sz-Sz spin correlation function
include("correlation_measurements/spin_z_correlation.jl")
export spin_z_correlation!

# implement pair correlation function
include("correlation_measurements/pair_correlation.jl")
export pair_correlation!

# impelement bond correlation function
include("correlation_measurements/bond_correlation.jl")
export bond_correlation!

end
