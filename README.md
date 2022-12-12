# JDQMCMeasurements.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://SmoQySuite.github.io/JDQMCMeasurements.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://SmoQySuite.github.io/JDQMCMeasurements.jl/dev/)
[![Build Status](https://github.com/SmoQySuite/JDQMCMeasurements.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/SmoQySuite/JDQMCMeasurements.jl/actions/workflows/CI.yml?query=branch%3Amaster)
[![Coverage](https://codecov.io/gh/SmoQySuite/JDQMCMeasurements.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/SmoQySuite/JDQMCMeasurements.jl)

This package implements a variety of functions that can be called to measure various correlation functions in a
determinant quantum Monte Carlo (DQMC) simulation.
The exported correlation function measurements support arbitrary lattice geometries.
This package also exports several additional utility functions for transforming measurements from position space to momentum space,
and also measuring susceptibilities by integrating correlation functions over the imaginary time axis.

This package relies on the [`LatticeUtilities.jl`](https://github.com/cohensbw/LatticeUtilities.jl.git) to represent arbitary lattice geometries.

## Funding

The development of this code was supported by the U.S. Department of Energy, Office of Science, Basic Energy Sciences,
under Award Number DE-SC0022311.

## Installation
To install [`JDQMCMeasurements.jl`](https://github.com/SmoQySuite/JDQMCMeasurements.jl.git) run following in the Julia REPL:

```julia
] add JDQMCFramework
```