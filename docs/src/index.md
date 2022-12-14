```@meta
CurrentModule = JDQMCMeasurements
```

# JDQMCMeasurements

Documentation for [JDQMCMeasurements](https://github.com/SmoQySuite/JDQMCMeasurements.jl).

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

## Notation

The operators ``\{\hat{a}_{\sigma,\mathbf{i}}^{\dagger},\hat{b}_{\sigma,\mathbf{i}}^{\dagger},\hat{c}_{\sigma,\mathbf{i}}^{\dagger},\hat{d}_{\sigma,\mathbf{i}}^{\dagger}\}\,\big(\{\hat{a}_{\sigma,\mathbf{i}},\hat{b}_{\sigma,\mathbf{i}},\hat{c}_{\sigma,\mathbf{i}},\hat{d}_{\sigma,\mathbf{i}}\}\big)``
are fermion creation (annihilation) operators for an electron with
spin ``\sigma=\{\uparrow,\downarrow\}``. The letter of the operator
is treated as a variable denoting a speicific orbital in the unit
cell. Bolded variables absent a subscript, like ``\mathbf{i}``, denote
a displacement in unit cells. As an example, the term
```math
\hat{a}_{\uparrow,\mathbf{i}+\mathbf{r}}^{\dagger}\hat{b}_{\uparrow,\mathbf{i}}
```
describes a spin up electron being annihilated from orital ``b`` in the
unit cell located at ``\mathbf{i}``, and a spin up electron being created
in orbital ``a`` in the unit cell located at ``\mathbf{i}+\mathbf{r}``.
The locations of orbitals ``a`` and ``b`` in the lattice are ``\mathbf{i}+\mathbf{r}+\mathbf{r}_{a}``
and ``\mathbf{i}+\mathbf{r}_{b}`` respectively, with the corresponding
displacement vector between the two orbitals given by ``\mathbf{r}+(\mathbf{r}_{a}-\mathbf{r}_{b})``.
Here ``\mathbf{r}_{a}`` and ``\mathbf{r}_{b}`` are basis vectors describing
the relative location of the ``a`` and ``b`` orbitals within the unit
cell. In the docs we assume a finite lattice of ``N`` unit cells with periodic boundary conditions.

## Green's Function Definition

The single-particle electron Green's function is given by
```math
\begin{align*}
G_{\sigma,\mathbf{i}+\mathbf{r},\mathbf{i}}^{a,b}(\tau,0)= & \langle\hat{\mathcal{T}}\hat{a}_{\sigma,\mathbf{i}+\mathbf{r}}(\tau)\hat{b}_{\sigma,\mathbf{i}}^{\dagger}(0)\rangle\\
= & \begin{cases}
\langle\hat{a}_{\sigma,\mathbf{i}+\mathbf{r}}(\tau)\hat{b}_{\sigma,\mathbf{i}}^{\dagger}(0)\rangle & \beta>\tau\ge0\\
-\langle\hat{b}_{\sigma,\mathbf{i}}^{\dagger}(0)\hat{a}_{\sigma,\mathbf{i}+\mathbf{r}}(\tau)\rangle & -\beta\le\tau<0,
\end{cases}
\end{align*}
```
subject to the aperiodic boundary condition
```math
\begin{align*}
G_{\sigma,\mathbf{i}+\mathbf{r},\mathbf{i}}^{a,b}(\tau+\beta,0)= & -G_{\sigma,\mathbf{i}+\mathbf{r},\mathbf{i}}^{a,b}(\tau,0),
\end{align*}
```
where ``\hat{\mathcal{T}}`` is the time ordering operator in imaginary time.

## Comment on Representing Arbitrary Lattice Geometries

This package uses the [`LatticeUtilities.jl`](https://github.com/cohensbw/LatticeUtilities.jl.git) to represent arbitary lattice geometries.
The [`LatticeUtilities.jl`](https://github.com/cohensbw/LatticeUtilities.jl.git) exports three types that are used
in this package. The `UnitCell` type describes the unit cell, the `Lattice` type describes the size of the
finite periodic lattice in question, and the `Bond` type defines a bond/static displacement in the lattice.

With regard to the `Bond` type in particular, a displacement from orbital ``b`` to orbital ``a`` displaced
``\mathbf{r}`` unit cells away, with the corresponding static displacement vector given by
``\mathbf{r} + (\mathbf{r}_a-\mathbf{r}_b),`` should be defined as follows:
```julia
b = 2
a = 1
r = [1,0,0]
bond = Bond((b,a),r)
```
In particular, note the ordering of the tuple when declaring `bond` above, the tuple `(b,a)` indicates a bond going
from orbital ``b`` to orbital ``a.``
Refer to the [`LatticeUtilities.jl`](https://github.com/cohensbw/LatticeUtilities.jl.git) documentation for more information.