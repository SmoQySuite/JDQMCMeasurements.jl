@doc raw"""
    pair_correlation!(P::AbstractArray{C,D}, b″::Bond{D}, b′::Bond{D}, unit_cell::UnitCell{D}, lattice::Lattice{D},
                      Gup_τ0::AbstractMatrix{T}, Gdn_τ0::AbstractMatrix{T}, sgn::T=one(T)) where {D, C<:Number, T<:Number}

Calculate the unequal-time pair correlation function
```math
\mathcal{P}_{\mathbf{r}}^{(a,b,r''),(c,d,r')}(\tau)=\frac{1}{N}\sum_{\mathbf{i}}\mathcal{P}_{\mathbf{i}+\mathbf{r},\mathbf{i}}^{(a,b,r''),(c,d,r')}(\tau,0)
=\frac{1}{N}\sum_{\mathbf{i}}\langle\hat{\Delta}_{\mathbf{i}+\mathbf{r},a,b,\mathbf{r}''}(\tau)\hat{\Delta}_{\mathbf{i},c,d,\mathbf{r}'}^{\dagger}(0)\rangle,
```
where the bond `b″` defines the pair creation operator
```math
\hat{\Delta}_{\mathbf{i},a,b,\mathbf{r}''}^{\dagger}=\hat{a}_{\uparrow,\mathbf{i}+\mathbf{r}''}^{\dagger}\hat{b}_{\downarrow,\mathbf{i}}^{\dagger},
```
and the bond  `b′` defines the pair creation operator
```math
\hat{\Delta}_{\mathbf{i},c,d,\mathbf{r}'}^{\dagger}=\hat{c}_{\uparrow,\mathbf{i}+\mathbf{r}'}^{\dagger}\hat{d}_{\downarrow,\mathbf{i}}^{\dagger}.
```

# Fields

- `P::AbstractArray{C,D}`: Array the pair correlation function ``\mathcal{P}_{\mathbf{r}}^{(a,b,r''),(c,d,r')}(\tau)`` is added to.
- `b″::Bond{D}`: Bond defining pair annihilation operator appearing in pair correlation function.
- `b′::Bond{D}`: Bond defining pair creation operator appearing in pair correlation function.
- `unit_cell::UnitCell{D}`: Defines unit cell.
- `lattice::Lattice{D}`: Specifies size of finite lattice.
- `Gup_τ0::AbstractMatrix{T}`: The matrix ``G_{\uparrow}(\tau,0).``
- `Gdn_τ0::AbstractMatrix{T}`: The matrix ``G_{\downarrow}(\tau,0).``
- `sgn::T=one(T)`: The sign of the weight appearing in a DQMC simulation.
"""
function pair_correlation!(P::AbstractArray{C,D}, b″::Bond{D}, b′::Bond{D}, unit_cell::UnitCell{D}, lattice::Lattice{D},
                           Gup_τ0::AbstractMatrix{T}, Gdn_τ0::AbstractMatrix{T}, sgn::T=one(T)) where {D, C<:Number, T<:Number}

    # P(τ,r) = G₊(a,i+r+r″,τ|c,i+r′,0)⋅G₋(b,i+r,τ|d,i,0)
    contract_Gr0_Gr0!(P, Gup_τ0, Gdn_τ0, b″, b′, 1, unit_cell, lattice, sgn)

    return nothing
end