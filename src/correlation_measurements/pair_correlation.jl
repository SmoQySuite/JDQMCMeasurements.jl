@doc raw"""
    pair_correlation!(ΔΔᵀ::AbstractArray{C}, b″::Bond{D}, b′::Bond{D}, unit_cell::UnitCell{D}, lattice::Lattice{D},
                      Gτ0up::AbstractArray{T,3}, Gτ0dn::AbstractArray{T,3}, sgn::T=one(T)) where {D, C<:Complex, T<:Number}

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
"""
function pair_correlation!(ΔΔᵀ::AbstractArray{C}, b″::Bond{D}, b′::Bond{D}, unit_cell::UnitCell{D}, lattice::Lattice{D},
                           Gτ0up::AbstractArray{T,3}, Gτ0dn::AbstractArray{T,3}, sgn::T=one(T)) where {D, C<:Complex, T<:Number}

    # length of imaginary time axis
    Lτ = size(ΔΔᵀ,D+1) - 1

    # iterate over imaginary time
    for l in 0:Lτ
        # get the density correlations for τ = Δτ⋅l
        ΔΔᵀ_τ = selectdim(ΔΔᵀ, ndims(ΔΔᵀ), l+1)
        # get relevant Green's function matrices
        Gup_τ0 = @view Gτ0up[:,:,l+1] # G₊(τ,0)
        Gdn_τ0 = @view Gτ0dn[:,:,l+1] # G₋(τ,0)
        # ΔΔᵀ(τ,r) = G₊(a,i+r+r″,τ|c,i+r′,0)⋅G₋(b,i+r,τ|d,i,0)
        contract_Gr0_Gr0!(ΔΔᵀ_τ, Gup_τ0, Gdn_τ0, b″, b′, 1, unit_cell, lattice, sgn)
    end

    return nothing
end

@doc raw"""
    pair_correlation!(ΔΔᵀ::AbstractArray{C}, b″::Bond{D}, b′::Bond{D}, unit_cell::UnitCell{D}, lattice::Lattice{D},
                      Gup::AbstractMatrix{T}, Gdn::AbstractMatrix{T}, sgn::T=one(T)) where {C<:Complex, T<:Number}

Calculate the equal-time pair correlation function
```math
\mathcal{P}_{\mathbf{r}}^{(a,b,r''),(c,d,r')}=\frac{1}{N}\sum_{\mathbf{i}}\mathcal{P}_{\mathbf{i}+\mathbf{r},\mathbf{i}}^{(a,b,r''),(c,d,r')}
=\frac{1}{N}\sum_{\mathbf{i}}\langle\hat{\Delta}_{\mathbf{i}+\mathbf{r},a,b,\mathbf{r}''}\hat{\Delta}_{\mathbf{i},c,d,\mathbf{r}'}^{\dagger}\rangle,
```
where the bond `b″` defines the pair creation operator
```math
\hat{\Delta}_{\mathbf{i},a,b,\mathbf{r}''}^{\dagger}=\hat{a}_{\uparrow,\mathbf{i}+\mathbf{r}''}^{\dagger}\hat{b}_{\downarrow,\mathbf{i}}^{\dagger},
```
and the bond  `b′` defines the pair creation operator
```math
\hat{\Delta}_{\mathbf{i},c,d,\mathbf{r}'}^{\dagger}=\hat{c}_{\uparrow,\mathbf{i}+\mathbf{r}'}^{\dagger}\hat{d}_{\downarrow,\mathbf{i}}^{\dagger}.
```
"""
function pair_correlation!(ΔΔᵀ::AbstractArray{C}, b″::Bond{D}, b′::Bond{D}, unit_cell::UnitCell{D}, lattice::Lattice{D},
                           Gup::AbstractMatrix{T}, Gdn::AbstractMatrix{T}, sgn::T=one(T)) where {D, C<:Complex, T<:Number}
    
    # ΔΔᵀ(r) = G₊(a,i+r+r″|c,i+r′)⋅G₋(b,i+r|d,i)
    contract_Gr0_Gr0!(ΔΔᵀ, Gup, Gdn, b″, b′, 1, unit_cell, lattice, sgn)

    return nothing
end