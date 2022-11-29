@doc raw"""
    measure_greens!(G::AbstractArray{C}, a::Int, b::Int, 
                    unit_cell::UnitCell, lattice::Lattice,
                    Gτ0::AbstractArray{T,3}, sgn::T=one(T)) where {C<:Complex, T<:Number}

Measure the unequal time Green's function averaged over translation symmetry
```math
G_{\sigma,\mathbf{r}}^{a,b}(\tau)=\frac{1}{N}\sum_{\mathbf{i}}G_{\sigma,\mathbf{i}+\mathbf{r},\mathbf{i}}^{a,b}(\tau,0)
=\frac{1}{N}\sum_{\mathbf{i}}\langle\hat{\mathcal{T}}\hat{a}_{\sigma,\mathbf{i}+\mathbf{r}}^{\phantom{\dagger}}(\tau)\hat{b}_{\sigma,\mathbf{i}}^{\dagger}(0)\rangle,
```
with the result being added to `G`.
"""
function greens!(G::AbstractArray{C}, a::Int, b::Int, 
                 unit_cell::UnitCell, lattice::Lattice,
                 Gτ0::AbstractArray{T,3}, sgn::T=one(T)) where {C<:Complex, T<:Number}

    # get dimension of system
    D = unit_cell.D

    # length of imaginary time axis
    Lτ = size(G,D+1) - 1

    # construct the relevant bond definition
    d = Bond((b,a), zeros(Int, D))

    # iterate over imaginary time
    for l in 0:Lτ
        # get green's function for τ=Δτ⋅(l-1)
        G_τ = selectdim(G, ndims(G), l+1)
        # get G(τ,0)
        Gτ0_τ = @view Gτ0[:,:,l+1]
        # average green's function over translation symmetry
        contract_Gr0!(G_τ, Gτ0_τ, d, 1, unit_cell, lattice, sgn)
    end

    return nothing
end

@doc raw"""
    greens!(G::AbstractArray{C}, a::Int, b::Int, 
            unit_cell::UnitCell, lattice::Lattice,
            G00::AbstractMatrix{T}, sgn::T=one(T)) where {C<:Complex, T<:Number}

Measure the equal time Green's function averaged over translation symmetry
```math
G_{\sigma,\mathbf{r}}^{a,b}=\frac{1}{N}\sum_{\mathbf{i}}G_{\sigma,\mathbf{i}+\mathbf{r},\mathbf{i}}^{a,b}(0,0)
=\frac{1}{N}\sum_{\mathbf{i}}\langle\hat{a}_{\sigma,\mathbf{i}+\mathbf{r}}^{\phantom{\dagger}}\hat{b}_{\sigma,\mathbf{i}}^{\dagger}\rangle,
```
with the result being added to `G`.
"""
function greens!(G::AbstractArray{C}, a::Int, b::Int, 
                 unit_cell::UnitCell, lattice::Lattice,
                 G00::AbstractMatrix{T}, sgn::T=one(T)) where {C<:Complex, T<:Number}

    # get dimension of system
    D = unit_cell.D
    
    # construct the relevant bond definition
    d = Bond((b,a), zeros(Int, D))

    # average green's function over translation symmetry
    contract_Gr0!(G, G00, d, 1, unit_cell, lattice, sgn)

    return nothing
end