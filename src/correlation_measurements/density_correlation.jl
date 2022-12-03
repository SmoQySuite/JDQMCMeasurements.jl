@doc raw"""
    density_correlation!(DD::AbstractArray{C}, a::Int, b::Int,
                         unit_cell::UnitCell{D}, lattice::Lattice{D},
                         Gτ0up::AbstractArray{T,3}, Gτ0dn::AbstractArray{T,3},
                         Gττup::AbstractArray{T,3}, Gττdn::AbstractArray{T,3},
                         sgn::T=one(T)) where {D, C<:Complex, T<:Number}

Calculate the unequal time density-density (charge) correlation function
```math
\mathcal{D}_{\mathbf{r}}^{a,b}(\tau) = \frac{1}{N}\sum_{\mathbf{i}}\mathcal{D}_{\mathbf{i}+\mathbf{r},\mathbf{i}}^{a,b}(\tau,0)
= \frac{1}{N}\sum_{\mathbf{i}}\langle\hat{n}_{a,\mathbf{i}+\mathbf{r}}(\tau)\hat{n}_{b,\mathbf{i}}(0)\rangle,
```
where ``\hat{n}_{b,\mathbf{i}} = (\hat{n}_{\uparrow, b, \mathbf{i}} + \hat{n}_{\downarrow, b, \mathbf{i}})``
and ``\hat{n}_{\sigma, b,\mathbf{i}} = \hat{b}^\dagger_{\sigma, \mathbf{i}} \hat{b}_{\sigma, \mathbf{i}}``
is the number operator for an electron with spin ``\sigma`` on orbital ``b`` in unit cell ``\mathbf{i}``,
with the result being added to the array `DD`.

The arrays `Gτ0up` and `Gτ0dn` represent the unequal time Green's functions ``G_{\uparrow}(\tau,0)`` and ``G_{\downarrow}(\tau,0)`` respectively.
The arrays `Gττup` and `Gττdn` represent the equal time Green's functions ``G_{\uparrow}(\tau,\tau)`` and ``G_{\downarrow}(\tau,\tau)``
for all imaginary time slices ``\tau = \Delta\tau\cdot l`` respectively.
"""
function density_correlation!(DD::AbstractArray{C}, a::Int, b::Int,
                              unit_cell::UnitCell{D}, lattice::Lattice{D},
                              Gτ0up::AbstractArray{T,3}, Gτ0dn::AbstractArray{T,3},
                              Gττup::AbstractArray{T,3}, Gττdn::AbstractArray{T,3},
                              sgn::T=one(T)) where {D, C<:Complex, T<:Number}

    # length of imaginary time axis
    Lτ = size(DD,D+1) - 1

    # define zero unit cell displacement bonds for all combos of a and b orbitals
    zero_displacement = zeros(Int,D)
    b_aa = Bond((a,a), zero_displacement) # displacement r_a - r_a = 0
    b_bb = Bond((b,b), zero_displacement) # displacement r_b - r_b = 
    b_ab = Bond((b,a), zero_displacement) # displacement r_a - r_b
    b_ba = Bond((a,b), zero_displacement) # displacement r_b - r_a

    # get τ=0 Green's function matrices
    Gup_00 = @view Gττup[:,:,1] # G₊(0,0)
    Gdn_00 = @view Gττdn[:,:,1] # G₋(0,0)

    # iterate over imaginary time
    for l in 0:Lτ
        # get the density correlations for τ = Δτ⋅l
        DD_τ = selectdim(DD, ndims(DD), l+1)
        # get relevant Green's function matrices
        Gup_ττ = @view Gττup[:,:,l+1] # G₊(τ,τ)
        Gdn_ττ = @view Gττdn[:,:,l+1] # G₋(τ,τ)
        Gup_τ0 = @view Gτ0up[:,:,l+1] # G₊(τ,0)
        Gdn_τ0 = @view Gτ0dn[:,:,l+1] # G₋(τ,0)
        Gup_βmτ0 = @view Gτ0up[:,:,Lτ-l+1] # G₊(β-τ,0)
        Gdn_βmτ0 = @view Gτ0dn[:,:,Lτ-l+1] # G₋(β-τ,0)
        # DD(τ,r) = DD(τ,r) + 4
        @. DD_τ = DD_τ + 4 * sgn
        # DD(τ,r) = DD(τ,r) - 2/N sum_i G₊(a,i,τ|a,i,τ)
        contract_G00!(DD_τ, Gup_ττ, a, a, -2, unit_cell, lattice, sgn)
        # DD(τ,r) = DD(τ,r) - 2/N sum_i G₋(a,i,τ|a,i,τ)
        contract_G00!(DD_τ, Gdn_ττ, a, a, -2, unit_cell, lattice, sgn)
        # DD(τ,r) = DD(τ,r) - 2/N sum_i G₊(b,i,0|b,i,0)
        contract_G00!(DD_τ, Gup_00, b, b, -2, unit_cell, lattice, sgn)
        # DD(τ,r) = DD(τ,r) - 2/N sum_i G₋(b,i,0|b,i,0)
        contract_G00!(DD_τ, Gdn_00, b, b, -2, unit_cell, lattice, sgn)
        # DD(τ,r) = DD(τ,r) + 1/N sum_i G₊(a,i+r,τ|a,i+r,τ)⋅G₊(b,i,0|b,i,0)
        contract_Grr_G00!(DD_τ, Gup_ττ, Gup_00, b_aa, b_bb, 1, unit_cell, lattice, sgn)
        # DD(τ,r) = DD(τ,r) + 1/N sum_i G₋(a,i+r,τ|a,i+r,τ)⋅G₋(b,i,0|b,i,0)
        contract_Grr_G00!(DD_τ, Gdn_ττ, Gdn_00, b_aa, b_bb, 1, unit_cell, lattice, sgn)
        # DD(τ,r) = DD(τ,r) + 1/N sum_i G₊(a,i+r,τ|a,i+r,τ)⋅G₋(b,i,0|b,i,0)
        contract_Grr_G00!(DD_τ, Gup_ττ, Gdn_00, b_aa, b_bb, 1, unit_cell, lattice, sgn)
        # DD(τ,r) = DD(τ,r) + 1/N sum_i G₋(a,i+r,τ|a,i+r,τ)⋅G₊(b,i,0|b,i,0)
        contract_Grr_G00!(DD_τ, Gdn_ττ, Gup_00, b_aa, b_bb, 1, unit_cell, lattice, sgn)
        # DD(τ,r) = DD(τ,r) + 1/N sum_i G₊(b,i,β-τ|a,i+r,0)⋅G₊(a,i+r,τ|b,i,0)
        contract_G0r_Gr0!(DD_τ, Gup_βmτ0, Gup_τ0, b_ba, b_ab, 1, unit_cell, lattice, sgn)
        # DD(τ,r) = DD(τ,r) + 1/N sum_i G₋(b,i,β-τ|a,i+r,0)⋅G₋(a,i+r,τ|b,i,0)
        contract_G0r_Gr0!(DD_τ, Gdn_βmτ0, Gdn_τ0, b_ba, b_ab, 1, unit_cell, lattice, sgn)
    end
    
    return nothing
end

@doc raw"""
    density_correlation!(DD::AbstractArray{C}, a::Int, b::Int,
                         unit_cell::UnitCell{D}, lattice::Lattice{D},
                         Gup::AbstractMatrix{T}, Gdn::AbstractMatrix{T},
                         sgn::T=one(T)) where {D, C<:Complex, T<:Number}

Calculate the equaltime density-density (charge) correlation funciton
```math
\mathcal{D}_{\mathbf{r}}^{a,b} = \frac{1}{N}\sum_{\mathbf{i}}\mathcal{D}_{\mathbf{i}+\mathbf{r},\mathbf{i}}^{a,b}
= \frac{1}{N}\sum_{\mathbf{i}}\langle\hat{n}_{a,\mathbf{i}+\mathbf{r}}\hat{n}_{b,\mathbf{i}}\rangle,
```
with the result being added to the array `DD`.

The array `Gup` and `Gdn` are the eqaultime Green's functions ``G_{\uparrow}(0,0)`` and ``G_{\downarrow}(0,0)`` respectively.
"""
function density_correlation!(DD::AbstractArray{C}, a::Int, b::Int,
                              unit_cell::UnitCell{D}, lattice::Lattice{D},
                              Gup::AbstractMatrix{T}, Gdn::AbstractMatrix{T},
                              sgn::T=one(T)) where {D, C<:Complex, T<:Number}

    # define zero unit cell displacement bonds
    z = zeros(Int,D) # zero displacement
    b_aa = Bond((a,a), z) # displacement r_a - r_a = 0
    b_bb = Bond((b,b), z) # displacement r_b - r_b = 0
    b_ab = Bond((b,a), z) # displacement r_a - r_b
    b_ba = Bond((a,b), z) # displacement r_b - r_a

    # DD(r) = DD(r) + 4
    @. DD = DD + 4 * sgn
    # DD(r) = DD(r) - 2/N sum_i G₊(a,i|a,i)
    contract_G00!(DD, Gup, a, a, -2, unit_cell, lattice, sgn)
    # DD(r) = DD(r) - 2/N sum_i G₋(a,i|a,i)
    contract_G00!(DD, Gdn, a, a, -2, unit_cell, lattice, sgn)
    # DD(r) = DD(r) - 2/N sum_i G₊(b,i|b,i)
    contract_G00!(DD, Gup, b, b, -2, unit_cell, lattice, sgn)
    # DD(r) = DD(r) - 2/N sum_i G₋(b,i|b,i)
    contract_G00!(DD, Gdn, b, b, -2, unit_cell, lattice, sgn)
    # DD(r) = DD(r) + 1/N sum_i G₊(a,i+r|a,i+r)⋅G₊(b,i|b,i)
    contract_Grr_G00!(DD, Gup, Gup, b_aa, b_bb, 1, unit_cell, lattice, sgn)
    # DD(r) = DD(r) + 1/N sum_i G₋(a,i+r|a,i+r)⋅G₋(b,i|b,i)
    contract_Grr_G00!(DD, Gdn, Gdn, b_aa, b_bb, 1, unit_cell, lattice, sgn)
    # DD(r) = DD(r) + 1/N sum_i G₊(a,i+r|a,i+r)⋅G₋(b,i|b,i)
    contract_Grr_G00!(DD, Gup, Gdn, b_aa, b_bb, 1, unit_cell, lattice, sgn)
    # DD(r) = DD(r) + 1/N sum_i G₋(a,i+r|a,i+r)⋅G₊(b,i|b,i)
    contract_Grr_G00!(DD, Gdn, Gup, b_aa, b_bb, 1, unit_cell, lattice, sgn)
    # DD(r) = DD(r) - 1/N sum_i G₊(b,i|a,i+r)⋅G₊(a,i+r|b,i)
    contract_G0r_Gr0!(DD, Gup, Gup, b_ba, b_ab, -1, unit_cell, lattice, sgn)
    # DD(r) = DD(r) - 1/N sum_i G₋(b,i|a,i+r)⋅G₋(a,i+r|b,i)
    contract_G0r_Gr0!(DD, Gdn, Gdn, b_ba, b_ab, -1, unit_cell, lattice, sgn)
    # DD(r) = DD(r) + 1/N sum_i δ(r,0)⋅δ(a,b)⋅G₊(a,i+r|b,i)
    contract_δGr0!(DD, Gup, b_ab, b_ab, 1, unit_cell, lattice, sgn)
    # DD(r) = DD(r) + 1/N sum_i δ(r,0)⋅δ(a,b)⋅G₋(a,i+r|b,i)
    contract_δGr0!(DD, Gdn, b_ab, b_ab, 1, unit_cell, lattice, sgn)
    
    return nothing
end