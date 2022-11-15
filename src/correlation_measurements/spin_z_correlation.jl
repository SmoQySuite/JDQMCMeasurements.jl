@doc raw"""
    spin_z_correlation!(SzSz::AbstractArray{C}, a::Int, b::Int,
                        unit_cell::UnitCell, lattice::Lattice,
                        Gτ0up::AbstractArray{T,3}, Gτ0dn::AbstractArray{T,3},
                        Gττup::AbstractArray{T,3}, Gττdn::AbstractArray{T,3}) where {C<:Complex, T<:Number}

Calculate the unequal-time spin-spin correlation function in the ``\hat{z}`` direction, given by
```math
\mathcal{S}_{z,\mathbf{r}}^{a,b}(\tau)=\frac{1}{N}\sum_{\mathbf{i}}\mathcal{S}_{z,\mathbf{i}+\mathbf{r},\mathbf{i}}^{ab}(\tau,0)
=\frac{1}{N}\sum_{\mathbf{i}}\big\langle\hat{S}_{z,a,\mathbf{i}+\mathbf{r}}(\tau)\hat{S}_{z,b,\mathbf{i}}(0)\big\rangle,
```
where the spin-``\hat{z}`` operator is given by
```math
\begin{align*}
\hat{S}_{z,a,\mathbf{i}}= & (\hat{a}_{\uparrow,\mathbf{i}}^{\dagger},\hat{a}_{\downarrow,\mathbf{i}}^{\dagger})\left[\begin{array}{cc}
1 & 0\\
0 & -1
\end{array}\right]\left(\begin{array}{c}
\hat{a}_{\uparrow,\mathbf{i}}\\
\hat{a}_{\downarrow,\mathbf{i}}
\end{array}\right)\\
= & \hat{n}_{\uparrow,a,\mathbf{i}}-\hat{n}_{\downarrow,a,\mathbf{i}}.
\end{align*}
```
"""
function spin_z_correlation!(SzSz::AbstractArray{C}, a::Int, b::Int,
                             unit_cell::UnitCell, lattice::Lattice,
                             Gτ0up::AbstractArray{T,3}, Gτ0dn::AbstractArray{T,3},
                             Gττup::AbstractArray{T,3}, Gττdn::AbstractArray{T,3}) where {C<:Complex, T<:Number}

    # get dimension of system
    D = unit_cell.D

    # length of imaginary time axis
    Lτ = size(SzSz,D+1) - 1

    # define zero unit cell displacement bonds for all combos of a and b orbitals
    zero_displacement = zeros(Int,D)
    b_aa = Bond((a,a), zero_displacement) # displacement r_a - r_a = 0
    b_bb = Bond((b,b), zero_displacement) # displacement r_b - r_b = 
    b_ab = Bond((b,a), zero_displacement) # displacement r_a - r_b
    b_ba = Bond((a,b), zero_displacement) # displacement r_b - r_a

    # get τ=0 Green's function matrices
    Gup_00 = @view Gττup[:,:,1] # G₊(0,0)
    Gdn_00 = @view Gττdn[:,:,1] # G₋(0,0)

    # iterate over imagniary time
    for l in 0:Lτ
        # get spin z correlations for τ = Δτ⋅l
        SzSz_τ = selectdim(SzSz, ndims(SzSz), l+1)
        # get relevant Green's function matrices
        Gup_ττ = @view Gττup[:,:,l+1] # G₊(τ,τ)
        Gdn_ττ = @view Gττdn[:,:,l+1] # G₋(τ,τ)
        Gup_τ0 = @view Gτ0up[:,:,l+1] # G₊(τ,0)
        Gdn_τ0 = @view Gτ0dn[:,:,l+1] # G₋(τ,0)
        Gup_βmτ0 = @view Gτ0up[:,:,Lτ-l+1] # G₊(β-τ,0)
        Gdn_βmτ0 = @view Gτ0dn[:,:,Lτ-l+1] # G₋(β-τ,0)
        # SzSz(τ,r) = SzSz(τ,r) + 1/N sum_i G₊(a,i+r,τ|a,i+r,τ)⋅G₊(b,i,0|b,i,0)
        contract_Grr_G00!(SzSz_τ, Gup_ττ, Gup_00, b_aa, b_bb, 1, unit_cell, lattice)
        # SzSz(τ,r) = SzSz(τ,r) + 1/N sum_i G₋(a,i+r,τ|a,i+r,τ)⋅G₋(b,i,0|b,i,0)
        contract_Grr_G00!(SzSz_τ, Gdn_ττ, Gdn_00, b_aa, b_bb, 1, unit_cell, lattice)
        # SzSz(τ,r) = SzSz(τ,r) - 1/N sum_i G₊(a,i+r,τ|a,i+r,τ)⋅G₋(b,i,0|b,i,0)
        contract_Grr_G00!(SzSz_τ, Gup_ττ, Gdn_00, b_aa, b_bb, -1, unit_cell, lattice)
        # SzSz(τ,r) = SzSz(τ,r) - 1/N sum_i G₋(a,i+r,τ|a,i+r,τ)⋅G₊(b,i,0|b,i,0)
        contract_Grr_G00!(SzSz_τ, Gdn_ττ, Gup_00, b_aa, b_bb, -1, unit_cell, lattice)
        # SzSz(τ,r) = SzSz(τ,r) + 1/N sum_i G₊(b,i,β-τ|a,i+r,0)⋅G₊(a,i+r,τ|b,i,0)
        contract_G0r_Gr0!(SzSz_τ, Gup_βmτ0, Gup_τ0, b_ba, b_ab, 1, unit_cell, lattice)
        # SzSz(τ,r) = SzSz(τ,r) + 1/N sum_i G₋(b,i,β-τ|a,i+r,0)⋅G₋(a,i+r,τ|b,i,0)
        contract_G0r_Gr0!(SzSz_τ, Gdn_βmτ0, Gdn_τ0, b_ba, b_ab, 1, unit_cell, lattice)
    end

    return nothing
end

@doc raw"""
    spin_z_correlation!(SzSz::AbstractArray{C}, a::Int, b::Int,
                        unit_cell::UnitCell, lattice::Lattice,
                        Gup::AbstractMatrix{T}, Gdn::AbstractMatrix{T}) where {C<:Complex, T<:Number}

Calculate the equal-time spin-spin correlation function in the ``\hat{z}`` direction, given by
```math
\mathcal{S}_{z,\mathbf{r}}^{a,b}=\frac{1}{N}\sum_{\mathbf{i}}\mathcal{S}_{z,\mathbf{i}+\mathbf{r},\mathbf{i}}^{ab}
=\frac{1}{N}\sum_{\mathbf{i}}\big\langle\hat{S}_{z,a,\mathbf{i}+\mathbf{r}}\hat{S}_{z,b,\mathbf{i}}\big\rangle.
```
"""
function spin_z_correlation!(SzSz::AbstractArray{C}, a::Int, b::Int,
                             unit_cell::UnitCell, lattice::Lattice,
                             Gup::AbstractMatrix{T}, Gdn::AbstractMatrix{T}) where {C<:Complex, T<:Number}

    # get dimension of system
    D = unit_cell.D

    # define zero unit cell displacement bonds
    z = zeros(Int,D) # zero displacement
    b_aa = Bond((a,a), z) # displacement r_a - r_a = 0
    b_bb = Bond((b,b), z) # displacement r_b - r_b = 0
    b_ab = Bond((b,a), z) # displacement r_a - r_b
    b_ba = Bond((a,b), z) # displacement r_b - r_a

    # SzSz(r) = SzSz(r) + 1/N sum_i G₊(a,i+r|a,i+r)⋅G₊(b,i|b,i)
    contract_Grr_G00!(SzSz, Gup, Gup, b_aa, b_bb, 1, unit_cell, lattice)
    # SzSz(r) = SzSz(r) + 1/N sum_i G₋(a,i+r|a,i+r)⋅G₋(b,i|b,i)
    contract_Grr_G00!(SzSz, Gdn, Gdn, b_aa, b_bb, 1, unit_cell, lattice)
    # SzSz(r) = SzSz(r) - 1/N sum_i G₊(a,i+r|a,i+r)⋅G₋(b,i|b,i)
    contract_Grr_G00!(SzSz, Gup, Gdn, b_aa, b_bb, -1, unit_cell, lattice)
    # SzSz(r) = SzSz(r) - 1/N sum_i G₋(a,i+r|a,i+r)⋅G₊(b,i|b,i)
    contract_Grr_G00!(SzSz, Gdn, Gup, b_aa, b_bb, -1, unit_cell, lattice)
    # SzSz(r) = SzSz(r) - 1/N sum_i G₊(b,i|a,i+r)⋅G₊(a,i+r|b,i)
    contract_G0r_Gr0!(SzSz, Gup, Gup, b_ba, b_ab, -1, unit_cell, lattice)
    # SzSz(r) = SzSz(r) - 1/N sum_i G₋(b,i|a,i+r)⋅G₋(a,i+r|b,i)
    contract_G0r_Gr0!(SzSz, Gdn, Gdn, b_ba, b_ab, -1, unit_cell, lattice)
    # SzSz(r) = SzSz(r) + 1/N sum_i δ(a,b)⋅δ(r,0)⋅G₊(a,i+r|b,i)
    contract_δGr0!(SzSz, Gup, b_ab, b_ab, 1, unit_cell, lattice)
    # SzSz(r) = SzSz(r) + 1/N sum_i δ(a,b)⋅δ(r,0)⋅G₋(a,i+r|b,i)
    contract_δGr0!(SzSz, Gdn, b_ab, b_ab, 1, unit_cell, lattice)

    return nothing
end