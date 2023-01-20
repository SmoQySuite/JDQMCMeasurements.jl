@doc raw"""
    spin_x_correlation!(SxSx::AbstractArray{C}, a::Int, b::Int,
                        unit_cell::UnitCell{D}, lattice::Lattice{D},
                        Gτ0up::AbstractArray{T,3}, Gτ0dn::AbstractArray{T,3},
                        sgn::T=one(T)) where {D, C<:Complex, T<:Number}

Calculate the unequal-time spin-spin correlation function in the ``\hat{x}`` direction, given by
```math
\mathcal{S}_{x,\mathbf{r}}^{a,b}(\tau)=\frac{1}{N}\sum_{\mathbf{i}}\mathcal{S}_{x,\mathbf{i}+\mathbf{r},\mathbf{i}}^{ab}(\tau,0)
=\frac{1}{N}\sum_{\mathbf{i}}\big\langle\hat{S}_{x,a,\mathbf{i}+\mathbf{r}}(\tau)\hat{S}_{x,b,\mathbf{i}}(0)\big\rangle,
```
where the spin-``\hat{x}`` operator is given by
```math
\begin{align*}
\hat{S}_{x,\mathbf{i},a}= & (\hat{a}_{\uparrow,\mathbf{i}}^{\dagger},\hat{a}_{\downarrow,\mathbf{i}}^{\dagger})\left[\begin{array}{cc}
0 & 1\\
1 & 0
\end{array}\right]\left(\begin{array}{c}
\hat{a}_{\uparrow,\mathbf{i}}\\
\hat{a}_{\downarrow,\mathbf{i}}
\end{array}\right)\\
= & \hat{a}_{\uparrow,\mathbf{i}}^{\dagger}\hat{a}_{\downarrow,\mathbf{i}}+\hat{a}_{\downarrow,\mathbf{i}}^{\dagger}\hat{a}_{\uparrow,\mathbf{i}}.
\end{align*}
```
"""
function spin_x_correlation!(SxSx::AbstractArray{C}, a::Int, b::Int,
                             unit_cell::UnitCell{D}, lattice::Lattice{D},
                             Gτ0up::AbstractArray{T,3}, Gτ0dn::AbstractArray{T,3},
                             sgn::T=one(T)) where {D, C<:Complex, T<:Number}

    # length of imaginary time axis
    Lτ = size(SxSx,D+1) - 1

    # define zero unit cell displacement bonds for all combos of a and b orbitals
    zero_displacement = zeros(Int,D)
    b_ab = Bond((b,a), zero_displacement) # displacement r_a - r_b
    b_ba = Bond((a,b), zero_displacement) # displacement r_b - r_a

    # iterate over imagniary time
    for l in 0:Lτ
        # get spin x correlations for τ = Δτ⋅l
        SxSx_τ = selectdim(SxSx, ndims(SxSx), l+1)
        # get relevant Green's function matrices
        Gup_τ0 = @view Gτ0up[:,:,l+1] # G₊(τ,0)
        Gdn_τ0 = @view Gτ0dn[:,:,l+1] # G₋(τ,0)
        Gup_βmτ0 = @view Gτ0up[:,:,Lτ-l+1] # G₊(β-τ,0)
        Gdn_βmτ0 = @view Gτ0dn[:,:,Lτ-l+1] # G₋(β-τ,0)
        # SxSx(τ,r) = SxSx(τ,r) + 1/N sum_i G₊(b,i,β-τ|a,i+r,0)⋅G₋(a,i+r,τ|b,i,0)
        contract_G0r_Gr0!(SxSx_τ, Gup_βmτ0, Gdn_τ0, b_ba, b_ab, 1, unit_cell, lattice, sgn)
        # SxSx(τ,r) = SxSx(τ,r) + 1/N sum_i G₋(b,i,β-τ|a,i+r,0)⋅G₊(a,i+r,τ|b,i,0)
        contract_G0r_Gr0!(SxSx_τ, Gdn_βmτ0, Gup_τ0, b_ba, b_ab, 1, unit_cell, lattice, sgn)
    end

    return nothing
end

@doc raw"""
    spin_x_correlation!(SxSx::AbstractArray{C}, a::Int, b::Int,
                        unit_cell::UnitCell{D}, lattice::Lattice{D},
                        Gup::AbstractMatrix{T}, Gdn::AbstractMatrix{T},
                        sgn::T=one(T)) where {D, C<:Complex, T<:Number}

Calculate the equal-time spin-spin correlation function in the ``\hat{x}`` direction, given by
```math
\mathcal{S}_{x,\mathbf{r}}^{a,b}=\frac{1}{N}\sum_{\mathbf{i}}\mathcal{S}_{x,\mathbf{i}+\mathbf{r},\mathbf{i}}^{ab}
=\frac{1}{N}\sum_{\mathbf{i}}\big\langle\hat{S}_{x,a,\mathbf{i}+\mathbf{r}}\hat{S}_{x,b,\mathbf{i}}\big\rangle.
```
"""
function spin_x_correlation!(SxSx::AbstractArray{C}, a::Int, b::Int,
                             unit_cell::UnitCell{D}, lattice::Lattice{D},
                             Gup::AbstractMatrix{T}, Gdn::AbstractMatrix{T},
                             sgn::T=one(T)) where {D, C<:Complex, T<:Number}

    # define zero unit cell displacement bonds
    z = zeros(Int,D) # zero displacement
    b_ab = Bond((b,a), z) # displacement r_a - r_b
    b_ba = Bond((a,b), z) # displacement r_b - r_a

    # SxSx(r) = SxSx(r) - 1/N sum_i G₊(b,i|a,i+r)⋅G₋(a,i+r|b,i)
    contract_G0r_Gr0!(SxSx, Gup, Gdn, b_ba, b_ab, -1, unit_cell, lattice, sgn)
    # SxSx(r) = SxSx(r) - 1/N sum_i G₋(b,i|a,i+r)⋅G₊(a,i+r|b,i)
    contract_G0r_Gr0!(SxSx, Gdn, Gup, b_ba, b_ab, -1, unit_cell, lattice, sgn)
    # SxSx(r) = SxSx(r) + 1/N sum_i δ(a,b)⋅δ(r,0)⋅G₊(a,i+r|b,i)
    contract_δGr0!(SxSx, Gup, b_ab, b_ab, 1, unit_cell, lattice, sgn)
    # SxSx(r) = SxSx(r) + 1/N sum_i δ(a,b)⋅δ(r,0)⋅G₋(a,i+r|b,i)
    contract_δGr0!(SxSx, Gdn, b_ab, b_ab, 1, unit_cell, lattice, sgn)

    return nothing
end