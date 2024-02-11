@doc raw"""
    spin_z_correlation!(
        SzSz::AbstractArray{C,D},
        a::Int, b::Int, unit_cell::UnitCell{D}, lattice::Lattice{D},
        Gup_τ0::AbstractMatrix{T}, Gup_0τ::AbstractMatrix{T},
        Gup_ττ::AbstractMatrix{T}, Gup_00::AbstractMatrix{T},
        Gdn_τ0::AbstractMatrix{T}, Gdn_0τ::AbstractMatrix{T},
        Gdn_ττ::AbstractMatrix{T}, Gdn_00::AbstractMatrix{T},
        sgn::T=one(T)
    ) where {D, C<:Complex, T<:Number}

Calculate the unequal-time spin-spin correlation function in the ``\hat{z}`` direction, given by
```math
\begin{align*}
\mathcal{S}_{z,\mathbf{r}}^{a,b}(\tau)=\frac{1}{N}\sum_{\mathbf{i}}\mathcal{S}_{z,\mathbf{i}+\mathbf{r},\mathbf{i}}^{ab}(\tau,0) 
    = & \frac{1}{N}\sum_{\mathbf{i}}\big\langle\hat{S}_{z,a,\mathbf{i}+\mathbf{r}}(\tau)\hat{S}_{z,b,\mathbf{i}}(0)\big\rangle,
\end{align*}
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

# Fields

- `SzSz::AbstractArray{C,D}`: Array the spin-``z`` correlation function ``\mathcal{S}_{z,\mathbf{r}}^{a,b}(\tau)`` is added to.
- `a::Int`: Index specifying an orbital species in the unit cell.
- `b::Int`: Index specifying an orbital species in the unit cell.
- `unit_cell::UnitCell{D}`: Defines unit cell.
- `lattice::Lattice{D}`: Specifies size of finite lattice.
- `Gup_τ0::AbstractMatrix{T}`: The matrix ``G_{\uparrow}(\tau,0).``
- `Gup_0τ::AbstractMatrix{T}`: The matrix ``G_{\uparrow}(0,\tau).``
- `Gup_ττ::AbstractMatrix{T}`: The matrix ``G_{\uparrow}(\tau,\tau).``
- `Gup_00::AbstractMatrix{T}`: The matrix ``G_{\uparrow}(0,0).``
- `Gdn_τ0::AbstractMatrix{T}`: The matrix ``G_{\downarrow}(\tau,0).``
- `Gdn_0τ::AbstractMatrix{T}`: The matrix ``G_{\downarrow}(0,\tau).``
- `Gdn_ττ::AbstractMatrix{T}`: The matrix ``G_{\downarrow}(\tau,\tau).``
- `Gdn_00::AbstractMatrix{T}`: The matrix ``G_{\downarrow}(0,0).``
- `sgn::T=one(T)`: The sign of the weight appearing in a DQMC simulation.
"""
function spin_z_correlation!(
    SzSz::AbstractArray{C,D},
    a::Int, b::Int, unit_cell::UnitCell{D}, lattice::Lattice{D},
    Gup_τ0::AbstractMatrix{T}, Gup_0τ::AbstractMatrix{T},
    Gup_ττ::AbstractMatrix{T}, Gup_00::AbstractMatrix{T},
    Gdn_τ0::AbstractMatrix{T}, Gdn_0τ::AbstractMatrix{T},
    Gdn_ττ::AbstractMatrix{T}, Gdn_00::AbstractMatrix{T},
    sgn::T=one(T)
) where {D, C<:Complex, T<:Number}

    # define zero unit cell displacement bonds for all combos of a and b orbitals
    z = @SVector zeros(Int,D)
    b_aa = Bond((a,a), z)::Bond{D} # displacement r_a - r_a = 0
    b_bb = Bond((b,b), z)::Bond{D} # displacement r_b - r_b = 0
    b_ab = Bond((b,a), z)::Bond{D} # displacement r_a - r_b
    b_ba = Bond((a,b), z)::Bond{D} # displacement r_b - r_a

    # SzSz(τ,r) = SzSz(τ,r) + 1/N sum_i G₊(a,i+r,τ|a,i+r,τ)⋅G₊(b,i,0|b,i,0)
    contract_Grr_G00!(SzSz, Gup_ττ, Gup_00, b_aa, b_bb, 1, unit_cell, lattice, sgn)
    # SzSz(τ,r) = SzSz(τ,r) + 1/N sum_i G₋(a,i+r,τ|a,i+r,τ)⋅G₋(b,i,0|b,i,0)
    contract_Grr_G00!(SzSz, Gdn_ττ, Gdn_00, b_aa, b_bb, 1, unit_cell, lattice, sgn)
    
    # SzSz(τ,r) = SzSz(τ,r) - 1/N sum_i G₊(a,i+r,τ|a,i+r,τ)⋅G₋(b,i,0|b,i,0)
    contract_Grr_G00!(SzSz, Gup_ττ, Gdn_00, b_aa, b_bb, -1, unit_cell, lattice, sgn)
    # SzSz(τ,r) = SzSz(τ,r) - 1/N sum_i G₋(a,i+r,τ|a,i+r,τ)⋅G₊(b,i,0|b,i,0)
    contract_Grr_G00!(SzSz, Gdn_ττ, Gup_00, b_aa, b_bb, -1, unit_cell, lattice, sgn)

    # SzSz(τ,r) = SzSz(τ,r) + 1/N sum_i G₊(b,i,0|a,i+r,τ)⋅G₊(a,i+r,τ|b,i,0)
    contract_G0r_Gr0!(SzSz, Gup_0τ, Gup_τ0, b_ba, b_ab, -1, unit_cell, lattice, sgn)
    # SzSz(τ,r) = SzSz(τ,r) + 1/N sum_i G₋(b,i,0|a,i+r,τ)⋅G₋(a,i+r,τ|b,i,0)
    contract_G0r_Gr0!(SzSz, Gdn_0τ, Gdn_τ0, b_ba, b_ab, -1, unit_cell, lattice, sgn)

    return nothing
end