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

# Fields

- `SxSx::AbstractArray{C,D}`: Array the spin-``x`` correlation function ``\mathcal{S}_{x,\mathbf{r}}^{a,b}(\tau)`` is added to.
- `a::Int`: Index specifying an orbital species in the unit cell.
- `b::Int`: Index specifying an orbital species in the unit cell.
- `unit_cell::UnitCell{D}`: Defines unit cell.
- `lattice::Lattice{D}`: Specifies size of finite lattice.
- `Gup_τ0::AbstractMatrix{T}`: The matrix ``G_{\uparrow}(\tau,0).``
- `Gup_0τ::AbstractMatrix{T}`: The matrix ``G_{\uparrow}(0,\tau).``
- `Gdn_τ0::AbstractMatrix{T}`: The matrix ``G_{\downarrow}(\tau,0).``
- `Gdn_0τ::AbstractMatrix{T}`: The matrix ``G_{\downarrow}(0,\tau).``
- `sgn::T=one(T)`: The sign of the weight appearing in a DQMC simulation.
```
"""
function spin_x_correlation!(SxSx::AbstractArray{C,D}, a::Int, b::Int, unit_cell::UnitCell{D}, lattice::Lattice{D},
                             Gup_τ0::AbstractMatrix{T}, Gup_0τ::AbstractMatrix{T},
                             Gdn_τ0::AbstractMatrix{T}, Gdn_0τ::AbstractMatrix{T},
                             sgn::T=one(T)) where {D, C<:Number, T<:Number}

    # define zero unit cell displacement bonds for all combos of a and b orbitals
    z = @SVector zeros(Int,D)
    b_ab = Bond((b,a), z)::Bond{D} # displacement r_a - r_b
    b_ba = Bond((a,b), z)::Bond{D} # displacement r_b - r_a

    # SxSx(τ,r) = SxSx(τ,r) - 1/N sum_i G₊(b,i,β-τ|a,i+r,0)⋅G₋(a,i+r,τ|b,i,0)
    contract_G0r_Gr0!(SxSx, Gup_0τ, Gdn_τ0, b_ba, b_ab, -1, unit_cell, lattice, sgn)
    # SxSx(τ,r) = SxSx(τ,r) - 1/N sum_i G₋(b,i,β-τ|a,i+r,0)⋅G₊(a,i+r,τ|b,i,0)
    contract_G0r_Gr0!(SxSx, Gdn_0τ, Gup_τ0, b_ba, b_ab, -1, unit_cell, lattice, sgn)

    return nothing
end