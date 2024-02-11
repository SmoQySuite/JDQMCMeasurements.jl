@doc raw"""
    bond_correlation!(
        BB::AbstractArray{C,D},
        b′::Bond{D}, b″::Bond{D}, unit_cell::UnitCell{D}, lattice::Lattice{D},
        Gup_τ0::AbstractMatrix{T}, Gup_0τ::AbstractMatrix{T},
        Gup_ττ::AbstractMatrix{T}, Gup_00::AbstractMatrix{T},
        Gdn_τ0::AbstractMatrix{T}, Gdn_0τ::AbstractMatrix{T},
        Gdn_ττ::AbstractMatrix{T}, Gdn_00::AbstractMatrix{T},
        sgn::T=one(T)
    ) where {D, C<:Number, T<:Number}

Calculate the uneqaul-time bond-bond correlation function
```math
\begin{align*}
\mathcal{B}_{\mathbf{r}}^{(\mathbf{r}',a,b),(\mathbf{r}'',c,d)}(\tau) =
    & \frac{1}{N}\sum_{\mathbf{i}} \langle[\hat{B}_{\uparrow,\mathbf{i}+\mathbf{r},(\mathbf{r}',a,b)}(\tau)+\hat{B}_{\downarrow,\mathbf{i}+\mathbf{r},(\mathbf{r}',a,b)}(\tau)]
                                   \cdot[\hat{B}_{\uparrow,\mathbf{i},(\mathbf{r}'',c,d)}(0)+\hat{B}_{\downarrow,\mathbf{i},(\mathbf{r}'',c,d)}(0)]\rangle
\end{align*}
```
where the
```math
\hat{B}_{\sigma,\mathbf{i},(\mathbf{r},a,b)}
    = \hat{a}_{\sigma,\mathbf{i}+\mathbf{r}}^{\dagger}\hat{b}_{\sigma,\mathbf{i}}^{\phantom{\dagger}}
    + \hat{b}_{\sigma,\mathbf{i}}^{\dagger}\hat{a}_{\sigma,\mathbf{i}+\mathbf{r}}^{\phantom{\dagger}}
```
is the bond operator.

# Fields

- `BB::AbstractArray{C,D}`: Array the bond correlation function ``\mathcal{B}_{\mathbf{r}}^{(\mathbf{r}',a,b),(\mathbf{r}'',c,d)}(\tau)`` is added to.
- `b′::Bond{D}`: Bond defining the bond operator appearing on the left side of the bond correlation function.
- `b″::Bond{D}`: Bond defining the bond operator appearing on the right side of the bond correlation function.
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
function bond_correlation!(
    BB::AbstractArray{C,D},
    b′::Bond{D}, b″::Bond{D}, unit_cell::UnitCell{D}, lattice::Lattice{D},
    Gup_τ0::AbstractMatrix{T}, Gup_0τ::AbstractMatrix{T},
    Gup_ττ::AbstractMatrix{T}, Gup_00::AbstractMatrix{T},
    Gdn_τ0::AbstractMatrix{T}, Gdn_0τ::AbstractMatrix{T},
    Gdn_ττ::AbstractMatrix{T}, Gdn_00::AbstractMatrix{T},
    sgn::T=one(T)
) where {D, C<:Number, T<:Number}

    # up-up bond-bond correlation
    bond_correlation!(BB, b′, b″, unit_cell, lattice, Gup_τ0, Gup_0τ, Gup_ττ, Gup_00, +1, +1, sgn)

    # dn-dn bond-bond correlation
    bond_correlation!(BB, b′, b″, unit_cell, lattice, Gdn_τ0, Gdn_0τ, Gdn_ττ, Gdn_00, -1, -1, sgn)

    # up-dn bond-bond correlation
    bond_correlation!(BB, b′, b″, unit_cell, lattice, Gup_τ0, Gup_0τ, Gup_ττ, Gdn_00, +1, -1, sgn)

    # dn-up bond-bond correlation
    bond_correlation!(BB, b′, b″, unit_cell, lattice, Gdn_τ0, Gdn_0τ, Gdn_ττ, Gup_00, -1, +1, sgn)

    return nothing
end

@doc raw"""
    bond_correlation!(
        BB::AbstractArray{C,D},
        b′::Bond{D}, b″::Bond{D}, unit_cell::UnitCell{D}, lattice::Lattice{D},
        Gσ′_τ0::AbstractMatrix{T}, Gσ′_0τ::AbstractMatrix{T},
        Gσ′_ττ::AbstractMatrix{T}, Gσ″_00::AbstractMatrix{T},
        σ′::Int, σ″::Int, sgn::T=one(T)
    ) where {D, C<:Number, T<:Number}

Calculate the spin-resolved uneqaul-time bond-bond correlation function
```math
\begin{align*}
    \mathcal{B}_{\mathbf{r}}^{(\mathbf{r}',a,b,\sigma'),(\mathbf{r}'',c,d,\sigma'')}(\tau)
        = \frac{1}{N}\sum_{\mathbf{i}} & \mathcal{B}_{\mathbf{i}+\mathbf{r},\mathbf{i}}^{(\mathbf{r}',a,b,\sigma'),(\mathbf{r}'',c,d,\sigma'')}(\tau,0)\\
        =\frac{1}{N}\sum_{\mathbf{i}} & \langle\hat{B}_{\sigma',\mathbf{i}+\mathbf{r},(\mathbf{r}',a,b)}(\tau)\hat{B}_{\sigma'',\mathbf{i},(\mathbf{r}'',c,d)}(0)\rangle,
\end{align*}
```
where
```math
\begin{align*}
    \hat{B}_{\sigma,\mathbf{i},(\mathbf{r},a,b)}
        & = \hat{a}_{\sigma,\mathbf{i}+\mathbf{r}}^{\dagger}\hat{b}_{\sigma,\mathbf{i}}^{\phantom{\dagger}}+\hat{b}_{\sigma,\mathbf{i}}^{\dagger}\hat{a}_{\sigma,\mathbf{i}+\mathbf{r}}^{\phantom{\dagger}}\\
        & = -\hat{a}_{\sigma,\mathbf{i}+\mathbf{r}}^{\phantom{\dagger}}\hat{b}_{\sigma,\mathbf{i}}^{\dagger}-\hat{b}_{\sigma,\mathbf{i}}^{\phantom{\dagger}}\hat{a}_{\sigma,\mathbf{i}+\mathbf{r}}^{\dagger}
\end{align*}
```
is the bond operator.

# Fields

- `BB::AbstractArray{C,D}`: Array the spin-resolved bond correlation function ``\mathcal{B}_{\mathbf{r}}^{(\mathbf{r}',a,b,\sigma'),(\mathbf{r}'',c,d,\sigma'')}(\tau)`` is added to.
- `b′::Bond{D}`: Bond defining the bond operator appearing on the left side of the bond correlation function.
- `b″::Bond{D}`: Bond defining the bond operator appearing on the right side of the bond correlation function.
- `unit_cell::UnitCell{D}`: Defines unit cell.
- `lattice::Lattice{D}`: Specifies size of finite lattice.
- `Gσ′_τ0::AbstractMatrix{T}`: The matrix ``G_{\sigma'}(\tau,0).``
- `Gσ′_0τ::AbstractMatrix{T}`: The matrix ``G_{\sigma'}(0,\tau).``
- `Gσ′_ττ::AbstractMatrix{T}`: The matrix ``G_{\sigma'}(\tau,\tau).``
- `Gσ″_00::AbstractMatrix{T}`: The matrix ``G_{\sigma''}(0,0).``
- `σ′::Int`: The electron spin appearing in the ``\hat{B}_{\sigma',\mathbf{i}+\mathbf{r},(\mathbf{r}',a,b)}`` bond operator.
- `σ″::Int`: The electron spin appearing in the ``\hat{B}_{\sigma'',\mathbf{i},(\mathbf{r}'',c,d)}`` bond operator.
- `sgn::T=one(T)`: The sign of the weight appearing in a DQMC simulation.
"""
function bond_correlation!(
    BB::AbstractArray{C,D},
    b′::Bond{D}, b″::Bond{D}, unit_cell::UnitCell{D}, lattice::Lattice{D},
    Gσ′_τ0::AbstractMatrix{T}, Gσ′_0τ::AbstractMatrix{T},
    Gσ′_ττ::AbstractMatrix{T}, Gσ″_00::AbstractMatrix{T},
    σ′::Int, σ″::Int, sgn::T=one(T)
) where {D, C<:Number, T<:Number}

    # b′ = r′ + (r_a - r_b)
    b, a = b′.orbitals
    r′ = b′.displacement

    # b″ = r″ + (r_c - r_d)
    d, c = b″.orbitals
    r″ = b″.displacement

    # zero vector
    z = @SVector zeros(Int, D)

    # BB(τ,r) = BB(τ,r) + Gσ′(a,i+r+r′,τ|b,i+r,τ)⋅Gσ″(c,i+r″,0|d,i,0)
    contract_Grr_G00!(BB, Gσ′_ττ, Gσ″_00, a, b, c, d, r′, z, r″, z, 1, unit_cell, lattice, sgn)

    # BB(τ,r) = BB(τ,r) + Gσ′(a,i+r+r′,τ|b,i+r,τ)⋅Gσ″(d,i,0|c,i+r″,0)
    contract_Grr_G00!(BB, Gσ′_ττ, Gσ″_00, a, b, d, c, r′, z, z, r″, 1, unit_cell, lattice, sgn)

    # BB(τ,r) = BB(τ,r) + Gσ′(b,i+r,τ|a,i+r+r′,τ)⋅Gσ″(c,i+r″,0|d,i,0)
    contract_Grr_G00!(BB, Gσ′_ττ, Gσ″_00, b, a, c, d, z, r′, r″, z, 1, unit_cell, lattice, sgn)

    # BB(τ,r) = BB(τ,r) + Gσ′(b,i+r,τ|a,i+r+r′,τ)⋅Gσ″(d,i,0|c,i+r″,0)
    contract_Grr_G00!(BB, Gσ′_ττ, Gσ″_00, b, a, d, c, z, r′, z, r″, 1, unit_cell, lattice, sgn)

    # if bond correlation between same spin species
    if σ′ == σ″

        # BB(τ,r) = BB(τ,r) - Gσ′(c,i+r″,0|b,i+r,τ)⋅Gσ′(a,i+r+r′,τ|d,i,0)
        contract_G0r_Gr0!(BB, Gσ′_0τ, Gσ′_τ0, c, b, a, d, r″, z, r′, z, -1, unit_cell, lattice, sgn)

        # BB(τ,r) = BB(τ,r) - Gσ′(d,i,0|b,i+r,τ)⋅Gσ′(a,i+r+r′,τ|c,i+r″,0)
        contract_G0r_Gr0!(BB, Gσ′_0τ, Gσ′_τ0, d, b, a, c, z, z, r′, r″, -1, unit_cell, lattice, sgn)

        # BB(τ,r) = BB(τ,r) - Gσ′(c,i+r″,0|a,i+r+r′,τ)⋅Gσ′(b,i+r,τ|d,i,0)
        contract_G0r_Gr0!(BB, Gσ′_0τ, Gσ′_τ0, c, a, b, d, r″, r′, z, z, -1, unit_cell, lattice, sgn)

        # BB(τ,r) = BB(τ,r) - Gσ′(d,i,0|a,i+r+r′,τ)⋅Gσ′(b,i+r,τ|c,i+r″,0)
        contract_G0r_Gr0!(BB, Gσ′_0τ, Gσ′_τ0, d, a, b, c, z, r′, z, r″, -1, unit_cell, lattice, sgn)
    end

    return nothing
end