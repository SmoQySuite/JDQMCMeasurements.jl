@doc raw"""
    current_correlation!(
        CC::AbstractArray{C,D},
        b′::Bond{D}, b″::Bond{D},
        tup′::AbstractArray{T,D}, tup″::AbstractArray{T,D},
        tdn′::AbstractArray{T,D}, tdn″::AbstractArray{T,D},
        unit_cell::UnitCell{D}, lattice::Lattice{D},
        Gup_τ0::AbstractMatrix{T}, Gup_0τ::AbstractMatrix{T},
        Gup_ττ::AbstractMatrix{T}, Gup_00::AbstractMatrix{T},
        Gdn_τ0::AbstractMatrix{T}, Gdn_0τ::AbstractMatrix{T},
        Gdn_ττ::AbstractMatrix{T}, Gdn_00::AbstractMatrix{T},
        sgn=one(C)
    ) where {D, C<:Number, T<:Number}

Calculate the uneqaul-time current-current correlation function
```math
\mathcal{J}_{\mathbf{r}}^{(\mathbf{r}',a,b),(\mathbf{r}'',c,d)}(\tau) = \frac{1}{N}\sum_{\mathbf{i}}
    \langle[\hat{J}_{\uparrow,\mathbf{i}+\mathbf{r},(\mathbf{r}',a,b)}(\tau)+\hat{J}_{\downarrow,\mathbf{i}+\mathbf{r},(\mathbf{r}',a,b)}(\tau)]
    \cdot[\hat{J}_{\uparrow,\mathbf{i},(\mathbf{r}'',c,d)}(0)+\hat{J}_{\downarrow,\mathbf{i},(\mathbf{r}'',c,d)}(0)]\rangle,
```
where the current operator is given by
```math
\begin{align*}
\hat{J}_{\sigma,\mathbf{i},(\mathbf{r},a,b)}= & -{\rm i}t_{\mathbf{i}+\mathbf{r},\mathbf{i}}^{a,b}(\hat{a}_{\sigma,\mathbf{i}+\mathbf{r}}^{\dagger}\hat{b}_{\sigma,\mathbf{i}}-\hat{b}_{\sigma,\mathbf{i}}^{\dagger}\hat{a}_{\sigma,\mathbf{i}+\mathbf{r}})\\
= & -{\rm i}t_{\mathbf{i}+\mathbf{r},\mathbf{i}}^{a,b}(\hat{a}_{\sigma,\mathbf{i}+\mathbf{r}}\hat{b}_{\sigma,\mathbf{i}}^{\dagger}-\hat{b}_{\sigma,\mathbf{i}}\hat{a}_{\sigma,\mathbf{i}+\mathbf{r}}^{\dagger}).
\end{align*}
```

# Fields

- `CC::AbstractArray{C,D}`: Array the current correlation function ``\mathcal{J}_{\mathbf{r}}^{(\mathbf{r}',a,b),(\mathbf{r}'',c,d)}(\tau)`` is added to.
- `b′::Bond{D}`: Bond defining the current operator appearing on the left side of the current correlation function.
- `b″::Bond{D}`: Bond defining the current operator appearing on the right side of the current correlation function.
- `tup′::AbstractArray{T,D}`: Spin up position and imaginary time dependent hopping amplitudes corresponding to bond `b′`.
- `tup″::AbstractArray{T,D}`: Spin up position and imaginary time dependent hopping amplitudes corresponding to bond `b″`.
- `tdn′::AbstractArray{T,D}`: Spin down position and imaginary time dependent hopping amplitudes corresponding to bond `b′`.
- `tdn″::AbstractArray{T,D}`: Spin down position and imaginary time dependent hopping amplitudes corresponding to bond `b″`.
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
- `sgn=one(C)`: The sign of the weight appearing in a DQMC simulation.
"""
function current_correlation!(
    CC::AbstractArray{C,D},
    b′::Bond{D}, b″::Bond{D},
    tup′::AbstractArray{T,D}, tup″::AbstractArray{T,D},
    tdn′::AbstractArray{T,D}, tdn″::AbstractArray{T,D},
    unit_cell::UnitCell{D}, lattice::Lattice{D},
    Gup_τ0::AbstractMatrix{T}, Gup_0τ::AbstractMatrix{T},
    Gup_ττ::AbstractMatrix{T}, Gup_00::AbstractMatrix{T},
    Gdn_τ0::AbstractMatrix{T}, Gdn_0τ::AbstractMatrix{T},
    Gdn_ττ::AbstractMatrix{T}, Gdn_00::AbstractMatrix{T},
    sgn=one(C)
) where {D, C<:Number, T<:Number}

    # up-up current-current correlation
    current_correlation!(CC, b′, b″, tup′, tup″, unit_cell, lattice, Gup_τ0, Gup_0τ, Gup_ττ, Gup_00, +1, +1, sgn)

    # dn-dn current-current correlation
    current_correlation!(CC, b′, b″, tdn′, tdn″, unit_cell, lattice, Gdn_τ0, Gdn_0τ, Gdn_ττ, Gdn_00, -1, -1, sgn)

    # up-dn current-current correlation
    current_correlation!(CC, b′, b″, tup′, tdn″, unit_cell, lattice, Gup_τ0, Gup_0τ, Gup_ττ, Gdn_00, +1, -1, sgn)

    # dn-up current-current correlation
    current_correlation!(CC, b′, b″, tdn′, tup″, unit_cell, lattice, Gdn_τ0, Gdn_0τ, Gdn_ττ, Gup_00, -1, +1, sgn)

    return nothing
end

@doc raw"""
    current_correlation!(
        CC::AbstractArray{C,D},
        b′::Bond{D}, b″::Bond{D},
        t′::AbstractArray{T,D}, t″::AbstractArray{T,D},
        unit_cell::UnitCell{D}, lattice::Lattice{D},
        Gσ′_τ0::AbstractMatrix{T}, Gσ′_0τ::AbstractMatrix{T},
        Gσ′_ττ::AbstractMatrix{T}, Gσ″_00::AbstractMatrix{T},
        σ′::Int, σ″::Int, sgn=one(C)
    ) where {D, C<:Number, T<:Number}

Calculate the spin-resolved uneqaul-time current-current correlation function
```math
\begin{align*}
\mathcal{J}_{\mathbf{r}}^{(\mathbf{r}',a,b,\sigma'),(\mathbf{r}'',c,d,\sigma'')}(\tau)
    & = \frac{1}{N}\sum_{\mathbf{i}}\mathcal{J}_{\mathbf{i}+\mathbf{r},\mathbf{i}}^{(\mathbf{r}',a,b,\sigma'),(\mathbf{r}'',c,d,\sigma'')}(\tau,0)\\
    & = \frac{1}{N}\sum_{\mathbf{i}}\langle\hat{J}_{\sigma',\mathbf{i}+\mathbf{r},(\mathbf{r}',a,b)}(\tau)\hat{J}_{\sigma'',\mathbf{i},(\mathbf{r}'',c,d)}(0)\rangle,
\end{align*}
```
where the spin-resolved current operator is given by
```math
\begin{align*}
    \hat{J}_{\sigma,\mathbf{i},(\mathbf{r},a,b)}
        & = -{\rm i}t_{\sigma,\mathbf{i}+\mathbf{r},\mathbf{i}}^{a,b}(\hat{a}_{\sigma,\mathbf{i}+\mathbf{r}}^{\dagger}\hat{b}_{\sigma,\mathbf{i}}-\hat{b}_{\sigma,\mathbf{i}}^{\dagger}\hat{a}_{\sigma,\mathbf{i}+\mathbf{r}})\\
        & = -{\rm i}t_{\sigma,\mathbf{i}+\mathbf{r},\mathbf{i}}^{a,b}(\hat{a}_{\sigma,\mathbf{i}+\mathbf{r}}\hat{b}_{\sigma,\mathbf{i}}^{\dagger}-\hat{b}_{\sigma,\mathbf{i}}\hat{a}_{\sigma,\mathbf{i}+\mathbf{r}}^{\dagger}).
\end{align*}
```

# Fields

- `CC::AbstractArray{C,D}`: Array the spin-resolved current correlation function ``\mathcal{J}_{\mathbf{r}}^{(\mathbf{r}',a,b,\sigma'),(\mathbf{r}'',c,d,\sigma'')}(\tau)`` is added to.
- `b′::Bond{D}`: Bond defining the current operator appearing on the left side of the current correlation function.
- `b″::Bond{D}`: Bond defining the current operator appearing on the right side of the current correlation function.
- `t′::AbstractArray{T,D}`: Position and imaginary time dependent hopping amplitude associated with bond `b′`.
- `t″::AbstractArray{T,D}`: Position and imaginary time dependent hopping amplitude associated with bond `b″`.
- `unit_cell::UnitCell{D}`: Defines unit cell.
- `lattice::Lattice{D}`: Specifies size of finite lattice.
- `Gσ′_τ0::AbstractMatrix{T}`: The matrix ``G_{\sigma'}(\tau,0).``
- `Gσ′_0τ::AbstractMatrix{T}`: The matrix ``G_{\sigma'}(0,\tau).``
- `Gσ′_ττ::AbstractMatrix{T}`: The matrix ``G_{\sigma'}(\tau,\tau).``
- `Gσ″_00::AbstractMatrix{T}`: The matrix ``G_{\sigma''}(0,0).``
- `σ′::Int`: The electron spin appearing in the left current operator.
- `σ″::Int`: The electron spin appearing in the right current operator.
- `sgn=one(C)`: The sign of the weight appearing in a DQMC simulation.
"""
function current_correlation!(
    CC::AbstractArray{C,D},
    b′::Bond{D}, b″::Bond{D},
    t′::AbstractArray{T,D}, t″::AbstractArray{T,D},
    unit_cell::UnitCell{D}, lattice::Lattice{D},
    Gσ′_τ0::AbstractMatrix{T}, Gσ′_0τ::AbstractMatrix{T}, Gσ′_ττ::AbstractMatrix{T}, Gσ″_00::AbstractMatrix{T},
    σ′::Int, σ″::Int, sgn=one(C)
) where {D, C<:Number, T<:Number}

    # b′ = r′ + (r_a - r_b)
    b, a = b′.orbitals
    r′ = b′.displacement

    # b″ = r″ + (r_c - r_d)
    d, c = b″.orbitals
    r″ = b″.displacement

    # zero vector
    z = @SVector zeros(Int, D)

    # CC(τ,r) = CC(τ,r) + sum_i tσ′(a,i+r|b,i)⋅tσ″(c,i+r″|d,i)⋅Gσ′(a,i+r+r′,τ|b,i+r,τ)⋅Gσ″(d,i,0|c,i+r″,0)
    contract_Grr_G00!(CC, Gσ′_ττ, Gσ″_00, t′, t″, a, b, d, c, r′, z, z, r″, +1, unit_cell, lattice, sgn)

    # CC(τ,r) = CC(τ,r) - sum_i tσ′(a,i+r|b,i)⋅tσ″(c,i+r″|d,i)⋅Gσ′(a,i+r+r′,τ|b,i+r,τ)⋅Gσ″(c,i+r″,0|d,i,0)
    contract_Grr_G00!(CC, Gσ′_ττ, Gσ″_00, t′, t″, a, b, c, d, r′, z, r″, z, -1, unit_cell, lattice, sgn)

    # CC(τ,r) = CC(τ,r) + sum_i tσ′(a,i+r|b,i)⋅tσ″(c,i+r″|d,i)⋅Gσ′(b,i+r,τ|a,i+r+r′,τ)⋅Gσ″(c,i+r″,0|d,i,0)
    contract_Grr_G00!(CC, Gσ′_ττ, Gσ″_00, t′, t″, b, a, c, d, z, r′, r″, z, +1, unit_cell, lattice, sgn)

    # CC(τ,r) = CC(τ,r) - sum_i tσ′(a,i+r|b,i)⋅tσ″(c,i+r″|d,i)⋅Gσ′(b,i+r,τ|a,i+r+r′,τ)⋅Gσ″(d,i,0|c,i+r″,0)
    contract_Grr_G00!(CC, Gσ′_ττ, Gσ″_00, t′, t″, b, a, d, c, z, r′, z, r″, -1, unit_cell, lattice, sgn)

    # if current correlation between same spin species
    if σ′ == σ″

        # CC(τ,r) = CC(τ,r) + sum_i tσ′(a,i+r|b,i)⋅tσ″(c,i+r″|d,i)⋅Gσ′(c,i+r″,0|b,i+r,τ)⋅Gσ″(a,i+r+r′,τ|d,i,0)
        contract_G0r_Gr0!(CC, Gσ′_0τ, Gσ′_τ0, c, b, a, d, r″, z, r′, z, +1, unit_cell, lattice, sgn)

        # CC(τ,r) = CC(τ,r) - sum_i tσ′(a,i+r|b,i)⋅tσ″(c,i+r″|d,i)⋅Gσ′(d,i,0|b,i+r,τ)⋅Gσ″(a,i+r+r′,τ|c,i+r″,0)
        contract_G0r_Gr0!(CC, Gσ′_0τ, Gσ′_τ0, d, b, a, c, z, z, r′, r″, -1, unit_cell, lattice, sgn)

        # CC(τ,r) = CC(τ,r) - sum_i tσ′(a,i+r|b,i)⋅tσ″(c,i+r″|d,i)⋅Gσ′(c,i+r″,0|a,i+r+r′,τ)⋅Gσ″(b,i+r,τ|d,i,0)
        contract_G0r_Gr0!(CC, Gσ′_0τ, Gσ′_τ0, c, a, b, d, r″, r′, z, z, -1, unit_cell, lattice, sgn)

        # CC(τ,r) = CC(τ,r) + sum_i tσ′(a,i+r|b,i)⋅tσ″(c,i+r″|d,i)⋅Gσ′(d,i,0|a,i+r+r′,τ)⋅Gσ″(b,i+r,τ|c,i+r″,0)
        contract_G0r_Gr0!(CC, Gσ′_0τ, Gσ′_τ0, d, a, b, c, z, r′, z, r″, +1, unit_cell, lattice, sgn)
    end

    return nothing
end