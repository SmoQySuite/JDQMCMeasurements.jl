@doc raw"""
    current_correlation!(CC::AbstractArray{C,D},
                         b′::Bond{D}, b″::Bond{D}, t′::AbstractArray{T,D}, t″::AbstractArray{T,D},
                         unit_cell::UnitCell{D}, lattice::Lattice{D},
                         Gup_τ0::AbstractMatrix{T}, Gup_0τ::AbstractMatrix{T}, Gup_ττ::AbstractMatrix{T}, Gup_00::AbstractMatrix{T},
                         Gdn_τ0::AbstractMatrix{T}, Gdn_0τ::AbstractMatrix{T}, Gdn_ττ::AbstractMatrix{T}, Gdn_00::AbstractMatrix{T},
                         sgn::T=one(T)) where {D, C<:Number, T<:Number}

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
"""
function current_correlation!(CC::AbstractArray{C,D},
                              b′::Bond{D}, b″::Bond{D}, t′::AbstractArray{T,D}, t″::AbstractArray{T,D},
                              unit_cell::UnitCell{D}, lattice::Lattice{D},
                              Gup_τ0::AbstractMatrix{T}, Gup_0τ::AbstractMatrix{T}, Gup_ττ::AbstractMatrix{T}, Gup_00::AbstractMatrix{T},
                              Gdn_τ0::AbstractMatrix{T}, Gdn_0τ::AbstractMatrix{T}, Gdn_ττ::AbstractMatrix{T}, Gdn_00::AbstractMatrix{T},
                              sgn::T=one(T)) where {D, C<:Number, T<:Number}

    # b′ = r′ + (r_a - r_b)
    b, a = b′.orbitals
    r′ = b′.displacement

    # b″ = r″ + (r_c - r_d)
    d, c = b″.orbitals
    r″ = b″.displacement

    # zero vector
    z = @SVector zeros(Int, D)

    # CC(τ,r) -= 1/N sum_i [t(a,i+r+r′|b,i+r)⋅t(c,i+r″|d,i)]⋅G₊(a,i+r+r′,τ|b,i+r,τ)⋅G₊(c,i+r″,0|d,i,0)
    contract_Grr_G00!(CC, Gup_ττ, Gup_00, t′, t″, a, b, c, d, r′, z, r″, z, -1, unit_cell, lattice, sgn)
    # CC(τ,r) -= 1/N sum_i [t(a,i+r+r′|b,i+r)⋅t(c,i+r″|d,i)]⋅G₋(a,i+r+r′,τ|b,i+r,τ)⋅G₋(c,i+r″,0|d,i,0)
    contract_Grr_G00!(CC, Gdn_ττ, Gdn_00, t′, t″, a, b, c, d, r′, z, r″, z, -1, unit_cell, lattice, sgn)

    # CC(τ,r) += 1/N sum_i [t(a,i+r+r′|b,i+r)⋅t(c,i+r″|d,i)]⋅G₊(c,i+r″,0|b,i+r,τ)⋅G₊(a,i+r+r′,τ|d,i,0)
    contract_G0r_Gr0!(CC, Gup_0τ, Gup_τ0, t′, t″, c, b, a, d, r″, z, r′, z, +1, unit_cell, lattice, sgn)
    # CC(τ,r) += 1/N sum_i [t(a,i+r+r′|b,i+r)⋅t(c,i+r″|d,i)]⋅G₋(c,i+r″,0|b,i+r,τ)⋅G₋(a,i+r+r′,τ|d,i,0)
    contract_G0r_Gr0!(CC, Gdn_0τ, Gdn_τ0, t′, t″, c, b, a, d, r″, z, r′, z, +1, unit_cell, lattice, sgn)

    # CC(τ,r) += 1/N sum_i [t(a,i+r+r′|b,i+r)⋅t(c,i+r″|d,i)]⋅G₊(a,i+r+r′,τ|b,i+r,τ)⋅G₊(d,i,0|c,i+r″,0)
    contract_Grr_G00!(CC, Gup_ττ, Gup_00, t′, t″, a, b, d, c, r′, z, z, r″, +1, unit_cell, lattice, sgn)
    # CC(τ,r) += 1/N sum_i [t(a,i+r+r′|b,i+r)⋅t(c,i+r″|d,i)]⋅G₋(a,i+r+r′,τ|b,i+r,τ)⋅G₋(d,i,0|c,i+r″,0)
    contract_Grr_G00!(CC, Gdn_ττ, Gdn_00, t′, t″, a, b, d, c, r′, z, z, r″, +1, unit_cell, lattice, sgn)

    # CC(τ,r) -= 1/N sum_i [t(a,i+r+r′|b,i+r)⋅t(c,i+r″|d,i)]⋅G₊(d,i,0|b,i+r,τ)⋅G₊(a,i+r+r′,τ|c,i+r″,0)
    contract_G0r_Gr0!(CC, Gup_0τ, Gup_τ0, t′, t″, d, b, a, c, z, z, r′, r″, -1, unit_cell, lattice, sgn)
    # CC(τ,r) -= 1/N sum_i [t(a,i+r+r′|b,i+r)⋅t(c,i+r″|d,i)]⋅G₋(d,i,0|b,i+r,τ)⋅G₋(a,i+r+r′,τ|c,i+r″,0)
    contract_G0r_Gr0!(CC, Gdn_0τ, Gdn_τ0, t′, t″, d, b, a, c, z, z, r′, r″, -1, unit_cell, lattice, sgn)

    # CC(τ,r) += 1/N sum_i [t(a,i+r+r′|b,i+r)⋅t(c,i+r″|d,i)]⋅G₊(b,i+r,τ|a,i+r+r′,τ)⋅G₊(c,i+r″,0|d,i,0)
    contract_Grr_G00!(CC, Gup_ττ, Gup_00, t′, t″, b, a, c, d, z, r′, r″, z, +1, unit_cell, lattice, sgn)
    # CC(τ,r) += 1/N sum_i [t(a,i+r+r′|b,i+r)⋅t(c,i+r″|d,i)]⋅G₋(b,i+r,τ|a,i+r+r′,τ)⋅G₋(c,i+r″,0|d,i,0)
    contract_Grr_G00!(CC, Gdn_ττ, Gdn_00, t′, t″, b, a, c, d, z, r′, r″, z, +1, unit_cell, lattice, sgn)

    # CC(τ,r) -= 1/N sum_i [t(a,i+r+r′|b,i+r)⋅t(c,i+r″|d,i)]⋅G₊(c,i+r″,0|a,i+r+r′,τ)⋅G₊(b,i+r,τ|d,i,0)
    contract_G0r_Gr0!(CC, Gup_0τ, Gup_τ0, t′, t″, c, a, b, d, r″, r′, z, z, -1, unit_cell, lattice, sgn)
    # CC(τ,r) -= 1/N sum_i [t(a,i+r+r′|b,i+r)⋅t(c,i+r″|d,i)]⋅G₋(c,i+r″,0|a,i+r+r′,τ)⋅G₋(b,i+r,τ|d,i,0)
    contract_G0r_Gr0!(CC, Gdn_0τ, Gdn_τ0, t′, t″, c, a, b, d, r″, r′, z, z, -1, unit_cell, lattice, sgn)

    # CC(τ,r) -= 1/N sum_i [t(a,i+r+r′|b,i+r)⋅t(c,i+r″|d,i)]⋅G₊(b,i+r,τ|a,i+r+r′,τ)⋅G₊(d,i,0|c,i+r″,0)
    contract_Grr_G00!(CC, Gup_ττ, Gup_00, t′, t″, b, a, d, c, z, r′, z, r″, -1, unit_cell, lattice, sgn)
    # CC(τ,r) -= 1/N sum_i [t(a,i+r+r′|b,i+r)⋅t(c,i+r″|d,i)]⋅G₋(b,i+r,τ|a,i+r+r′,τ)⋅G₋(d,i,0|c,i+r″,0)
    contract_Grr_G00!(CC, Gdn_ττ, Gdn_00, t′, t″, b, a, d, c, z, r′, z, r″, -1, unit_cell, lattice, sgn)

    # CC(τ,r) += 1/N sum_i [t(a,i+r+r′|b,i+r)⋅t(c,i+r″|d,i)]⋅G₊(d,i,0|a,i+r+r′,τ)⋅G₊(b,i+r,τ|c,i+r″,0)
    contract_G0r_Gr0!(CC, Gup_0τ, Gup_τ0, t′, t″, d, a, b, c, z, r′, z, r″, +1, unit_cell, lattice, sgn)
    # CC(τ,r) += 1/N sum_i [t(a,i+r+r′|b,i+r)⋅t(c,i+r″|d,i)]⋅G₋(d,i,0|a,i+r+r′,τ)⋅G₋(b,i+r,τ|c,i+r″,0)
    contract_G0r_Gr0!(CC, Gdn_0τ, Gdn_τ0, t′, t″, d, a, b, c, z, r′, z, r″, +1, unit_cell, lattice, sgn)

    # CC(τ,r) -= 1/N sum_i [t(a,i+r+r′|b,i+r)⋅t(c,i+r″|d,i)]⋅G₊(a,i+r+r′,τ|b,i+r,τ)⋅G₋(c,i+r″,0|d,i,0)
    contract_Grr_G00!(CC, Gup_ττ, Gdn_00, t′, t″, a, b, c, d, r′, z, r″, z, -1, unit_cell, lattice, sgn)
    # CC(τ,r) -= 1/N sum_i [t(a,i+r+r′|b,i+r)⋅t(c,i+r″|d,i)]⋅G₋(a,i+r+r′,τ|b,i+r,τ)⋅G₊(c,i+r″,0|d,i,0)
    contract_Grr_G00!(CC, Gdn_ττ, Gup_00, t′, t″, a, b, c, d, r′, z, r″, z, -1, unit_cell, lattice, sgn)

    # CC(τ,r) += 1/N sum_i [t(a,i+r+r′|b,i+r)⋅t(c,i+r″|d,i)]⋅G₊(a,i+r+r′,τ|b,i+r,τ)⋅G₋(d,i,0|c,i+r″,0)
    contract_Grr_G00!(CC, Gup_ττ, Gdn_00, t′, t″, a, b, d, c, r′, z, z, r″, +1, unit_cell, lattice, sgn)
    # CC(τ,r) += 1/N sum_i [t(a,i+r+r′|b,i+r)⋅t(c,i+r″|d,i)]⋅G₋(a,i+r+r′,τ|b,i+r,τ)⋅G₊(d,i,0|c,i+r″,0)
    contract_Grr_G00!(CC, Gdn_ττ, Gup_00, t′, t″, a, b, d, c, r′, z, z, r″, +1, unit_cell, lattice, sgn)

    # CC(τ,r) += 1/N sum_i [t(a,i+r+r′|b,i+r)⋅t(c,i+r″|d,i)]⋅G₊(b,i+r,τ|a,i+r+r′,τ)⋅G₋(c,i+r″,0|d,i,0)
    contract_Grr_G00!(CC, Gup_ττ, Gdn_00, t′, t″, b, a, c, d, z, r′, r″, z, +1, unit_cell, lattice, sgn)
    # CC(τ,r) += 1/N sum_i [t(a,i+r+r′|b,i+r)⋅t(c,i+r″|d,i)]⋅G₋(b,i+r,τ|a,i+r+r′,τ)⋅G₊(c,i+r″,0|d,i,0)
    contract_Grr_G00!(CC, Gdn_ττ, Gup_00, t′, t″, b, a, c, d, z, r′, r″, z, +1, unit_cell, lattice, sgn)

    # CC(τ,r) -= 1/N sum_i [t(a,i+r+r′|b,i+r)⋅t(c,i+r″|d,i)]⋅G₊(b,i+r,τ|a,i+r+r′,τ)⋅G₋(d,i,0|c,i+r″,0)
    contract_Grr_G00!(CC, Gup_ττ, Gup_00, t′, t″, b, a, d, c, z, r′, z, r″, -1, unit_cell, lattice, sgn)
    # CC(τ,r) -= 1/N sum_i [t(a,i+r+r′|b,i+r)⋅t(c,i+r″|d,i)]⋅G₋(b,i+r,τ|a,i+r+r′,τ)⋅G₊(d,i,0|c,i+r″,0)
    contract_Grr_G00!(CC, Gdn_ττ, Gup_00, t′, t″, b, a, d, c, z, r′, z, r″, -1, unit_cell, lattice, sgn)

    return nothing
end