@doc raw"""
    average_Gr0(G::AbstractMatrix{T}, r::Bond{D}, unit_cell::UnitCell{D,E}, lattice::Lattice{D},
                sgn::T=one(T)) where {D, T<:Number, E<:AbstractFloat}

Calculate the translationally averaged Green's function
```
G_\mathbf{r}^(a,b) = \frac{1}{N} \sum_{\mathbf{i}} G_{\mathbf{i}+\mathbf{r},\mathbf{i}}^{a,b}
```
defined by the bond `r`, which corresponds to the displacement ``\mathbf{r} + (\mathbf{u}_a-\mathbf{u}_b).``
"""
function average_Gr0(G::AbstractMatrix{T}, r::Bond{D}, unit_cell::UnitCell{D,E}, lattice::Lattice{D},
                     sgn::T=one(T)) where {D, T<:Number, E<:AbstractFloat}

    Gr0 = zero(T)

    # orbitals per unit cell
    n = unit_cell.n

    # number of unit cells
    N  = lattice.N

    # dimensions of lattice
    L = lattice.L

    # temporary vector
    tmp = lattice.lvec

    # get a view into the Greens matrix for a and b orbtials
    b, a = r.orbitals
    Gab = @view G[a:n:end, b:n:end]

    # get relative unit cell indices
    i = reshape(1:N, L...) # i
    @. tmp = -r.displacement
    ipr = sa.circshift(i, tmp) # i + r

    # iterate over unit cell
    @fastmath @inbounds for n in eachindex(i)
        Gr0 += Gab[ipr[n],i[n]] 
    end

    # normalize measurement
    Gr0 *= sgn/N

    return Gr0
end


@doc raw"""
    average_ηGr0(G::AbstractMatrix{T}, η::AbstractArray{T,D},
                 r::Bond{D}, unit_cell::UnitCell{D,E}, lattice::Lattice{D},
                 sgn::T=one(T)) where {D, T<:Number, E<:AbstractFloat}

Calculate and return the sum
```math
\frac{1}{N} \sum_{\mathbf{i}} \eta_{\mathbf{i}} G_{\mathbf{i}+\mathbf{r},\mathbf{i}}^{a,b},
```
where the bond `r` defines the displacement ``\mathbf{r} + (\mathbf{u}_a-\mathbf{u}_b).``
"""
function average_ηGr0(G::AbstractMatrix{T}, η::AbstractArray{T,D},
                      r::Bond{D}, unit_cell::UnitCell{D,E}, lattice::Lattice{D},
                      sgn::T=one(T)) where {D, T<:Number, E<:AbstractFloat}

    Gr0 = zero(T)

    # orbitals per unit cell
    n = unit_cell.n

    # number of unit cells
    N  = lattice.N

    # dimensions of lattice
    L = lattice.L

    # temporary vector
    tmp = lattice.lvec

    # get a view into the Greens matrix for a and b orbtials
    b, a = r.orbitals
    Gab = @view G[a:n:end, b:n:end]

    # get relative unit cell indices
    i = reshape(1:N, L...) # i
    @. tmp = -r.displacement
    ipr = sa.circshift(i, tmp) # i + r

    # iterate over unit cell
    @fastmath @inbounds for n in eachindex(i)
        Gr0 += η[i[n]] * Gab[ipr[n],i[n]] 
    end

    # normalize measurement
    Gr0 *= sgn/N

    return Gr0
end



@doc raw"""
    contract_G00!(S::AbstractArray{C}, G::AbstractMatrix{T}, a::Int, b::Int, α::Int,
                  unit_cell::UnitCell{D,E}, lattice::Lattice{D},
                  sgn::T=one(T)) where {D, C<:Complex, T<:Number, E<:AbstractFloat}
Evaluate the sum
```math
\begin{align*}
S_{\mathbf{r}} := S_{\mathbf{r}} + \frac{\alpha}{N}\sum_{\mathbf{i}}G_{\sigma,\mathbf{i},\mathbf{i}}^{a,b}(\tau,0)
\end{align*}
```
for all ``\mathbf{r}.``
"""
function contract_G00!(S::AbstractArray{C}, G::AbstractMatrix{T}, a::Int, b::Int, α::Int,
                       unit_cell::UnitCell{D,E}, lattice::Lattice{D},
                       sgn::T=one(T)) where {D, C<:Complex, T<:Number, E<:AbstractFloat}
    
    # number of unit cells
    N  = lattice.N

    # number of orbitals per unit cell
    n = unit_cell.n

    # view in Greens matrix for relevant orbitals
    Gab = @view G[a:n:end,b:n:end]

    # evaluate sum S(r) = S(r) + α/N sum_i G(a,i|b,i)
    tr_Gab = tr(Gab) # trace of matrix
    αN⁻¹ = sgn * α/N
    @. S += αN⁻¹ * tr_Gab

    return nothing
end


@doc raw"""
    contract_Gr0!(S::AbstractArray{C,D}, G::AbstractMatrix{T}, r′::Bond{D}, α::Int,
                  unit_cell::UnitCell{D,E}, lattice::Lattice{D},
                  sgn::T=one(T)) where {D, C<:Complex, T<:Number, E<:AbstractFloat}

Evaluate the sum
```math
S_{\mathbf{r}}:=S_{\mathbf{r}}+\frac{\alpha}{N}\sum_{\mathbf{i}} G_{\sigma,\mathbf{i}+\mathbf{r}+\mathbf{r}_{1},\mathbf{i}}^{a,b}(\tau,0)
```
for all ``\mathbf{r},`` where the bond `r′` represents the static displacement ``\mathbf{r}_1+(\mathbf{r}_a-\mathbf{r}_b).``
"""
function contract_Gr0!(S::AbstractArray{C,D}, G::AbstractMatrix{T}, r′::Bond{D}, α::Int,
                       unit_cell::UnitCell{D,E}, lattice::Lattice{D},
                       sgn::T=one(T)) where {D, C<:Complex, T<:Number, E<:AbstractFloat}

    # get bond definition
    r₁ = r′.displacement
    b, a = r′.orbitals

    # orbitals per unit cell
    n = unit_cell.n

    # number of unit cells
    N  = lattice.N

    # dimensions of lattice
    L = size(S)

    # temporary vector
    tmp = lattice.lvec

    # Get a view into the Greens matrix for a and b orbtials
    Gab = @view G[a:n:end, b:n:end]

    # evaluate S(r) = S(r) + α/N sum_i G(a,i+r+r₁|b,i)
    i = reshape(1:N, L)
    αN⁻¹ = sgn * α/N
    @fastmath @inbounds for r in CartesianIndices(S)
        @. tmp = -(r.I-1+r₁)
        iprpr₁ = sa.circshift(i, tmp)
        for n in CartesianIndices(S)
            S[r] += αN⁻¹ * Gab[iprpr₁[n], i[n]]
        end
    end

    return nothing
end

@doc raw"""
    contract_Gr0!(S::AbstractArray{C,D}, G::AbstractMatrix{T}, r′::Bond{D}, α::Int,
                  unit_cell::UnitCell{D,E}, lattice::Lattice{D},
                  sgn::T=one(T)) where {D, C<:Complex, T<:Number, E<:AbstractFloat}

Evaluate the sum
```math
S_{\mathbf{r}}:=S_{\mathbf{r}}+\frac{\alpha}{N}\sum_{\mathbf{i}} \eta_{\mathbf{i}} G_{\sigma,\mathbf{i}+\mathbf{r}+\mathbf{r}_{1},\mathbf{i}}^{a,b}(\tau,0)
```
for all ``\mathbf{r},`` where the bond `r′` represents the static displacement ``\mathbf{r}_1+(\mathbf{r}_a-\mathbf{r}_b).``
"""
function contract_Gr0!(S::AbstractArray{C,D}, G::AbstractMatrix{T}, η::AbstractArray{T}, r′::Bond{D}, α::Int,
                       unit_cell::UnitCell{D,E}, lattice::Lattice{D},
                       sgn::T=one(T)) where {D, C<:Complex, T<:Number, E<:AbstractFloat}

    # get bond definition
    r₁ = r′.displacement
    b, a = r′.orbitals

    # orbitals per unit cell
    n = unit_cell.n

    # number of unit cells
    N  = lattice.N

    # dimensions of lattice
    L = size(S)

    # temporary vector
    tmp = lattice.lvec

    # Get a view into the Greens matrix for a and b orbtials
    Gab = @view G[a:n:end, b:n:end]

    # evaluate S(r) = S(r) + α/N sum_i η(i)⋅G(a,i+r+r₁|b,i)
    i = reshape(1:N, L)
    αN⁻¹ = sgn * α/N
    @fastmath @inbounds for r in CartesianIndices(S)
        @. tmp = -(r.I-1+r₁)
        iprpr₁ = sa.circshift(i, tmp)
        for n in CartesianIndices(S)
            S[r] += αN⁻¹ * η[i[n]] * Gab[iprpr₁[n], i[n]]
        end
    end

    return nothing
end


@doc raw"""
    contract_Grr_G00!(S::AbstractArray{C,D}, G₂::AbstractMatrix{T}, G₁::AbstractMatrix{T}, b₂::Bond{D}, b₁::Bond{D},
                      α::Int, unit_cell::UnitCell{D,E}, lattice::Lattice{D},
                      sgn::T=one(T)) where {D, C<:Complex, T<:Number, E<:AbstractFloat}

Evaluate the sum
```math
S_{\mathbf{r}}:=S_{\mathbf{r}}+\frac{\alpha}{N}\sum_{\mathbf{i}}G_{\sigma_{2},\mathbf{i}+\mathbf{r}+\mathbf{r}_{2},\mathbf{i}+\mathbf{r}}^{a,b}(\tau_{2},0)\cdot G_{\sigma_{1},\mathbf{i}+\mathbf{r}_{1},\mathbf{i}}^{c,d}(\tau_{1},0)
```
for all ``\mathbf{r},`` where the bond `b₂` represents the static displacement ``\mathbf{r}_2 + (\mathbf{r}_a - \mathbf{r}_b),``
and the bond `b₁` represents the static displacement ``\mathbf{r}_1 + (\mathbf{r}_c - \mathbf{r}_d).``
"""
function contract_Grr_G00!(S::AbstractArray{C,D}, G₂::AbstractMatrix{T}, G₁::AbstractMatrix{T}, b₂::Bond{D}, b₁::Bond{D},
                           α::Int, unit_cell::UnitCell{D,E}, lattice::Lattice{D},
                           sgn::T=one(T)) where {D, C<:Complex, T<:Number, E<:AbstractFloat}

    b, a = b₂.orbitals
    r₂   = b₂.displacement
    d, c = b₁.orbitals
    r₁   = b₁.displacement

    # number of unit cells
    N = lattice.N

    # number of orbitals per unit cell
    n = unit_cell.n

    # dimensions of lattice
    L = size(S)

    # rename vectors
    tmp = lattice.lvec

    # view of Green's matrix for relevant orbitals
    G₂_ab = @view G₂[a:n:end,b:n:end]
    G₁_cd = @view G₁[c:n:end,d:n:end]

    i = reshape(1:N, L)
    @. tmp = -r₁
    ipr₁ = sa.circshift(i, tmp) # i + r₁
    αN⁻¹ = sgn * α/N
    # iterate over displacements
    @fastmath @inbounds for r in CartesianIndices(S)
        @. tmp = -(r.I-1)
        ipr = sa.circshift(i, tmp) # i + r
        @. tmp = -(r.I-1+r₂)
        iprpr₂ = sa.circshift(i, tmp) # i + r + r₂
        # average over translation symmetry
        for n in CartesianIndices(S)
            # S(r) = S(r) + α/N sum_i G₂(a,i+r+r₂|b,i+r)⋅G₁(c,i+r₁|d,i)
            S[r] += αN⁻¹ * G₂_ab[iprpr₂[n],ipr[n]] * G₁_cd[ipr₁[n],i[n]]
        end
    end

    return nothing
end

@doc raw"""
    contract_Grr_G00!(S::AbstractArray{C,D}, G₂::AbstractMatrix{T}, G₁::AbstractMatrix{T},
                      η₂::AbstractArray{T}, η₁::AbstractArray{T}, b₂::Bond{D}, b₁::Bond{D},
                      α::Int, unit_cell::UnitCell{D,E}, lattice::Lattice{D},
                      sgn::T=one(T)) where {D, C<:Complex, T<:Number, E<:AbstractFloat}

Evaluate the sum
```math
S_{\mathbf{r}}:=S_{\mathbf{r}}+\frac{\alpha}{N}\sum_{\mathbf{i}} \eta_2{\mathbf{i}+\mathbf{r}}\eta_1{\mathbf{i}} G_{\sigma_{2},\mathbf{i}+\mathbf{r}+\mathbf{r}_{2},\mathbf{i}+\mathbf{r}}^{a,b}(\tau_{2},0)\cdot G_{\sigma_{1},\mathbf{i}+\mathbf{r}_{1},\mathbf{i}}^{c,d}(\tau_{1},0)
```
for all ``\mathbf{r},`` where the bond `b₂` represents the static displacement ``\mathbf{r}_2 + (\mathbf{r}_a - \mathbf{r}_b),``
and the bond `b₁` represents the static displacement ``\mathbf{r}_1 + (\mathbf{r}_c - \mathbf{r}_d).``
"""
function contract_Grr_G00!(S::AbstractArray{C,D}, G₂::AbstractMatrix{T}, G₁::AbstractMatrix{T},
                           η₂::AbstractArray{T}, η₁::AbstractArray{T}, b₂::Bond{D}, b₁::Bond{D},
                           α::Int, unit_cell::UnitCell{D,E}, lattice::Lattice{D},
                           sgn::T=one(T)) where {D, C<:Complex, T<:Number, E<:AbstractFloat}

    b, a = b₂.orbitals
    r₂   = b₂.displacement
    d, c = b₁.orbitals
    r₁   = b₁.displacement

    # number of unit cells
    N = lattice.N

    # number of orbitals per unit cell
    n = unit_cell.n

    # dimensions of lattice
    L = size(S)

    # rename vectors
    tmp = lattice.lvec

    # view of Green's matrix for relevant orbitals
    G₂_ab = @view G₂[a:n:end,b:n:end]
    G₁_cd = @view G₁[c:n:end,d:n:end]

    i = reshape(1:N, L)
    @. tmp = -r₁
    ipr₁ = sa.circshift(i, tmp) # i + r₁
    αN⁻¹ = sgn * α/N
    # iterate over displacements
    @fastmath @inbounds for r in CartesianIndices(S)
        @. tmp = -(r.I-1)
        ipr = sa.circshift(i, tmp) # i + r
        @. tmp = -(r.I-1+r₂)
        iprpr₂ = sa.circshift(i, tmp) # i + r + r₂
        # average over translation symmetry
        for n in CartesianIndices(S)
            # S(r) = S(r) + α/N sum_i η(i+r)⋅η(i)⋅G₂(a,i+r+r₂|b,i+r)⋅G₁(c,i+r₁|d,i)
            S[r] += αN⁻¹ * η₂[ipr[n]] * η₁[i[n]] * G₂_ab[iprpr₂[n],ipr[n]] * G₁_cd[ipr₁[n],i[n]]
        end
    end

    return nothing
end

@doc raw"""
    contract_Grr_G00!(S::AbstractArray{C,D}, G₂::AbstractMatrix{T}, G₁::AbstractMatrix{T},
                      a::Int, b::Int, c::Int, d::Int,
                      r₄::AbstractVector{Int}, r₃::AbstractVector{Int}, r₂::AbstractVector{Int}, r₁::AbstractVector{Int},
                      α::Int, unit_cell::UnitCell{D,E}, lattice::Lattice{D},
                      sgn::T=one(T)) where {D, C<:Complex, T<:Number, E<:AbstractFloat}

Evaluate the sum
```math
S_{\mathbf{r}}:=S_{\mathbf{r}}+\frac{\alpha}{N}\sum_{\mathbf{i}} G_{\sigma_{2},\mathbf{i}+\mathbf{r}+\mathbf{r}_{4},\mathbf{i}+\mathbf{r}+\mathbf{r}_{3}}^{a,b}(\tau_{2},0)\cdot G_{\sigma_{1},\mathbf{i}+\mathbf{r}_{2},\mathbf{i}+\mathbf{r}_{1}}^{c,d}(\tau_{1},0)
```
for all ``\mathbf{r}.``
"""
function contract_Grr_G00!(S::AbstractArray{C,D}, G₂::AbstractMatrix{T}, G₁::AbstractMatrix{T},
                           a::Int, b::Int, c::Int, d::Int,
                           r₄::AbstractVector{Int}, r₃::AbstractVector{Int}, r₂::AbstractVector{Int}, r₁::AbstractVector{Int},
                           α::Int, unit_cell::UnitCell{D,E}, lattice::Lattice{D},
                           sgn::T=one(T)) where {D, C<:Complex, T<:Number, E<:AbstractFloat}

    # number of unit cells
    N = lattice.N

    # number of orbitals per unit cell
    n = unit_cell.n

    # dimensions of lattice
    L = size(S)

    # rename vectors
    tmp = lattice.lvec

    # view of Green's matrix for relevant orbitals
    G₂_ab = @view G₂[a:n:end,b:n:end]
    G₁_cd = @view G₁[c:n:end,d:n:end]

    # α/N
    αN⁻¹ = sgn * α/N

    i = reshape(1:N, L) # i
    @. tmp = -r₂
    ipr₂ = sa.circshift(i, tmp) # i + r₂
    @. tmp = -r₁
    ipr₁ = sa.circshift(i, tmp) # i + r₁
    # iterate over displacements
    @fastmath @inbounds for r in CartesianIndices(S)
        @. tmp = -(r.I - 1 + r₄)
        iprpr₄ = sa.circshift(i, tmp) # i + r + r₄
        @. tmp = -(r.I - 1 + r₃)
        iprpr₃ = sa.circshift(i, tmp) # i + r + r₃
        # average over translation symmetry
        for n in CartesianIndices(S)
            # S(r) = S(r) + α/N sum_i G₂(a,i+r+r₄|b,i+r+r₃)⋅G₁(c,i+r₂|d,i+r₁)
            S[r] += αN⁻¹ * G₂_ab[iprpr₄[n],iprpr₃[n]] * G₁_cd[ipr₂[n],ipr₁[n]]
        end
    end

    return nothing
end

@doc raw"""
    contract_Grr_G00!(S::AbstractArray{C,D}, G₂::AbstractMatrix{T}, G₁::AbstractMatrix{T},
                      η₂::AbstractArray{T}, η₁::AbstractArray{T}, a::Int, b::Int, c::Int, d::Int,
                      r₄::AbstractVector{Int}, r₃::AbstractVector{Int}, r₂::AbstractVector{Int}, r₁::AbstractVector{Int},
                      α::Int, unit_cell::UnitCell{D,E}, lattice::Lattice{D},
                      sgn::T=one(T)) where {D, C<:Complex, T<:Number, E<:AbstractFloat}

Evaluate the sum
```math
S_{\mathbf{r}}:=S_{\mathbf{r}}+\frac{\alpha}{N}\sum_{\mathbf{i}} \eta_2{\mathbf{i}+\mathbf{r}}\eta_1{\mathbf{i}} G_{\sigma_{2},\mathbf{i}+\mathbf{r}+\mathbf{r}_{4},\mathbf{i}+\mathbf{r}+\mathbf{r}_{3}}^{a,b}(\tau_{2},0)\cdot G_{\sigma_{1},\mathbf{i}+\mathbf{r}_{2},\mathbf{i}+\mathbf{r}_{1}}^{c,d}(\tau_{1},0)
```
for all ``\mathbf{r}.``
"""
function contract_Grr_G00!(S::AbstractArray{C,D}, G₂::AbstractMatrix{T}, G₁::AbstractMatrix{T},
                           η₂::AbstractArray{T}, η₁::AbstractArray{T}, a::Int, b::Int, c::Int, d::Int,
                           r₄::AbstractVector{Int}, r₃::AbstractVector{Int}, r₂::AbstractVector{Int}, r₁::AbstractVector{Int},
                           α::Int, unit_cell::UnitCell{D,E}, lattice::Lattice{D},
                           sgn::T=one(T)) where {D, C<:Complex, T<:Number, E<:AbstractFloat}

    # number of unit cells
    N = lattice.N

    # number of orbitals per unit cell
    n = unit_cell.n

    # dimensions of lattice
    L = size(S)

    # rename vectors
    tmp = lattice.lvec

    # view of Green's matrix for relevant orbitals
    G₂_ab = @view G₂[a:n:end,b:n:end]
    G₁_cd = @view G₁[c:n:end,d:n:end]

    # α/N
    αN⁻¹ = sgn * α/N

    i = reshape(1:N, L) # i
    @. tmp = -r₂
    ipr₂ = sa.circshift(i, tmp) # i + r₂
    @. tmp = -r₁
    ipr₁ = sa.circshift(i, tmp) # i + r₁
    # iterate over displacements
    @fastmath @inbounds for r in CartesianIndices(S)
        @. tmp = -(r.I-1)
        ipr    = sa.circshift(i, tmp)
        @. tmp = -(r.I - 1 + r₄)
        iprpr₄ = sa.circshift(i, tmp) # i + r + r₄
        @. tmp = -(r.I - 1 + r₃)
        iprpr₃ = sa.circshift(i, tmp) # i + r + r₃
        # average over translation symmetry
        for n in CartesianIndices(S)
            # S(r) = S(r) + α/N sum_i η₂(i+r)⋅η₁(i)⋅G₂(a,i+r+r₄|b,i+r+r₃)⋅G₁(c,i+r₂|d,i+r₁)
            S[r] += αN⁻¹ * η₂[ipr[n]] * η₁[i[n]] * G₂_ab[iprpr₄[n],iprpr₃[n]] * G₁_cd[ipr₂[n],ipr₁[n]]
        end
    end

    return nothing
end


@doc raw"""
    contract_Gr0_Gr0!(S::AbstractArray{C,D}, G₂::AbstractMatrix{T}, G₁::AbstractMatrix{T}, b₂::Bond, b₁::Bond,
                      α::Int, unit_cell::UnitCell{D,E}, lattice::Lattice{D},
                      sgn::T=one(T)) where {D, C<:Complex, T<:Number, E<:AbstractFloat}

Evaluate the sum
```math
S_{\mathbf{r}}:=S_{\mathbf{r}}+\frac{\alpha}{N}\sum_{\mathbf{i}}G_{\sigma_{2},\mathbf{i}+\mathbf{r}+\mathbf{r}_{2},\mathbf{i}+\mathbf{r}_{1}}^{a,c}(\tau_{2},0)\cdot G_{\sigma_{1},\mathbf{i}+\mathbf{r},\mathbf{i}}^{b,d}(\tau_{1},0)
```
for all ``\mathbf{r},`` where the bond `b₂` represents the static displacement ``\mathbf{r}_2 + (\mathbf{r}_a - \mathbf{r}_b),``
and the bond `b₁` represents the static displacement ``\mathbf{r}_1 + (\mathbf{r}_c - \mathbf{r}_d).``
"""
function contract_Gr0_Gr0!(S::AbstractArray{C,D}, G₂::AbstractMatrix{T}, G₁::AbstractMatrix{T}, b₂::Bond, b₁::Bond,
                           α::Int, unit_cell::UnitCell{D,E}, lattice::Lattice{D},
                           sgn::T=one(T)) where {D, C<:Complex, T<:Number, E<:AbstractFloat}

    b, a = b₂.orbitals
    r₂   = b₂.displacement
    d, c = b₁.orbitals
    r₁   = b₁.displacement

    # number of unit cells
    N  = lattice.N

    # dimensions of lattice
    L = size(S)

    # number of orbitals per unit cell
    n = unit_cell.n

    # rename vectors
    tmp = lattice.lvec

    # view greens matrix for given orbitals
    G₂_ac = @view G₂[a:n:end,c:n:end]
    G₁_bd = @view G₁[b:n:end,d:n:end]

    i = reshape(1:N, L)
    @. tmp = -r₁
    ipr₁ = sa.circshift(i, tmp) # i + r₁
    αN⁻¹ = sgn * α/N
    # iterate over displacements
    @fastmath @inbounds for r in CartesianIndices(S)
        @. tmp = -(r.I-1)
        ipr = sa.circshift(i, tmp) # i + r
        @. tmp = -(r.I-1+r₂)
        iprpr₂ = sa.circshift(i, tmp) # i + r + r₂
        # average over translation symmetry
        for n in CartesianIndices(S)
            # S(r) = S(r) + α/N sum_i G₂(a,i+r+r₂|c,i+r₁)⋅G₁(b,i+r|d,i)
            S[r] += αN⁻¹ * G₂_ac[iprpr₂[n],ipr₁[n]] * G₁_bd[ipr[n],i[n]]
        end
    end

    return nothing
end

@doc raw"""
    contract_Gr0_Gr0!(S::AbstractArray{C,D}, G₂::AbstractMatrix{T}, G₁::AbstractMatrix{T},
                      η₂::AbstractArray{T}, η₁::AbstractArray{T}, b₂::Bond, b₁::Bond,
                      α::Int, unit_cell::UnitCell{D,E}, lattice::Lattice{D},
                      sgn::T=one(T)) where {D, C<:Complex, T<:Number, E<:AbstractFloat}

Evaluate the sum
```math
S_{\mathbf{r}}:=S_{\mathbf{r}}+\frac{\alpha}{N}\sum_{\mathbf{i}} \eta_2(\mathbf{i}+\mathbf{r})\eta_1(\mathbf{i}) G_{\sigma_{2},\mathbf{i}+\mathbf{r}+\mathbf{r}_{2},\mathbf{i}+\mathbf{r}_{1}}^{a,c}(\tau_{2},0)\cdot G_{\sigma_{1},\mathbf{i}+\mathbf{r},\mathbf{i}}^{b,d}(\tau_{1},0)
```
for all ``\mathbf{r},`` where the bond `b₂` represents the static displacement ``\mathbf{r}_2 + (\mathbf{r}_a - \mathbf{r}_b),``
and the bond `b₁` represents the static displacement ``\mathbf{r}_1 + (\mathbf{r}_c - \mathbf{r}_d).``
"""
function contract_Gr0_Gr0!(S::AbstractArray{C,D}, G₂::AbstractMatrix{T}, G₁::AbstractMatrix{T},
                           η₂::AbstractArray{T}, η₁::AbstractArray{T}, b₂::Bond, b₁::Bond,
                           α::Int, unit_cell::UnitCell{D,E}, lattice::Lattice{D},
                           sgn::T=one(T)) where {D, C<:Complex, T<:Number, E<:AbstractFloat}

    b, a = b₂.orbitals
    r₂   = b₂.displacement
    d, c = b₁.orbitals
    r₁   = b₁.displacement

    # number of unit cells
    N  = lattice.N

    # dimensions of lattice
    L = size(S)

    # number of orbitals per unit cell
    n = unit_cell.n

    # rename vectors
    tmp = lattice.lvec

    # view greens matrix for given orbitals
    G₂_ac = @view G₂[a:n:end,c:n:end]
    G₁_bd = @view G₁[b:n:end,d:n:end]

    i = reshape(1:N, L)
    @. tmp = -r₁
    ipr₁ = sa.circshift(i, tmp) # i + r₁
    αN⁻¹ = sgn * α/N
    # iterate over displacements
    @fastmath @inbounds for r in CartesianIndices(S)
        @. tmp = -(r.I-1)
        ipr = sa.circshift(i, tmp) # i + r
        @. tmp = -(r.I-1+r₂)
        iprpr₂ = sa.circshift(i, tmp) # i + r + r₂
        # average over translation symmetry
        for n in CartesianIndices(S)
            # S(r) = S(r) + α/N sum_i η₂(i+r)⋅η₁(i)⋅G₂(a,i+r+r₂|c,i+r₁)⋅G₁(b,i+r|d,i)
            S[r] += αN⁻¹ * η₂[ipr[n]] * η₁[i[n]] * G₂_ac[iprpr₂[n],ipr₁[n]] * G₁_bd[ipr[n],i[n]]
        end
    end

    return nothing
end

@doc raw"""
    contract_Gr0_Gr0!(S::AbstractArray{C,D}, G₂::AbstractMatrix{T}, G₁::AbstractMatrix{T},
                      a::Int, b::Int, c::Int, d::Int,
                      r₄::AbstractVector{Int}, r₃::AbstractVector{Int}, r₂::AbstractVector{Int}, r₁::AbstractVector{Int},
                      α::Int, unit_cell::UnitCell{D,E}, lattice::Lattice{D},
                      sgn::T=one(T)) where {D, C<:Complex, T<:Number, E<:AbstractFloat}

Evaluate the sum
```math
S_{\mathbf{r}}:=S_{\mathbf{r}}+\frac{\alpha}{N}\sum_{\mathbf{i}}G_{\sigma_{2},\mathbf{i}+\mathbf{r}+\mathbf{r}_{4},\mathbf{i}+\mathbf{r}_{3}}^{a,b}(\tau_{2},0)\cdot G_{\sigma_{1},\mathbf{i}+\mathbf{r}+\mathbf{r}_{2},\mathbf{i}+\mathbf{r}_{1}}^{c,d}(\tau_{1},0)
```
for all ``\mathbf{r}.``
"""
function contract_Gr0_Gr0!(S::AbstractArray{C,D}, G₂::AbstractMatrix{T}, G₁::AbstractMatrix{T},
                           a::Int, b::Int, c::Int, d::Int,
                           r₄::AbstractVector{Int}, r₃::AbstractVector{Int}, r₂::AbstractVector{Int}, r₁::AbstractVector{Int},
                           α::Int, unit_cell::UnitCell{D,E}, lattice::Lattice{D},
                           sgn::T=one(T)) where {D, C<:Complex, T<:Number, E<:AbstractFloat}

    # number of unit cells
    N = lattice.N

    # number of orbitals per unit cell
    n = unit_cell.n

    # dimensions of lattice
    L = size(S)

    # rename vectors
    tmp = lattice.lvec

    # view of Green's matrix for relevant orbitals
    G₂_ab = @view G₂[a:n:end,b:n:end]
    G₁_cd = @view G₁[c:n:end,d:n:end]

    # α/N
    αN⁻¹ = sgn * α/N

    i = reshape(1:N, L) # i
    @. tmp = -r₃
    ipr₃ = sa.circshift(i, tmp) # i + r₃
    @. tmp = -r₁
    ipr₁ = sa.circshift(i, tmp) # i + r₁
    # iterate over displacements
    @fastmath @inbounds for r in CartesianIndices(S)
        @. tmp = -(r.I - 1 + r₄)
        iprpr₄ = sa.circshift(i, tmp) # i + r + r₄
        @. tmp = -(r.I - 1 + r₂)
        iprpr₂ = sa.circshift(i, tmp) # i + r + r₂
        # average over translation symmetry
        for n in CartesianIndices(S)
            # S(r) = S(r) + α/N sum_i G₂(a,i+r+r₄|b,i+r₃)⋅G₁(c,i+r+r₂|d,i+r₁)
            S[r] += αN⁻¹ * G₂_ab[iprpr₄[n],ipr₃[n]] * G₁_cd[iprpr₂[n],ipr₁[n]]
        end
    end

    return nothing
end

@doc raw"""
    contract_Gr0_Gr0!(S::AbstractArray{C,D}, G₂::AbstractMatrix{T}, G₁::AbstractMatrix{T},
                      η₂::AbstractArray{T}, η₁::AbstractArray{T}, a::Int, b::Int, c::Int, d::Int,
                      r₄::AbstractVector{Int}, r₃::AbstractVector{Int}, r₂::AbstractVector{Int}, r₁::AbstractVector{Int},
                      α::Int, unit_cell::UnitCell{D,E}, lattice::Lattice{D},
                      sgn::T=one(T)) where {D, C<:Complex, T<:Number, E<:AbstractFloat}

Evaluate the sum
```math
S_{\mathbf{r}}:=S_{\mathbf{r}}+\frac{\alpha}{N}\sum_{\mathbf{i}} \eta_2(\mathbf{i}+\mathbf{r})\eta_1(\mathbf{i}) G_{\sigma_{2},\mathbf{i}+\mathbf{r}+\mathbf{r}_{4},\mathbf{i}+\mathbf{r}_{3}}^{a,b}(\tau_{2},0)\cdot G_{\sigma_{1},\mathbf{i}+\mathbf{r}+\mathbf{r}_{2},\mathbf{i}+\mathbf{r}_{1}}^{c,d}(\tau_{1},0)
```
for all ``\mathbf{r}.``
"""
function contract_Gr0_Gr0!(S::AbstractArray{C,D}, G₂::AbstractMatrix{T}, G₁::AbstractMatrix{T},
                           η₂::AbstractArray{T}, η₁::AbstractArray{T}, a::Int, b::Int, c::Int, d::Int,
                           r₄::AbstractVector{Int}, r₃::AbstractVector{Int}, r₂::AbstractVector{Int}, r₁::AbstractVector{Int},
                           α::Int, unit_cell::UnitCell{D,E}, lattice::Lattice{D},
                           sgn::T=one(T)) where {D, C<:Complex, T<:Number, E<:AbstractFloat}

    # number of unit cells
    N = lattice.N

    # number of orbitals per unit cell
    n = unit_cell.n

    # dimensions of lattice
    L = size(S)

    # rename vectors
    tmp = lattice.lvec

    # view of Green's matrix for relevant orbitals
    G₂_ab = @view G₂[a:n:end,b:n:end]
    G₁_cd = @view G₁[c:n:end,d:n:end]

    # α/N
    αN⁻¹ = sgn * α/N

    i = reshape(1:N, L) # i
    @. tmp = -r₃
    ipr₃ = sa.circshift(i, tmp) # i + r₃
    @. tmp = -r₁
    ipr₁ = sa.circshift(i, tmp) # i + r₁
    # iterate over displacements
    @fastmath @inbounds for r in CartesianIndices(S)
        @. tmp = -(r.i - 1)
        ipr    = sa.circshift(i, tmp) # i + r
        @. tmp = -(r.I - 1 + r₄)
        iprpr₄ = sa.circshift(i, tmp) # i + r + r₄
        @. tmp = -(r.I - 1 + r₂)
        iprpr₂ = sa.circshift(i, tmp) # i + r + r₂
        # average over translation symmetry
        for n in CartesianIndices(S)
            # S(r) = S(r) + α/N sum_i η₂(i+r)⋅η₁(i)⋅G₂(a,i+r+r₄|b,i+r₃)⋅G₁(c,i+r+r₂|d,i+r₁)
            S[r] += αN⁻¹ * η₂[ipr[n]] * η₁[i[n]] * G₂_ab[iprpr₄[n],ipr₃[n]] * G₁_cd[iprpr₂[n],ipr₁[n]]
        end
    end

    return nothing
end


@doc raw"""
    contract_G0r_Gr0!(S::AbstractArray{C,D}, G₂::AbstractMatrix{T}, G₁::AbstractMatrix{T}, b₂::Bond, b₁::Bond,
                      α::Int, unit_cell::UnitCell{D,E}, lattice::Lattice{D},
                      sgn::T=one(T)) where {D, C<:Complex, T<:Number, E<:AbstractFloat}

Evaluate the sum
```math
S_{\mathbf{r}}:=S_{\mathbf{r}}+\frac{\alpha}{N}\sum_{\mathbf{i}}G_{\sigma_{2},\mathbf{i}+\mathbf{r}_{2},\mathbf{i}+\mathbf{r}}^{a,b}(\tau_{2},0)\cdot G_{\sigma_{1},\mathbf{i}+\mathbf{r}+\mathbf{r}_{1},\mathbf{i}}^{c,d}(\tau_{1},0)
```
for all ``\mathbf{r},`` where the bond `b₂` represents the static displacement ``\mathbf{r}_2 + (\mathbf{r}_a - \mathbf{r}_b),``
and the bond `b₁` represents the static displacement ``\mathbf{r}_1 + (\mathbf{r}_c - \mathbf{r}_d).``
"""
function contract_G0r_Gr0!(S::AbstractArray{C,D}, G₂::AbstractMatrix{T}, G₁::AbstractMatrix{T}, b₂::Bond, b₁::Bond,
                           α::Int, unit_cell::UnitCell{D,E}, lattice::Lattice{D},
                           sgn::T=one(T)) where {D, C<:Complex, T<:Number, E<:AbstractFloat}

    b, a = b₂.orbitals
    r₂   = b₂.displacement
    d, c = b₁.orbitals
    r₁   = b₁.displacement

    # number of unit cells
    N  = lattice.N

    # dimensions of lattice
    L = size(S)

    # number of orbitals in unit cell
    n = unit_cell.n

    # rename vectors
    tmp = lattice.lvec

    # view of greens matrix for given orbitals
    G₂_ab = @view G₂[a:n:end,b:n:end]
    G₁_cd = @view G₁[c:n:end,d:n:end]

    i = reshape(1:N, L)
    @. tmp = -r₂
    ipr₂ = sa.circshift(i, tmp) # i + r₂
    αN⁻¹ = sgn * α/N
    # iterate over displacements
    @fastmath @inbounds for r in CartesianIndices(S)
        @. tmp = -(r.I-1)
        ipr = sa.circshift(i, tmp) # i + r
        @. tmp = -((r.I-1)+r₁)
        iprpr₁ = sa.circshift(i, tmp) # i + r + r₁
        # average over translation symmetry
        for n in CartesianIndices(S)
            # S(r) = S(r) + α/N sum_i G₂(a,i+r₂|b,i+r)⋅G₁(c,i+r+r₁|d,i)
            S[r] += αN⁻¹ * G₂_ab[ipr₂[n],ipr[n]] * G₁_cd[iprpr₁[n],i[n]]
        end
    end
    
    return nothing
end

@doc raw"""
    contract_G0r_Gr0!(S::AbstractArray{C,D}, G₂::AbstractMatrix{T}, G₁::AbstractMatrix{T},
                      η₂::AbstractArray{T}, η₁::AbstractArray{T}, b₂::Bond, b₁::Bond,
                      α::Int, unit_cell::UnitCell{D,E}, lattice::Lattice{D},
                      sgn::T=one(T)) where {D, C<:Complex, T<:Number, E<:AbstractFloat}

Evaluate the sum
```math
S_{\mathbf{r}}:=S_{\mathbf{r}}+\frac{\alpha}{N}\sum_{\mathbf{i}} \eta_2(\mathbf{i}+\mathbf{r})\eta_1(\mathbf{i}) G_{\sigma_{2},\mathbf{i}+\mathbf{r}_{2},\mathbf{i}+\mathbf{r}}^{a,b}(\tau_{2},0)\cdot G_{\sigma_{1},\mathbf{i}+\mathbf{r}+\mathbf{r}_{1},\mathbf{i}}^{c,d}(\tau_{1},0)
```
for all ``\mathbf{r},`` where the bond `b₂` represents the static displacement ``\mathbf{r}_2 + (\mathbf{r}_a - \mathbf{r}_b),``
and the bond `b₁` represents the static displacement ``\mathbf{r}_1 + (\mathbf{r}_c - \mathbf{r}_d).``
"""
function contract_G0r_Gr0!(S::AbstractArray{C,D}, G₂::AbstractMatrix{T}, G₁::AbstractMatrix{T},
                           η₂::AbstractArray{T}, η₁::AbstractArray{T}, b₂::Bond, b₁::Bond,
                           α::Int, unit_cell::UnitCell{D,E}, lattice::Lattice{D},
                           sgn::T=one(T)) where {D, C<:Complex, T<:Number, E<:AbstractFloat}

    b, a = b₂.orbitals
    r₂   = b₂.displacement
    d, c = b₁.orbitals
    r₁   = b₁.displacement

    # number of unit cells
    N  = lattice.N

    # dimensions of lattice
    L = size(S)

    # number of orbitals in unit cell
    n = unit_cell.n

    # rename vectors
    tmp = lattice.lvec

    # view of greens matrix for given orbitals
    G₂_ab = @view G₂[a:n:end,b:n:end]
    G₁_cd = @view G₁[c:n:end,d:n:end]

    i = reshape(1:N, L)
    @. tmp = -r₂
    ipr₂ = sa.circshift(i, tmp) # i + r₂
    αN⁻¹ = sgn * α/N
    # iterate over displacements
    @fastmath @inbounds for r in CartesianIndices(S)
        @. tmp = -(r.I-1)
        ipr = sa.circshift(i, tmp) # i + r
        @. tmp = -((r.I-1)+r₁)
        iprpr₁ = sa.circshift(i, tmp) # i + r + r₁
        # average over translation symmetry
        for n in CartesianIndices(S)
            # S(r) = S(r) + α/N sum_i η₂(i+r)⋅η₁(i)⋅G₂(a,i+r₂|b,i+r)⋅G₁(c,i+r+r₁|d,i)
            S[r] += αN⁻¹ * η₂[ipr[n]] * η₁[i[n]] * G₂_ab[ipr₂[n],ipr[n]] * G₁_cd[iprpr₁[n],i[n]]
        end
    end
    
    return nothing
end

@doc raw"""
    contract_G0r_Gr0!(S::AbstractArray{C,D}, G₂::AbstractMatrix{T}, G₁::AbstractMatrix{T},
                      a::Int, b::Int, c::Int, d::Int,
                      r₄::AbstractVector{Int}, r₃::AbstractVector{Int}, r₂::AbstractVector{Int}, r₁::AbstractVector{Int},
                      α::Int, unit_cell::UnitCell{D,E}, lattice::Lattice{D},
                      sgn::T=one(T)) where {D, C<:Complex, T<:Number, E<:AbstractFloat}

Evaluate the sum
```math
S_{\mathbf{r}}:=S_{\mathbf{r}}+\frac{\alpha}{N}\sum_{\mathbf{i}}G_{\sigma_{2},\mathbf{i}+\mathbf{r}_{4},\mathbf{i}+\mathbf{r}+\mathbf{r}_{3}}^{a,b}(\tau_{2},0)\cdot G_{\sigma_{1},\mathbf{i}+\mathbf{r}+\mathbf{r}_{2},\mathbf{i}+\mathbf{r}_{1}}^{c,d}(\tau_{1},0)
```
for all ``\mathbf{r}.``
"""
function contract_G0r_Gr0!(S::AbstractArray{C,D}, G₂::AbstractMatrix{T}, G₁::AbstractMatrix{T},
                           a::Int, b::Int, c::Int, d::Int,
                           r₄::AbstractVector{Int}, r₃::AbstractVector{Int}, r₂::AbstractVector{Int}, r₁::AbstractVector{Int},
                           α::Int, unit_cell::UnitCell{D,E}, lattice::Lattice{D},
                           sgn::T=one(T)) where {D, C<:Complex, T<:Number, E<:AbstractFloat}

    # number of unit cells
    N = lattice.N

    # number of orbitals per unit cell
    n = unit_cell.n

    # dimensions of lattice
    L = size(S)

    # rename vectors
    tmp = lattice.lvec

    # view of Green's matrix for relevant orbitals
    G₂_ab = @view G₂[a:n:end,b:n:end]
    G₁_cd = @view G₁[c:n:end,d:n:end]

    # α/N
    αN⁻¹ = sgn * α/N

    i = reshape(1:N, L) # i
    @. tmp = -r₄
    ipr₄ = sa.circshift(i, tmp) # i + r₄
    @. tmp = -r₁
    ipr₁ = sa.circshift(i, tmp) # i + r₁
    # iterate over displacements
    @fastmath @inbounds for r in CartesianIndices(S)
        @. tmp = -(r.I - 1 + r₃)
        iprpr₃ = sa.circshift(i, tmp) # i + r + r₃
        @. tmp = -(r.I - 1 + r₂)
        iprpr₂ = sa.circshift(i, tmp) # i + r + r₂
        # average over translation symmetry
        for n in CartesianIndices(S)
            # S(r) = S(r) + α/N sum_i G₂(a,i+r₄|b,i+r+r₃)⋅G₁(c,i+r+r₂|d,i+r₁)
            S[r] += αN⁻¹ * G₂_ab[ipr₄[n],iprpr₃[n]] * G₁_cd[iprpr₂[n],ipr₁[n]]
        end
    end

    return nothing
end

@doc raw"""
    contract_G0r_Gr0!(S::AbstractArray{C,D}, G₂::AbstractMatrix{T}, G₁::AbstractMatrix{T},
                      η₂::AbstractArray{T}, η₁::AbstractArray{T}, a::Int, b::Int, c::Int, d::Int,
                      r₄::AbstractVector{Int}, r₃::AbstractVector{Int}, r₂::AbstractVector{Int}, r₁::AbstractVector{Int},
                      α::Int, unit_cell::UnitCell{D,E}, lattice::Lattice{D},
                      sgn::T=one(T)) where {D, C<:Complex, T<:Number, E<:AbstractFloat}

Evaluate the sum
```math
S_{\mathbf{r}}:=S_{\mathbf{r}}+\frac{\alpha}{N}\sum_{\mathbf{i}} \eta_2(\mathbf{i}+\mathbf{r})\eta_1(\mathbf{i}) G_{\sigma_{2},\mathbf{i}+\mathbf{r}_{4},\mathbf{i}+\mathbf{r}+\mathbf{r}_{3}}^{a,b}(\tau_{2},0)\cdot G_{\sigma_{1},\mathbf{i}+\mathbf{r}+\mathbf{r}_{2},\mathbf{i}+\mathbf{r}_{1}}^{c,d}(\tau_{1},0)
```
for all ``\mathbf{r}.``
"""
function contract_G0r_Gr0!(S::AbstractArray{C,D}, G₂::AbstractMatrix{T}, G₁::AbstractMatrix{T},
                           η₂::AbstractArray{T}, η₁::AbstractArray{T}, a::Int, b::Int, c::Int, d::Int,
                           r₄::AbstractVector{Int}, r₃::AbstractVector{Int}, r₂::AbstractVector{Int}, r₁::AbstractVector{Int},
                           α::Int, unit_cell::UnitCell{D,E}, lattice::Lattice{D},
                           sgn::T=one(T)) where {D, C<:Complex, T<:Number, E<:AbstractFloat}

    # number of unit cells
    N = lattice.N

    # number of orbitals per unit cell
    n = unit_cell.n

    # dimensions of lattice
    L = size(S)

    # rename vectors
    tmp = lattice.lvec

    # view of Green's matrix for relevant orbitals
    G₂_ab = @view G₂[a:n:end,b:n:end]
    G₁_cd = @view G₁[c:n:end,d:n:end]

    # α/N
    αN⁻¹ = sgn * α/N

    i = reshape(1:N, L) # i
    @. tmp = -r₄
    ipr₄ = sa.circshift(i, tmp) # i + r₄
    @. tmp = -r₁
    ipr₁ = sa.circshift(i, tmp) # i + r₁
    # iterate over displacements
    @fastmath @inbounds for r in CartesianIndices(S)
        @. tmp = -(r.I - 1)
        ipr    = sa.circshift(i, tmp) # i + r
        @. tmp = -(r.I - 1 + r₃)
        iprpr₃ = sa.circshift(i, tmp) # i + r + r₃
        @. tmp = -(r.I - 1 + r₂)
        iprpr₂ = sa.circshift(i, tmp) # i + r + r₂
        # average over translation symmetry
        for n in CartesianIndices(S)
            # S(r) = S(r) + α/N sum_i η₂(i+r)⋅η₁(i)⋅G₂(a,i+r₄|b,i+r+r₃)⋅G₁(c,i+r+r₂|d,i+r₁)
            S[r] += αN⁻¹ * η₂[ipr[n]] * η₁[i[n]] * G₂_ab[ipr₄[n],iprpr₃[n]] * G₁_cd[iprpr₂[n],ipr₁[n]]
        end
    end

    return nothing
end


@doc raw"""
    simpson(f::AbstractVector{T}, dx::E) where {T<:Number, E<:AbstractFloat}

Applying Simpson's rule, integrate over the vector `f` using a stepsize `dx`.
"""
function simpson(f::AbstractVector{T}, dx::E) where {T<:Number, E<:AbstractFloat}

    L = length(f)
    F = zero(T)
    @fastmath @inbounds for i in 2:2:L-1
        F += dx * ( 1/3*f[i-1] + 4/3*f[i] + 1/3*f[i+1] )
    end
    if iseven(L)
        F += dx * ( 5/12*f[L] + 2/3*f[L-1] - 1/12*f[L-2] )
    end
    return F
end