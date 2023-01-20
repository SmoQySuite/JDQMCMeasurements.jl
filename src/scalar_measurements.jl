@doc raw"""
    measure_N(G::AbstractMatrix{T}) where {T}

Measure the total particle number ``\langle \hat{N}_\sigma \rangle`` given an
equal-time Green's function matrix ``G_\sigma(\tau,\tau).``
"""
function measure_N(G::AbstractMatrix{T}) where {T}

    # number of orbitals in lattice
    N = size(G,1)

    # total particle number
    N̄ = N - tr(G)

    return N̄
end

@doc raw"""
    measure_N(G::AbstractMatrix{T}, a::Int, unit_cell::UnitCell) where {T}

Measure the total particle number ``\langle \hat{N}_{\sigma,a} \rangle`` in orbital species ``a,`` 
given an equal-time Green's function matrix ``G_\sigma(\tau,\tau).``
"""
function measure_N(G::AbstractMatrix{T}, a::Int, unit_cell::UnitCell) where {T}

    # number of orbitals per unit cell
    n = unit_cell.n

    # view of Green's function for relevant orbital
    Ga = @view G[a:n:end,a:n:end]

    return measure_N(Ga)
end


@doc raw"""
    measure_n(G::AbstractMatrix{T}) where {T}

Measure the average density ``\langle \hat{n}_\sigma \rangle`` given the equal-time
Green's function matrix ``G_\sigma(\tau,\tau).``
"""
function measure_n(G::AbstractMatrix{T}) where {T}

    # measure ⟨N̂⟩
    N̄ = measure_N(G)
    N = size(G,1)

    # calculate ⟨n̂⟩ = ⟨N̂⟩/N
    n̄ = N̄/N

    return n̄
end

@doc raw"""
    measure_n(G::AbstractMatrix{T}, a::Int, unit_cell::UnitCell) where {T}

Measure the average density ``\langle \hat{n}_{\sigma,a} \rangle`` for orbital species ``a,``
given the equal-time Green's function matrix ``G_\sigma(\tau,\tau).``
"""
function measure_n(G::AbstractMatrix{T}, a::Int, unit_cell::UnitCell) where {T}

    # number of orbitals per unit cell
    n = unit_cell.n

    # view of Green's function for relevant orbital
    Ga = @view G[a:n:end,a:n:end]

    return measure_n(Ga)
end


@doc raw"""
    measure_Nsqrd(Gup::AbstractMatrix{T}, Gdn::AbstractMatrix{T}) where {T}

Measure the expectation value of the total particle number squared ``\langle \hat{N}^2 \rangle``
given both the spin-up and spin-down equal-time Green's function matrices ``G_\uparrow(\tau,\tau)``
and ``G_\downarrow(\tau,\tau)`` respectively.
"""
function measure_Nsqrd(Gup::AbstractMatrix{T}, Gdn::AbstractMatrix{T}) where {T}

    # Full Expression: ⟨N²⟩ = ⟨N⟩² + Tr[G₊] + Tr[G₋] - Tr[G₊²] - Tr[G₋²]

    # measure ⟨N₊⟩
    N̄₊ = measure_N(Gup)

    # measure ⟨N₋⟩
    N̄₋ = measure_N(Gdn)

    # ⟨N⟩ = ⟨N₊⟩ + ⟨N₋⟩
    N̄ = N̄₊ + N̄₋

    # ⟨N⟩² + Tr[G₊] + Tr[G₋]
    N² = N̄^2 + tr(Gup) + tr(Gdn)

    # ⟨N²⟩ = (⟨N⟩² + Tr[G₊] + Tr[G₋]) - Tr[G₊²] - Tr[G₋²]
    @fastmath @inbounds for i in axes(Gup,2)
        for j in axes(Gup,1)
            N² -= Gup[i,j] * Gup[j,i]
            N² -= Gdn[i,j] * Gdn[j,i]
        end
    end

    return N²
end


@doc raw"""
    measure_double_occ(Gup::AbstractMatrix{T}, Gdn::AbstractMatrix{T}) where {T}

Measure the double-occupancy ``\langle \hat{n}_\uparrow \hat{n}_\downarrow \rangle``
given both the spin-up and spin-down equal-time Green's function matrices ``G_\uparrow(\tau,\tau)``
and ``G_\downarrow(\tau,\tau)`` respectively.
"""
function measure_double_occ(Gup::AbstractMatrix{T}, Gdn::AbstractMatrix{T}) where {T}

    nup_ndn = zero(T)
    @fastmath @inbounds for i in axes(Gup,1)
        # ⟨n̂₊[i]⋅n̂₋[i]⟩ = (1-G₊[i,i])⋅(1-G₋[i,i])
        nup_ndn += (1-Gup[i,i]) * (1-Gdn[i,i])
    end
    N = size(Gup,1)
    nup_ndn = nup_ndn / N

    return nup_ndn
end

@doc raw"""
    measure_double_occ(Gup::AbstractMatrix{T}, Gdn::AbstractMatrix{T}, a::Int, unit_cell::UnitCell) where {T}

Measure the double-occupancy ``\langle \hat{n}_{\uparrow,a} \hat{n}_{\downarrow,a} \rangle`` for orbital species ``a,``
given both the spin-up and spin-down equal-time Green's function matrices ``G_\uparrow(\tau,\tau)``
and ``G_\downarrow(\tau,\tau)`` respectively.
"""
function measure_double_occ(Gup::AbstractMatrix{T}, Gdn::AbstractMatrix{T}, a::Int, unit_cell::UnitCell) where {T}

    # number of orbitals per unit cell
    n = unit_cell.n

    # get view for orbital a
    Gup_a = @view Gup[a:n:end,a:n:end]
    Gdn_a = @view Gdn[a:n:end,a:n:end]

    return measure_double_occ(Gup_a, Gdn_a)
end