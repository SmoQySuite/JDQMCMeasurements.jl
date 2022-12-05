@doc raw"""
    fourier_transform!(C::AbstractArray{Complex{T}}, a::Int, b::Int, dim::Int,
                       unit_cell::UnitCell{D,T}, lattice::Lattice{D}) where {D, T<:AbstractFloat}

    fourier_transform!(C::AbstractArray{Complex{T}}, a::Int, b::Int,
                       unit_cell::UnitCell{D,T}, lattice::Lattice{D}) where {D, T<:AbstractFloat}

Calculate the fourier transform from position to momentum space
```math
\begin{align*}
C_{\mathbf{k}}^{a,b}= & \sum_{\mathbf{r}}e^{{\rm -i}\mathbf{k}\cdot(\mathbf{r}+\mathbf{r}_{a}-\mathbf{r}_{b})}C_{\mathbf{r}}^{a,b}
\end{align*}
```
where ``a`` and ``b`` specify orbital species in the unit cell. Note that the array `C` is modified in-place. If `dim` is passed,
iterate over this dimension of the array, performing a fourier transform on each slice.
"""
function fourier_transform!(C::AbstractArray{Complex{T}}, a::Int, b::Int, dim::Int, unit_cell::UnitCell{D,T}, lattice::Lattice{D}) where {D, T<:AbstractFloat}

    for C_l in eachslice(C, dims=dim)
        fourier_transform!(C_l, a, b, unit_cell, lattice)
    end

    return nothing
end

function fourier_transform!(C::AbstractArray{Complex{T}}, a::Int, b::Int, unit_cell::UnitCell{D,T}, lattice::Lattice{D}) where {D, T<:AbstractFloat}

    # perform standard FFT from position to momentum space
    fft!(C)

    # if two different orbitals
    if a != b

        # initiailize temporary storage vecs
        r_vec   = MVector{D,T}(undef)
        k_point = MVector{D,T}(undef)

        # calculate displacement vector seperating two orbitals in question
        r_a = unit_cell.basis_vecs[a]
        r_b = unit_cell.basis_vecs[b]
        @. r_vec = r_a - r_b

        # have the array index from zero
        C′ = OffsetArrays.Origin(0)(C)        

        # iterate over each k-point
        @fastmath @inbounds for k in CartesianIndices(C′)
            # transform to appropriate gauge accounting for basis vector
            # i.e. relative position of orbitals within unit cell
            calc_k_point!(k_point, k.I, unit_cell, lattice)
            C′[k] = exp(-im*dot(k_point,r_vec)) * C′[k]
        end
    end

    return nothing
end
