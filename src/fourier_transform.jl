@doc raw"""
    fourier_transform!(C::AbstractArray{Complex{T}}, a::Int, b::Int, dim::Int,
                      unit_cell::UnitCell{T}, lattice::Lattice) where {T<:AbstractFloat}

    fourier_transform!(C::AbstractArray{Complex{T}}, a::Int, b::Int,
                       unit_cell::UnitCell{T}, lattice::Lattice) where {T<:AbstractFloat}

Calculate the fourier transform from position to momentum space
```math
\begin{align*}
C_{\mathbf{k}}^{a,b}= & \sum_{\mathbf{r}}e^{{\rm -i}\mathbf{k}\cdot(\mathbf{r}+\mathbf{r}_{a}-\mathbf{r}_{b})}C_{\mathbf{r}}^{a,b}
\end{align*}
```
where ``a`` and ``b`` specify orbital species in the unit cell. Note that the array `C` is modified in-place. If `dim` is passed,
iterate over this dimension of the array, performing a fourier transform on each slice.
"""
function fourier_transform!(C::AbstractArray{Complex{T}}, a::Int, b::Int, dim::Int, unit_cell::UnitCell{T}, lattice::Lattice) where {T<:AbstractFloat}

    for C_l in eachslice(C, dims=dim)
        fourier_transform!(C_l, a, b, unit_cell, lattice)
    end

    return nothing
end

function fourier_transform!(C::AbstractArray{Complex{T}}, a::Int, b::Int, unit_cell::UnitCell{T}, lattice::Lattice) where {T<:AbstractFloat}

    # perform standard FFT from position to momentum space
    fft!(C)

    # if two different orbitals
    if a != b

        # get dimension of system
        D = unit_cell.D

        # initiailize temporary storage vecs
        r_vec = zeros(T, D)
        k_pnt = zeros(T, D)
        k_loc = lattice.lvec

        # calculate displacement vector seperating two orbitals in question
        r_a = @view unit_cell.lattice_vecs[:,a]
        r_b = @view unit_cell.lattice_vecs[:,b]
        @. r_vec = r_a - r_b

        # iterate over each k-point
        @fastmath @inbounds for k in CartesianIndices(C)
            # transform to appropriate gauge accounting for basis vector
            # i.e. relative position of orbitals within unit cell
            @. k_loc = k.I - 1
            calc_k_point!(k_pnt, k_loc, unit_cell, lattice)
            C[k] = exp(-im*dot(k_pnt,r_vec)) * C[k]
        end
    end

    return nothing
end
