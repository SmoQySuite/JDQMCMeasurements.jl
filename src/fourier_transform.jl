@doc raw"""
    fourier_transform!(
        C::AbstractArray{Complex{T}},
        a::Int,
        b::Int,
        dims,
        unit_cell::UnitCell{D,T},
        lattice::Lattice{D},
        fftplan!::Union{F, Nothing} = nothing
    ) where {D, T<:AbstractFloat, F<:AbstractFFTs.Plan}

    fourier_transform!(
        C::AbstractArray{Complex{T}},
        r::AbstractVector{T},
        dims,
        unit_cell::UnitCell{D,T},
        lattice::Lattice{D},
        fftplan!::Union{F, Nothing} = nothing
    ) where {D, T<:AbstractFloat, F<:AbstractFFTs.Plan}

    fourier_transform!(
        C::AbstractArray{Complex{T},D},
        a::Int,
        b::Int,
        unit_cell::UnitCell{D,T},
        lattice::Lattice{D},
        fftplan!::Union{F, Nothing} = nothing
    ) where {D, T<:AbstractFloat, F<:AbstractFFTs.Plan}

    fourier_transform!(
        C::AbstractArray{Complex{T},D},
        r::AbstractVector{T},
        unit_cell::UnitCell{D,T},
        lattice::Lattice{D},
        fftplan!::Union{F, Nothing} = nothing
    ) where {D, T<:AbstractFloat, F<:AbstractFFTs.Plan}

Calculate the fourier transform from position to momentum space
```math
\begin{align*}
C_{\mathbf{K}} = & \sum_{\mathbf{R}}e^{{\rm -i}\mathbf{K}\cdot(\mathbf{R}+\mathbf{r})}C_{\mathbf{R}}
\end{align*}
```
where ``\mathbf{r}`` is a constant displacement vector.
If orbitals ``a`` and ``b`` are passed, then ``\mathbf{r} = \mathbf{r}_a - \mathbf{r}_b``
where ``\mathbf{r}_a`` and ``\mathbf{r}_b`` are the basis vectors for each orbital in the unit cell.
Note that the array `C` is modified in-place.
If `dims` is passed, iterate over these dimensions of the array, performing a fourier transform on each slice.
It is also possible to optionally pass an FFT plan to accelerate the fourier transformation, though it
need to be an FFT plan that operates in-place on an array of complexes with the dimension of `C`.
"""
function fourier_transform!(
    C::AbstractArray{Complex{T}},
    a::Int,
    b::Int,
    dims,
    unit_cell::UnitCell{D,T},
    lattice::Lattice{D},
    fftplan!::Union{F, Nothing} = nothing
) where {D, T<:AbstractFloat, F<:AbstractFFTs.Plan}

    # calculate displacement vector seperating the two orbitals in question
    r = zeros(T, D)
    if a != b
        r_a = unit_cell.basis_vecs[a]
        r_b = unit_cell.basis_vecs[b]
        @. r = r_a - r_b
    end
    rvec = SVector{D,T}(r)

    for C_l in eachslice(C, dims = dims)
        fourier_transform!(C_l, rvec, unit_cell, lattice, fftplan!)
    end

    return nothing
end


function fourier_transform!(
    C::AbstractArray{Complex{T}},
    r::AbstractVector{T},
    dims,
    unit_cell::UnitCell{D,T},
    lattice::Lattice{D},
    fftplan!::Union{F, Nothing} = nothing
) where {D, T<:AbstractFloat, F<:AbstractFFTs.Plan}

    @assert length(r) == D "r must be a vector of length $D"
    rvec = SVector{D,T}(r)

    for C_l in eachslice(C, dims = dims)
        fourier_transform!(C_l, rvec, unit_cell, lattice, fftplan!)
    end

    return nothing
end


function fourier_transform!(
    C::AbstractArray{Complex{T},D},
    a::Int,
    b::Int,
    unit_cell::UnitCell{D,T},
    lattice::Lattice{D},
    fftplan!::Union{F, Nothing} = nothing
) where {D, T<:AbstractFloat, F<:AbstractFFTs.Plan}

    # calculate displacement vector seperating the two orbitals in question
    r = zeros(T, D)
    if a != b
        r_a = unit_cell.basis_vecs[a]
        r_b = unit_cell.basis_vecs[b]
        @. r = r_a - r_b
    end
    rvec = SVector{D,T}(r)

    # perform fourier transform
    fourier_transform!(C, rvec, unit_cell, lattice, fftplan!)

    return nothing
end


# perform fourier transform where `r` is a constant displacement vector
function fourier_transform!(
    C::AbstractArray{Complex{T},D},
    r::AbstractVector{T},
    unit_cell::UnitCell{D,T},
    lattice::Lattice{D},
    fftplan!::Union{F, Nothing} = nothing
) where {D, T<:AbstractFloat, F<:AbstractFFTs.Plan}

    @assert length(r) == D "r must be a vector of length $D"
    rvec = SVector{D,T}(r)
    
    fourier_transform!(C, rvec, unit_cell, lattice, fftplan!)

    return nothing
end


# perform fourier transform where `r` is a constant displacement vector
function fourier_transform!(
    C::AbstractArray{Complex{T},D},
    r::SVector{D,T},
    unit_cell::UnitCell{D,T},
    lattice::Lattice{D},
    fftplan!::Union{F, Nothing} = nothing
) where {D, T<:AbstractFloat, F<:AbstractFFTs.Plan}

    # perform standard FFT from position to momentum space
    lmul_fft!(fftplan!, C)

    # if two different orbitals
    if !iszero(r)

        # initiailize temporary storage vecs
        k_point = MVector{D,T}(undef)

        # have the array index from zero
        C′ = oa.Origin(0)(C)        

        # iterate over each k-point
        @inbounds for k in CartesianIndices(C′)
            # transform to appropriate gauge accounting for basis vector
            # i.e. relative position of orbitals within unit cell
            calc_k_point!(k_point, k.I, unit_cell, lattice)
            C′[k] = exp(-im*dot(k_point,r)) * C′[k]
        end
    end

    return nothing
end

lmul_fft!(fftplan!::Nothing, a) = fft!(a)
lmul_fft!(fftplan!::AbstractFFTs.Plan, a) = mul!(a, fftplan!, a)