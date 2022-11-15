@doc raw"""
    susceptibility!(χ::AbstractArray{T}, S::AbstractArray{T}, Δτ::E, dim::Int) where {T<:Number, E<:AbstractFloat}

Calculate the susceptibilities
```math
\chi_\mathbf{n} = \int_0^\beta S_\mathbf{n}(\tau) d\tau,
```
where the ``\chi_\mathbf{n}`` susceptibilities are written to `χ`,
and `S` contains the ``S_\mathbf{n}(\tau)`` correlations that need to be integrated over.
The parameter `Δτ` is the discretization in imaginary time ``\tau,`` and is the step size
used in Simpson's method to numerically evaluate the integral over imaginary time.
The argument `dim` specifies which dimension of `S` corresponds to imaginary time, and needs to be integrated over.
Accordingly,
```julia
ndim(χ)+1 == ndim(S)
```
and
```julia
size(χ) == size(selectdim(S, dim, 1))
```
must both be true.
"""
function susceptibility!(χ::AbstractArray{T}, S::AbstractArray{T}, Δτ::E, dim::Int) where {T<:Number, E<:AbstractFloat}

    # initialize susceptibilities to zero
    fill!(χ,0)

    # length of imaginary time axis
    L = size(S, dim)

    # perform simpson integration
    @fastmath @inbounds for l in 2:2:L-1
        S_lm1 = selectdim(S, dim, l-1)
        S_l = selectdim(S, dim, l)
        S_lp1 = selectdim(S, dim, l+1)
        @. χ += Δτ * (1/3*S_lm1 + 4/3*S_l + 1/3*S_lp1)
    end

    # take care of boundary condition
    if iseven(L)
        S_L = selectdim(S, dim, L)
        S_Lm1 = selectdim(S, dim, L-1)
        S_Lm2 = selectdim(S, dim, L-2)
        @. χ += Δτ * (5/12*S_L + 2/3*S_Lm1 - 1/12*S_Lm2)
    end

    return nothing
end

@doc raw"""
    susceptibility(S::AbstractVector{T}, Δτ::E) where {T<:Number, E<:AbstractFloat}

Calculate the suceptibility
```math
\chi = \int_0^\beta S(\tau) d\tau,
```
where the correlation data is stored in `S`. The integration is performed
using Simpson's method using a step size of `Δτ`.
"""
function susceptibility(S::AbstractVector{T}, Δτ::E) where {T<:Number, E<:AbstractFloat}

    return simpson(S,Δτ)
end