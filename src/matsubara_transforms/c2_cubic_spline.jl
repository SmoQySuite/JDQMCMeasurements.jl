# Calculate the coefficients of a C2 cubic spline i.e. a spline with a continuous first and second derivative.
# The boundary conditions are such that the second derivative of the spline at τ=0 and τ=β equals
# the one calculated using a second order forward and backward finite difference method respectively.
function c2_cubic_spline!(
    b::AbstractVector{T},
    c::AbstractVector{T},
    d::AbstractVector{T},
    C::AbstractVector{T},
    Δτ::E;
    Cprime::AbstractVector{T} = zeros(T, length(C)),
    middle::AbstractVector{T} = zeros(T, length(C)),
    upper::AbstractVector{T} = zeros(T, length(C)-1),
    lower::AbstractVector{T} = zeros(T, length(C)-1)
) where {T<:Number, E<:AbstractFloat}

    @assert length(b) == length(c) == length(d) == (length(C)-1)
    @assert length(Cprime) == length(middle) == (length(upper)+1) == (length(lower)+1) == length(C)

    # assume piecewise cubic polynomial polynomial that is continuous,
    # as well as its first two derivatives.
    # function form of polynomial for each interval:
    # pₗ(τ) = aₗ + bₗ⋅(τ-τₗ) + cₗ⋅(τ-τₗ)² + dₗ⋅(τ-τₗ)³ for τ ∈ [τₗ,τₗ+Δτ)

    # Calculate S(τₗ) = [C(τₗ₊₁) - C(τₗ)]/Δτ
    S = b
    @views @. S = (C[2:end] - C[1:end-1])/Δτ

    # Calculate v(τₗ₊₁) = 3Δτ[S(τₗ₊₁) + S(τₗ)]
    @views @. Cprime[2:end-1] = 3*Δτ*(S[2:end] + S[1:end-1])

    # Set rows n ∈ [2,Lτ] of tridiaonal matrix
    @views @. upper[2:end] = Δτ
    @views @. middle[2:end-1] = 4*Δτ
    @views @. lower[1:end-1] = Δτ

    # Calculate C″(0) using second order forward finite difference method
    C″0 = (2*C[1]-5*C[2]+4*C[3]-1*C[4])/Δτ^2

    # # Calculate C″(0) using first order forward finite difference method
    # C″0 = (1*C[1]-2*C[2]+1*C[3])/Δτ^2

    # Use C″(0) as boundary conditions and set top n=1 row of tridiagonal matrix
    middle[1] = 2
    upper[1] = 1
    Cprime[1] = 3*S[1] - C″0*Δτ/2

    # Calculate C″(β) using second order backward finite difference method
    C″β = (2*C[end]-5*C[end-1]+4*C[end-2]-1*C[end-3])/Δτ^2

    # # Calculate C″(β) using first order backward finite difference method
    # C″β = (1*C[end]-2*C[end-1]+1*C[end-2])/Δτ^2

    # Use C″(β) as boundary conditions and set bottom n=Lτ+1 row of tridiagonal matrix
    middle[end] = 2
    lower[end] = 1
    Cprime[end] = 3*S[end] + C″β*Δτ/2
    
    # construct tridiagonal matrix
    tri = Tridiagonal(lower, middle, upper)

    # solve C′(τ) = T⋅v(τ) where T is tridiagonal matrix
    ldiv!(tri, Cprime)

    # calculate b, c, and d coefficients
    @views @. d = (Cprime[2:end] + Cprime[1:end-1] - 2*S ) / Δτ^2
    @views @. c = (3*S - Cprime[2:end] - 2*Cprime[1:end-1]) / Δτ
    @views @. b = Cprime[1:end-1]

    return nothing
end