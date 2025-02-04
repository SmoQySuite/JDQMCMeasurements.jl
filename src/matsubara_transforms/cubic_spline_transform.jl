@doc raw"""
    cubic_spline_τ_to_ωn!(
        Cn::AbstractVector{Complex{E}},
        Cτ::AbstractVector{T},
        β::E, Δτ::E;
        spline_type::String = "C2",
        M1::T = NaN,
        M2::T = NaN
    ) where {E<:AbstractFloat, T<:Number}

Calculate the Matsubara frequency representation
```math
\begin{align*}
C(\mathrm{i}\omega_n) = \int_0^\beta d\tau \ e^{\mathrm{i}\omega_n\tau} C(\tau)
\end{align*}
```
of an imaginary-time correlation function ``C(\tau)`` defined on a regular grid of ``L + 1``
imaginary-time points ``\tau \in \{ 0, \Delta\tau, 2\Delta\tau, \ldots, (\beta-\Delta\tau), \beta \}.``
This is done by fitting a cubic spline through the ``C(\tau)`` points, and then transforming the spline.
Here, the Matsubara correlations ``C(\mathrm{i}\omega_n)`` are written to the vector `Cn`, and
the imaginary-time correlations ``C(\tau)`` are stored in the vector `Cτ`. It is assumed that
`C[1]` and `C[end]` correspond to ``C(\tau = 0)`` and ``C(\tau = \beta)`` respectively. It is also
assumed that ``L = \beta/\Delta\tau.``

Note that the length of the vector `Cn` determines whether the correlation function is assumed to be fermionic or bosonic.

For fermoinic correlation functions, `mod(M, 2L) == 0`, where `M = length(Cn)` and `L = length(Cτ) - 1`.
Put another way, it must be that `length(Cn) == 2*n*(length(Cτ)-1)` for some positive integer `n ≥ 1`.
In this case the vector `Cn` contains Matsubara frequency correlation ``C(\mathrm{i}\omega_n)``
for ``n \in [-N, (N-1)]``, where `N = M÷2` and ``\omega_n = (2n+1)\pi/\beta``.

For bosonic correlation functions `mod(M+1, 2L) == 0`, where `M = length(Cn)` and `L = length(Cτ) - 1`.
Put another way, it must be that `length(Cn) == 2*n*(length(Cτ)-1) - 1` for some positive integer `n ≥ 1`.
In this case the vector `Cn` contains Matsubara frequency correlation ``C(\mathrm{i}\omega_n)``
for ``n \in [-N, N]``, where `N = (M-1)÷2` and ``\omega_n = 2n\pi/\beta``.

The `spline_type` argument can be set to `"C2"`, `"akima"`, or `"makima"` to specify the type of spline to use.
The default argument (and best for most cases) is `"C2"`, which refers to a C2 cubic spline with continuous first
and second derivatives. In this case, boundary conditions are imposed such that the second derivative of the spline
at `τ=0` and `τ=β` are set equal to the second derivative as calculated using a second-order forward and backward finite
difference respectively.

The `M1` and `M2` arguments can be used to specify the first and second moments of the spectral function if known.
By default the are set to `NaN`, which results in them being calculated internally based on the imaginary-time correlation
function data points and corresponding spline fit.
"""
function cubic_spline_τ_to_ωn!(
    Cn::AbstractVector{Complex{E}},
    Cτ::AbstractVector{T},
    β::E, Δτ::E;
    spline_type::String = "C2",
    M1::T = NaN,
    M2::T = NaN
) where {E<:AbstractFloat, T<:Number}

    # check inputs
    @assert spline_type ∈ ["C2", "akima", "makima"]
    @assert round(Int, β/Δτ) == length(Cτ)-1
    @assert iszero(mod(length(Cn), 2*(length(Cτ)-1))) || iszero(mod(length(Cn)+1, 2*(length(Cτ)-1)))

    # length of the imaginary-time axis
    L = length(Cτ)-1

    # total number of matsubara frequencies
    M = length(Cn)

    # vectors to contain spline coefficients
    b = zeros(T, L)
    c = zeros(T, L)
    d = zeros(T, L)

    # calculate C2 spline coefficients.
    if spline_type == "C2"
        
        c2_cubic_spline!(b, c, d, Cτ, Δτ)

        # label spline as C2 (continuous second derivative)
        C2 = true

    # calculate akima spline coefficients
    elseif spline_type == "akima"

        akima_spline!(b, c, d, Cτ, Δτ)

        # label spline as not C2 (discontinuous second derivative)
        C2 = false
    
    # calculate makima spline coefficients
    elseif spline_type == "makima"

        makima_spline!(b, c, d, Cτ, Δτ)

        # label spline as not C2 (discontinuous second derivative)
        C2 = false
    end

    # if imaginary-time correlations are fermionic
    if iszero(mod(M,2*L))

        # transform fermionic imaginary-time correlation to matsubara frequency
        fermionic_cubic_spline_τ_to_ωn!(
            Cn, Cτ, b, c, d, β, Δτ,
            M1 = M1, M2 = M2, C2 = C2
        )

    # if imaginary-time correlations are bosonic
    elseif iszero(mod(M+1,2*L))

        # transform bosonic imaginary-time correlation to matsubara frequency
        bosonic_cubic_spline_τ_to_ωn!(
            Cn, Cτ, b, c, d, β, Δτ,
            M1 = M1, M2 = M2, C2 = C2
        )
    end

    return nothing
end


# transform fermionic imaginary-time correlation function spline to matsubara frequency
function fermionic_cubic_spline_τ_to_ωn!(
    Cn::AbstractVector{Complex{E}},
    Cτ::AbstractVector{T},
    b::AbstractVector{T},
    c::AbstractVector{T},
    d::AbstractVector{T},
    β::E, Δτ::E;
    M1::T = NaN,
    M2::T = NaN,
    C2::Bool = false,
    tmp::AbstractVector{Complex{E}} = similar(Cn)
) where {E<:AbstractFloat, T<:Number}

    # length of the imaginary-time axis
    L = length(Cτ)-1

    # number of matsubara frequencies
    N = length(Cn) ÷ 2

    # check input vector sizes are consistent
    @assert round(Int, β/Δτ) == L
    @assert allequal(L′ -> (L′==L), (length(b), length(c), length(d)))
    @assert (N%L) == 0

    # calculate N/L
    NoL = N ÷ L

    # matsubara frequency indices
    n = -N:N-1

    # Get two views into temporary vector for positive and
    # negative matsubara frequencies
    Sp = @view tmp[N+1:2N] # for n ∈ [0,N-1]
    Sm = @view tmp[N:-1:1] # for n ∈ [-N,-1]

    # get reducude discretization constant
    Δτ′ = β/N

    # zero'th moment M₀ = C(β) + C(0)
    M0 = Cτ[end] + Cτ[1]

    # C(n) = -M₀/(iωₙ)
    @. Cn = -M0/(1im*ωn_fermi(n,β))

    # if first moment M₁ not defined
    if isnan(M1)
        # C′(β) = b[L] + 2⋅c[L]⋅Δτ + 3⋅d[L]⋅Δτ²
        C′β = b[L] + 2*c[L]*Δτ + 3*d[L]*Δτ^2
        # C′(0) = b[0]
        C′0 = b[1]
        # M₁ = -(C′(0) + C′(β))
        M1 = -(C′0 + C′β)
    end

    # C(n) = C(n) - M₁/(iωₙ)²
    @. Cn += -M1/(1im*ωn_fermi(n,β))^2

    # if C2 spline and M₂ = NaN, then calculate M₂
    if C2 && isnan(M2)
        # C″(β) = 2⋅c[L] + 6⋅d[L]⋅Δτ
        C″β = 2*c[L] + 6*d[L]*Δτ
        # C″(0) = 2⋅c[0]
        C″0 = 2*c[1]
        # M₂ = C″(0) + C″(β)
        M2 = C″0 + C″β
    end

    # if second moment M₂ not defined
    if isnan(M2)

        # iterate of oringinal time slice
        @inbounds for l in 1:L
            # get polynomial coefficients
            cl, dl = c[l], d[l]
            # iterate over sub-timesteps
            for j in 1:NoL
                # get index
                m = (l-1)*NoL + j
                # Cₗ″(τₘ) = 2⋅cₗ + 6⋅dₗ⋅j⋅Δτ′
                Sp[m] = 2*cl + 6*dl*j*Δτ′
            end
        end
        # Cₗ″(τₘ) = exp(i⋅m⋅π/N)⋅Cₗ″(τₘ)
        copyto!(Sm, Sp)
        @. Sp *= exp(1im*π*(0:N-1)/N)
        @. Sm *= exp(-1im*π*(0:N-1)/N)
        # fourier transform Cₗ″(τₘ)
        bfft!(Sp)
        fft!(Sm)
        # C(n) = C(n) + exp(iωₙ⋅Δτ′)/(iωₙ)³⋅F[Cₗ″(τₘ)]
        @views @. Cn += exp(1im*ωn_fermi(n,β)*Δτ′)/(1im*ωn_fermi(n,β))^3 * tmp

        # iterate of oringinal time slice
        @inbounds for l in 1:L
            # get polynomial coefficients
            cl, dl = c[l], d[l]
            # iterate over sub-timesteps
            for j in 1:NoL
                # get index
                m = (l-1)*NoL + j
                # Cₗ″(τₘ₋₁) = (2⋅cₗ + 6⋅dₗ⋅(j-1)⋅Δτ′)
                Sp[m] = 2*cl + 6*dl*(j-1)*Δτ′
            end
        end
        # Cₗ″(τₘ₋₁) = exp(i⋅m⋅π/N)⋅Cₗ″(τₘ₋₁)
        copyto!(Sm, Sp)
        @. Sp *= exp(1im*π*(0:N-1)/N)
        @. Sm *= exp(-1im*π*(0:N-1)/N)
        # fourier transform Cₗ″(τₘ₋₁)
        bfft!(Sp)
        fft!(Sm)
        # C(n) = C(n) - 1/(iωₙ)³⋅F[Cₗ″(τₘ₋₁)]
        @views @. Cn += -1/(1im*ωn_fermi(n,β))^3 * tmp
    else

        # C(n) = C(n) - M₂/(iωₙ)³
        @. Cn += -M2/(1im*ωn_fermi(n,β))^3
    end

    # iterate of oringinal time slice
    @inbounds for l in 1:L
        # get polynomial coefficients
        dl = d[l]
        # iterate over sub-timesteps
        for j in 1:NoL
            # get index
            m = (l-1)*NoL + j
            # Cₗ‴ = 6⋅dₗ
            Sp[m] = 6*dl
        end
    end
    # Cₗ‴ = exp(i⋅m⋅π/N)⋅Cₗ‴
    copyto!(Sm, Sp)
    @. Sp *= exp(1im*π*(0:N-1)/N)
    @. Sm *= exp(-1im*π*(0:N-1)/N)
    # fourier transform Cₗ‴
    bfft!(Sp)
    fft!(Sm)
    # C(n) = C(n) + (1-exp(iωₙ⋅Δτ′))/(iωₙ)⁴⋅F[Cₗ‴]
    @views @. Cn += (1-exp(1im*ωn_fermi(n,β)*Δτ′))/(1im*ωn_fermi(n,β))^4 * tmp

    return nothing
end


# transform bosonic imaginary-time correlation function spline to matsubara frequency
function bosonic_cubic_spline_τ_to_ωn!(
    Cn::AbstractVector{Complex{E}},
    Cτ::AbstractVector{T},
    b::AbstractVector{T},
    c::AbstractVector{T},
    d::AbstractVector{T},
    β::E, Δτ::E;
    M1::T = NaN,
    M2::T = NaN,
    C2::Bool = false,
    tmp = similar(Cn)
) where {E<:AbstractFloat, T<:Number}

    # length of the imaginary-time axis
    L = length(Cτ)-1

    # max matsubara frequency
    N = (length(Cn)-1)÷2

    # check input vector sizes are consistent
    @assert round(Int, β/Δτ) == L
    @assert allequal(L′ -> (L′==L), (length(b), length(c), length(d)))
    @assert (N+1) % L == 0

    # calculate (N+1)/L
    Np1oL = (N+1) ÷ L

    # matsubara frequency indices
    n = -N:N

    # Get two views into temporary vector for positive and
    # negative matsubara frequencies
    # example for N = 4:
    # index = [ (1,  2,  3,  4,  {5),  6,  7,  8,  9}]
    # n     = [(-4, -3, -2, -1,  {0),  1,  2,  3,  4}]
    Sp = @view tmp[N+1:2N+1] # for n ∈ [0,N]
    Sm = @view tmp[N+1:-1:1] # for n ∈ [-N,0]

    # get reducude discretization constant
    Δτ′ = β/(N+1)

    # zero'th moment M₀ = C(0) - C(β)
    M0 = Cτ[1] - Cτ[end]

    # C(n) = -M₀/(iωₙ)
    @. Cn = -M0/(1im*ωn_bose_reg(n,β))

    # if first moment M₁ not defined
    if isnan(M1)
        # C′(β) = b[L] + 2⋅c[L]⋅Δτ + 3⋅d[L]⋅Δτ²
        C′β = b[L] + 2*c[L]*Δτ + 3*d[L]*Δτ^2
        # C′(0) = b[0]
        C′0 = b[1]
        # M₁ = -(C′(0) - C′(β))
        M1 = -(C′0 - C′β)
    end

    # C(n) = C(n) - M₁/(iωₙ)²
    @. Cn += -M1/(1im*ωn_bose_reg(n,β))^2

    # if C2 spline and M₂ = NaN, then calculate M₂
    if C2 && isnan(M2)
        # C″(β) = 2⋅c[L] + 6⋅d[L]⋅Δτ
        C″β = 2*c[L] + 6*d[L]*Δτ
        # C″(0) = 2⋅c[0]
        C″0 = 2*c[1]
        # M₂ = C″(0) - C″(β)
        M2 = C″0 - C″β
    end

    # if second moment M₂ not defined
    if isnan(M2)

        # iterate of oringinal time slice
        @inbounds for l in 1:L
            # get polynomial coefficients
            cl, dl = c[l], d[l]
            # iterate over sub-timesteps
            for j in 1:Np1oL
                # get index
                m = (l-1)*Np1oL + j
                # Cₗ″(τₘ) = 2⋅cₗ + 6⋅dₗ⋅j⋅Δτ′
                Sp[m] = 2*cl + 6*dl*j*Δτ′
            end
        end
        copyto!(Sm, Sp)
        # fourier transform Cₗ″(τₘ)
        S0 = Sp[1] # record value overwritten by ω₀ = 0 term
        bfft!(Sp)
        Sm[1] = S0 # restore value overwritten by ω₀ = 0 term
        fft!(Sm)
        # C(n) = C(n) + exp(iωₙ⋅Δτ′)/(iωₙ)³⋅F[Cₗ″(τₘ)]
        @views @. Cn += exp(1im*ωn_bose_reg(n,β)*Δτ′)/(1im*ωn_bose_reg(n,β))^3 * tmp

        # iterate of oringinal time slice
        @inbounds for l in 1:L
            # get polynomial coefficients
            cl, dl = c[l], d[l]
            # iterate over sub-timesteps
            for j in 1:Np1oL
                # get index
                m = (l-1)*Np1oL + j
                # Cₗ″(τₘ₋₁) = (2⋅cₗ + 6⋅dₗ⋅(j-1)⋅Δτ′)
                Sp[m] = 2*cl + 6*dl*(j-1)*Δτ′
            end
        end
        copyto!(Sm, Sp)
        # fourier transform Cₗ″(τₘ₋₁)
        S0 = Sp[1] # record value overwritten by ω₀ = 0 term
        bfft!(Sp)
        Sm[1] = S0 # restore value overwritten by ω₀ = 0 term
        fft!(Sm)
        # C(n) = C(n) - 1/(iωₙ)³⋅F[Cₗ″(τₘ₋₁)]
        @views @. Cn += -1/(1im*ωn_bose_reg(n,β))^3 * tmp
    else

        # C(n) = C(n) - M₂/(iωₙ)³
        @. Cn += -M2/(1im*ωn_bose_reg(n,β))^3
    end

    # iterate of oringinal time slice
    @inbounds for l in 1:L
        # get polynomial coefficients
        dl = d[l]
        # iterate over sub-timesteps
        for j in 1:Np1oL
            # get index
            m = (l-1)*Np1oL + j
            # Cₗ‴ = 6⋅dₗ
            Sp[m] = 6*dl
        end
    end
    copyto!(Sm, Sp)
    # fourier transform Cₗ‴
    S0 = Sp[1] # record value overwritten by ω₀ = 0 term
    bfft!(Sp)
    Sm[1] = S0 # restore value overwritten by ω₀ = 0 term
    fft!(Sm)
    # C(n) = C(n) + (1-exp(iωₙ⋅Δτ′))/(iωₙ)⁴⋅F[Cₗ‴]
    @views @. Cn += (1-exp(1im*ωn_bose_reg(n,β)*Δτ′))/(1im*ωn_bose_reg(n,β))^4 * tmp

    # calculate C₀ (iωₙ=0) term by simply integrating the spline from τ=0 to β
    Cn[N+1] = 0.0
    for l in eachindex(b)
        # integrate spline from τ = (l-1)⋅Δτ to l⋅Δτ
        Cn[N+1] += Δτ*Cτ[l] + (Δτ^2/12)*(6*b[l] + Δτ*(4*c[l] + Δτ*3*d[l]))
    end

    return nothing
end


# calculate fermionic matsubara frequency
ωn_fermi(n,β) = (2n+1)*π/β

# regularized bosonic matsubara frequency
ωn_bose_reg(n,β) = iszero(n) ? 1.0 : ωn_bose(n,β)

# calculate bosonic matsubara frequency
ωn_bose(n,β) = 2*n*π/β