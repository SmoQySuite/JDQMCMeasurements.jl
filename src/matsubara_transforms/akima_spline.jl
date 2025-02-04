# calculate the coefficients for an akima spline
function akima_spline!(
    b::AbstractVector{T},
    c::AbstractVector{T},
    d::AbstractVector{T},
    Cτ::AbstractVector{T},
    Δτ::E;
    tol::E = 1e-12,
    m::AbstractVector{T} = zeros(T, length(Cτ)+3),
    s::AbstractVector{T} = zeros(T, length(Cτ))
) where {T<:Number, E<:AbstractFloat}

    # calculate first finite difference
    @views @. m[3:end-2] = (Cτ[2:end] - Cτ[1:end-1])/Δτ

    # parabolic extrapolation to get firt and last two endpoints
    m[2] = 2m[3] - m[4]
    m[1] = 2m[2] - m[3]
    m[end - 1] = 2m[end - 2] - m[end - 3]
    m[end] = 2m[end - 1] - m[end - 2]

    # calculate akima slope
    @inbounds for i in eachindex(s)
        # get appropriate index into m
        j = i + 2
        # calculate slope for real part
        δ2 = abs(real(m[j+1] - m[j]))
        δ1 = abs(real(m[j-1] - m[j-2]))
        if (δ1 + δ2) < tol
            s[i] = real(m[j] + m[j-1])/2
        else
            s[i] = real((δ2*m[j-1] + δ1*m[j])/(δ2 + δ1))
        end
        # calculate slope for imaginary part
        if T<:Complex
            δ2 = abs(imag(m[j+1] - m[j]))
            δ1 = abs(imag(m[j-1] - m[j-2]))
            if (δ1 + δ2) < tol
                s[i] += 1im*imag(m[j] + m[j-1])/2
            else
                s[i] += 1im*imag((δ2*m[j-1] + δ1*m[j])/(δ2 + δ1))
            end
        end
    end

    # set b coefficients
    @views @. b = s[1:end-1]

    # calculate c coeffients
    @views @. c = (3m[3:end-2] - 2s[1:end-1] - s[2:end]) / Δτ

    # calculate d coefficients
    @views @. d = (s[1:end-1] + s[2:end] - 2m[3:end-2]) / Δτ^2

    return nothing
end

# calculate the coefficients for a modified akima spline
function makima_spline!(
    b::AbstractVector{T},
    c::AbstractVector{T},
    d::AbstractVector{T},
    Cτ::AbstractVector{T},
    Δτ::E;
    tol::E = 1e-12,
    m::AbstractVector{T} = zeros(T, length(Cτ)+3),
    s::AbstractVector{T} = zeros(T, length(Cτ))
) where {T<:Number, E<:AbstractFloat}

    # calculate first finite difference
    @views @. m[3:end-2] = (Cτ[2:end] - Cτ[1:end-1])/Δτ

    # parabolic extrapolation to get firt and last two endpoints
    m[2] = 2m[3] - m[4]
    m[1] = 2m[2] - m[3]
    m[end - 1] = 2m[end - 2] - m[end - 3]
    m[end] = 2m[end - 1] - m[end - 2]

    # calculate akima slope
    @inbounds for i in eachindex(s)
        # get appropriate index into m
        j = i + 2
        # calculate slope for real part
        δ2 = abs(real(m[j+1] - m[j])) + abs(real(m[j+1] + m[j]))/2
        δ1 = abs(real(m[j-1] - m[j-2])) + abs(real(m[j-1] + m[j-2]))/2
        if (δ1 + δ2) < tol
            s[i] = real(m[j] + m[j-1])/2
        else
            s[i] = real((δ2*m[j-1] + δ1*m[j])/(δ2 + δ1))
        end
        # calculate slope for imaginary part
        if T<:Complex
            δ2 = abs(imag(m[j+1] - m[j])) + abs(imag(m[j+1] + m[j]))/2
            δ1 = abs(imag(m[j-1] - m[j-2])) + abs(imag(m[j-1] + m[j-2]))/2
            if (δ1 + δ2) < tol
                s[i] += 1im*imag(m[j] + m[j-1])/2
            else
                s[i] += 1im*imag((δ2*m[j-1] + δ1*m[j])/(δ2 + δ1))
            end
        end
    end

    # set b coefficients
    @views @. b = s[1:end-1]

    # calculate c coeffients
    @views @. c = (3m[3:end-2] - 2s[1:end-1] - s[2:end]) / Δτ

    # calculate d coefficients
    @views @. d = (s[1:end-1] + s[2:end] - 2m[3:end-2]) / Δτ^2

    return nothing
end