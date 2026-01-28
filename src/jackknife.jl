@doc raw"""
    jackknife(
        # ARGUMENTS
        g::Function,
        samples...;
        # KEYWORD ARGUMENTS
        bias_corrected = false,
        jackknife_sample_means = similar.(samples),
        jackknife_g = similar(first(samples)),
    )

Propagate errors through the evaluation of a function `g` given the binned `samples`,
returning both the mean and standard error as a tuple `(mean, std_error)`.

# Arguments
- `g::Function`: Function to evaluate, taking the same number of arguments as `samples`
- `samples...`: Vectors of individual observations (binned samples)

# Keyword Arguments
- `bias_corrected = false`: If `true`, apply ``\mathcal{O}(1/N)`` bias correction
- `jackknife_sample_means`: Preallocated arrays to avoid temporary allocations.
- `jackknife_g`: Preallocated array for jackknife function evaluations to avoid temporary allocations.

# Returns
- `mean`: By default, the full sample estimate `g(mean.(samples)...)`. If `bias_corrected=true`, returns the bias-corrected estimate.
- `std_error`: Jackknife standard error estimate.

# Notes
The standard error is computed from the jackknife replicates. By default, the returned mean
is the full sample estimate rather than the jackknife mean, as this is typically the best
point estimate from the available data.
"""
function jackknife(
    # ARGUMENTS
    g::Function,
    samples...;
    # KEYWORD ARGUMENTS
    bias_corrected = false,
    jackknife_sample_means = similar.(samples),
    jackknife_g = similar(first(samples)),
)

    # get sample size
    N = length(jackknife_g)

    # calculate mean of each input variable
    x̄ = map(mean, samples)

    # iterate over input variables
    for i in eachindex(samples)

        # get the mean of the current sample multiplied by sample size
        Nx̄_i = N * x̄[i]

        # get the vector to contain the jackknife sample means
        jackknife_sample_means_i = jackknife_sample_means[i]

        # get the input samples
        samples_i = samples[i]

        # calculate jackknife sample means
        @. jackknife_sample_means_i = (Nx̄_i - samples_i)/(N-1)
    end

    # evaluate the input function using the jackknife sample means
    @. jackknife_g = g(jackknife_sample_means...)

    # calculate jackknife mean
    ḡ_jackknife = mean(jackknife_g)

    # calculate jackknife error
    Δg = sqrt( (N-1) * varm(jackknife_g, ḡ_jackknife, corrected=false) )

    # calculate full sample mean
    ḡ_full = g(x̄...)

    # correct O(1/N) bias, usually doesn't matter as error scales as O(1/sqrt(N))
    # and is typically much larger than the bias
    if bias_corrected
        ḡ = N * ḡ_full - (N-1) * ḡ_jackknife
    else
        ḡ = ḡ_full
    end

    return ḡ, Δg
end