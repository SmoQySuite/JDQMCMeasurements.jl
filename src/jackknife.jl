@doc raw"""
    jackknife(
        g::Function,
        samples...;
        # KEYWORD ARGUMENTS
        bias_corrected = true,
        jackknife_sample_means = similar.(samples),
        jackknife_g = similar(first(samples))
    )

Propagate errors through the evaluation of a function `g` given the binned `samples`,
returning both the mean and error.
If the keyword argument `bias = true`, then the ``\mathcal{O}(1/N)`` bias is corrected.
The keyword arguments `jackknife_sample_means` and `jackknife_g` can be passed to avoid
temporary memory allocations.
"""
function jackknife(
    g::Function,
    samples...;
    # KEYWORD ARGUMENTS
    bias_corrected = true,
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
    ḡ = mean(jackknife_g)

    # calculate jackkife error
    Δg = sqrt( (N-1) * varm(jackknife_g, ḡ, corrected=false) )

    # correct O(1/N) bias, usually doesn't matter as error scales as O(1/sqrt(N))
    # and is typically much larger than the bias
    if bias_corrected
        Ḡ = g(x̄...)
        ḡ = N * Ḡ - (N-1) * ḡ
    end

    return ḡ, Δg
end