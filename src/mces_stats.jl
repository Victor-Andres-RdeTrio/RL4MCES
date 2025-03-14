
function mean_std(arr)
    println("μ = $(mean(arr)). σ = $(std(arr))")
    mean(arr), std(arr)
end


@inline function z_score(data::Real, mean::Real, std::Real)
    std = max(1f-8, std)
    (data .- mean)./std
end

###################################################33
# Online Stats

abstract type Normaliser end

"""
    mutable struct OnlineNorm <: Normaliser

Online normalization structure that incrementally computes means and variances.

# Fields
- `means::Vector{OnlineStat}`: Vector of online mean statistics for each state dimension.
- `var::Vector{OnlineStat}`: Vector of online variance statistics for each state dimension.
"""
mutable struct OnlineNorm <: Normaliser
    means::Vector{OnlineStat} 
    var::Vector{OnlineStat} 
end

"""
    OnlineNorm(n_states::Integer)

Construct an OnlineNorm with specified state dimensions.

# Arguments
- `n_states::Integer`: Number of state dimensions to track.

# Returns
- `OnlineNorm`: Initialized normalizer with zeroed mean and unitary variance statistics.
"""
function OnlineNorm(n_states::Integer)
    OnlineNorm(
        [Mean(Float32) for _ in 1:n_states], 
        [Variance(Float32) for _ in 1:n_states]
    )
end

"""
    update!(norm::OnlineNorm, state_vector::Vector)

Update online statistics with a new state observation.

# Arguments
- `norm::OnlineNorm`: Normalizer to update.
- `state_vector::Vector`: New state observation vector.

# Behavior
- Updates means and variances for each dimension using online statistics.
- Verifies dimensions match between normalizer and state vector.
"""
function update!(norm::OnlineNorm, state_vector::Vector)
    @assert length(norm.means) == length(state_vector) "Dimension Mismatch"
    
    StatsBase.fit!.(norm.means, state_vector)
    StatsBase.fit!.(norm.var, state_vector)
    nothing
end

"""
    z_score!(state_vector::Vector{<:AbstractFloat}, norm::OnlineNorm)

Normalize a state vector in-place using z-score normalization based on tracked statistics.

# Arguments
- `state_vector::Vector{<:AbstractFloat}`: State vector to normalize in-place.
- `norm::OnlineNorm`: Normalizer containing mean and variance statistics.

# Behavior
- Applies z-score normalization: (x - μ) / σ for each dimension.
- Verifies dimensions match between normalizer and state vector.
"""
function z_score!(state_vector::Vector{<:AbstractFloat}, norm::OnlineNorm)
    @assert length(norm.means) == length(state_vector) "Dimension Mismatch"
    μ = OnlineStats.value.(norm.means)
    σ = sqrt.(OnlineStats.value.(norm.var))

    for i in eachindex(state_vector)
        state_vector[i] = z_score(state_vector[i], μ[i], σ[i])
    end
    nothing
end

"""
    reset_stats!(norm::OnlineNorm)

Reset all statistical accumulators in an OnlineNorm instance.

# Arguments
- `norm::OnlineNorm`: Normalizer to reset.

# Behavior
- Sets all means to zero.
- Sets all variances to one.
- Resets observation counters to zero.
"""
function reset_stats!(norm::OnlineNorm)
    for i in eachindex(norm.means)
        norm.means[i].μ = 0f0
        norm.var[i].μ = 0f0
        norm.means[i].n = 0f0
        norm.var[i].n = 0f0
        norm.var[i].σ2 = 1f0
    end
end

####################################################################
# Fitting distributions

function fitdist(arr)
    # Define distributions to fit
    distributions = [
        Normal, 
        LogNormal, 
        Gamma, 
        Weibull, 
        Exponential,
        Pareto,
        InverseGaussian,
        Laplace,
        Rayleigh
    ]
    
    fits = Dict() 
    model_names = string.(distributions)
    
    # Distributions that can't handle zeros
    zero_sensitive = [LogNormal, Gamma, Weibull, Pareto, InverseGaussian, Frechet, Laplace, Rayleigh]
    
    # Add small epsilon to zero values if needed
    function handle_zeros(dist, data)
        if any(x -> x == 0, data) && dist in zero_sensitive
            epsilon = minimum(filter(x -> x > 0, data)) / 1000  # Small relative to smallest non-zero value
            return data .+ epsilon
        end
        return data
    end
    
    for (dist, name) in zip(distributions, model_names)
        try
            # Handle zeros if necessary
            processed_data = handle_zeros(dist, arr)
            
            # Try to fit the distribution
            fitted_dist = fit(dist, processed_data)
            fits[name] = fitted_dist
            
            # If zeros were handled, add a note
            if processed_data !== arr
                println("Note: Added small epsilon to zeros for $name distribution")
            end
            
        catch e
            # Handle specific error cases
            error_message = if isa(e, DomainError)
                "Error: Could not fit $name: Data outside valid domain"
            elseif isa(e, ArgumentError)
                "Error: Could not fit $name: Invalid arguments (possibly negative values)"
            else
                "Error: Could not fit $name: $(typeof(e))"
            end
            
            println(error_message)
            fits[name] = nothing
        end
    end
    
    return fits
end

function check_fit(arr, fit_model)
    # Get the log-likelihood of the fitted model
    loglike = loglikelihood(fit_model, arr)
    
    num_params = length(Distributions.params(fit_model))
    n = length(arr)
    
    # Compute AIC and BIC
    aic = -2 * loglike + 2 * num_params
    bic = -2 * loglike + num_params * log(n)
    
    # Perform Kolmogorov-Smirnov test
    # ks_test = ExactOneSampleKSTest(arr, fit_model)
    ks_test = HypothesisTests.ApproximateOneSampleKSTest(arr, fit_model)
    ks_pvalue = HypothesisTests.pvalue(ks_test)
    
    return (
        aic = aic, 
        bic = bic, 
        ks_pvalue = ks_pvalue, 
    )
end

function compare_fits(arr)
    fits = fitdist(arr)
    results = Dict{String, NamedTuple}()
    
    # Compute AIC, BIC, KS test
    for (name, fit_model) in fits
        isnothing(fit_model) && continue
        results[name] = check_fit(arr, fit_model)
    end
    
    # Sort results by AIC
    sorted_names = sort(collect(keys(results)), by=name -> results[name].aic)
    
    # Print results
    println("Model comparison:")
    for name in sorted_names
        metrics = results[name]
        println("Model: $name")
        println("  AIC: $(round(metrics.aic, digits=2))")
        println("  BIC: $(round(metrics.bic, digits=2))")
        # p > 0.05 is a good fit
        println("  KS p-value: $(round(metrics.ks_pvalue, digits=8))")
        # println("  Chi2 p-value: $(round(metrics.chi2_pvalue, digits=4))")
        println("----------------------------")
    end
    
    return results
end

########################################################
# Gamma KDE

"""
    gamma_kernel(x, xdata, h)

Compute the Gamma kernel for a single point x against all data points.
Altered slightly the implementation of KernelEstimators.jl
(https://github.com/panlanfeng/KernelEstimator.jl)

# Arguments
- `x`: Point at which to evaluate the kernel
- `xdata`: Vector of data points
- `h`: Bandwidth parameter
"""
function gamma_kernel(x, xdata, h)
    x ≤ 0 && return zeros(eltype(xdata), length(xdata))
    
    # Calculate ρₕ(x)
    rhob = x/h
    if x < 2h
        rhob = 0.25 * rhob^2 + 1.0
    end
    
    # Vectorized computation using broadcasting
    @. log(xdata) * (rhob - 1.0) - 
       rhob * log(h) - 
       SpecialFunctions.loggamma(rhob) - 
       xdata / h |> exp
end



"""
    gamma_kde(data, x_eval, h; weights=nothing)

Compute Kernel Density Estimation using the improved Gamma kernel.

# Arguments
- `data`: Vector of observations
- `x_eval`: Points at which to evaluate the density
- `h`: Bandwidth parameter
- `weights`: Optional weights for each observation (defaults to uniform weights)

# Returns
- Vector of estimated density values at evaluation points
"""
function gamma_kde(data, x_eval, h; weights=nothing)
    n = length(data)
    
    # Handle weights
    if isnothing(weights)
        w = fill(1/n, n)
    else
        w = weights ./ sum(weights)
    end
    
    # Compute density at each evaluation point
    [sum(gamma_kernel(x, data, h) .* w) for x in x_eval]
end



"""
    bootstrap_kde(data, x_eval, h; n_bootstrap=100)

Compute bootstrap confidence bounds for gamma kernel density estimation.

# Arguments
- `data`: Vector of observations
- `x_eval`: Points at which to evaluate the density
- `h`: Bandwidth parameter
- `n_bootstrap`: Number of bootstrap samples (default: 100)

# Returns
- Tuple of (density, lower_bound, upper_bound)
"""
function bootstrap_kde(data, x_eval, h; n_bootstrap=100)
    n = length(data)
    bootstrap_densities = zeros(n_bootstrap, length(x_eval))
    
    # Compute base density
    base_density = gamma_kde(data, x_eval, h)
    
    # Bootstrap iterations
    for i in 1:n_bootstrap
        # Sample with replacement
        boot_data = sample(data, n, replace=true)
        bootstrap_densities[i, :] = gamma_kde(boot_data, x_eval, h)
    end
    
    # Compute confidence bounds
    lower_bound = [quantile(bootstrap_densities[:, i], 0.025) for i in 1:length(x_eval)]
    upper_bound = [quantile(bootstrap_densities[:, i], 0.975) for i in 1:length(x_eval)]
    
    return base_density, lower_bound, upper_bound
end

"""
    plot_kde_with_bounds(data, x_eval, h; 
                        n_bootstrap=100, 
                        bins=20, 
                        title="Kernel Density Estimation with Gamma Kernel")

Create a plot of the KDE with confidence bounds and histogram.

# Arguments
- `data`: Vector of observations
- `x_eval`: Points at which to evaluate the density
- `h`: Bandwidth parameter
- `n_bootstrap`: Number of bootstrap samples
- `bins`: Number of histogram bins
- `title`: Plot title
"""
function plot_kde_with_bounds(data, x_eval, h; 
                            n_bootstrap=100, 
                            bins=20, 
                            title="KDE with Gamma Kernel")
    
    # Compute KDE and confidence bounds
    density, lower, upper = bootstrap_kde(data, x_eval, h; n_bootstrap=n_bootstrap)
    
    # Create plot
    p = plot(xlabel="x", ylabel="Density", title=title, legend=:topright)
    
    # Add histogram
    histogram!(p, data, 
              normalize=:pdf,
              bins=bins,
              alpha=0.3,
              color=:grey70,
              label="Data histogram")
    
    # Add KDE line and confidence bounds
    plot!(p, x_eval, density, 
          color=:violetred, 
          linewidth=2, 
          label="Gamma KDE")
    
    # Add confidence bounds
    plot!(p, x_eval, [lower upper], 
          fillrange=upper,
          fillalpha=0.2,
          fillcolor=:blue,
          linealpha=0,
          label="95% Confidence Band")
    
    # annotate!(p, minimum(x_eval), 
    #          0.95*maximum(density),
    #          text("n=$(length(data))\nbandwidth=$(round(h, digits=3))", 
    #              :left, 8, :black))
    
    # Add grid
    plot!(p, grid=false, gridalpha=0.15)
    display(p)
    
    return p
end

function calculate_binomial_stats(
    data_arrays::Vector{Vector{T}},
    threshold::Real,
    labels::Union{Vector{String}, Nothing} = nothing
) where {T}
    
    if isnothing(labels)
        labels = ["Series $i" for i in 1:length(data_arrays)]
    end
    
    stats = []
    for (i, data) in enumerate(data_arrays)
        n = length(data)
        successes = filter(x -> x >= threshold, data)
        p_success = length(successes) / n
        push!(stats, (label=labels[i], n=n, p=p_success))
    end
    
    return stats
end

"""
    bootstrap_ci(data::Vector{<:Real}; n_bootstrap::Int=10000, confidence_level::Real=0.95)

Calculate confidence interval using bootstrapping, which makes no assumptions about
the underlying distribution. Appropriate for large samples (n>30).

Parameters:
- data: Vector of numerical values (simulation outputs)
- n_bootstrap: Number of bootstrap samples to generate (default: 10000)
- confidence_level: Desired confidence level (default: 0.95 for 95% CI)

Returns:
- Tuple containing (mean, lower_bound, upper_bound)

Example:
```julia
results = [10.2, 10.5, 9.8, 10.1, 10.3]
mean_val, ci_lower, ci_upper = bootstrap_ci(results)
```
"""
function bootstrap_ci(data::Vector{<:Real}; 
                     n_bootstrap::Int=10000, 
                     confidence_level::Real=0.95)
    original_mean = mean(data)
    
    n = length(data)
    bootstrap_means = zeros(n_bootstrap)
    
    for i in 1:n_bootstrap
        bootstrap_sample = sample(data, n, replace=true)
        bootstrap_means[i] = mean(bootstrap_sample)
    end

    α = 1 - confidence_level
    sort!(bootstrap_means)
    lower_index = round(Int, n_bootstrap * α/2)
    upper_index = round(Int, n_bootstrap * (1 - α/2))
    
    ci_lower = bootstrap_means[lower_index]
    ci_upper = bootstrap_means[upper_index]
    
    # Calculate standard error from bootstrap distribution
    bootstrap_std = std(bootstrap_means) # Standard deviation the means obtained from resampling
    
    return original_mean, ci_lower, ci_upper, bootstrap_std
end

"""
    print_bootstrap_results(data::Vector{<:Real}; 
                          n_bootstrap::Int=10000, 
                          confidence_level::Real=0.95)

Print the bootstrap confidence interval results in a readable format.
"""
function print_bootstrap_ci_mean(data::Vector{<:Real}; 
                               n_bootstrap::Int=10000, 
                               confidence_level::Real=0.95,
                               io::IO = stdout)
    mean_val, ci_lower, ci_upper, std_error = bootstrap_ci(data; 
                                                          n_bootstrap=n_bootstrap, 
                                                          confidence_level=confidence_level)
    
    println(io, "Bootstrap results from $(length(data)) samples ($(n_bootstrap) resamples):")
    println(io, "Mean: $(round(mean_val, digits=4))")
    println(io, "95% CI: [$(round(ci_lower, digits=4)), $(round(ci_upper, digits=4))]")
    println(io, "CI Width: ±$(round((ci_upper - ci_lower)/2, digits=4))")
    println(io, "Standard Error: $(round(std_error, digits=4))")
end


@info "Stats are Online"