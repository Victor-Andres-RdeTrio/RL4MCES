# This file contains mostly ways of processing the objects produced during hyperparameter optimisation, 
# which is performed with the function `threaded_hyperopt`, found in the file "policy_update\\policy.jl".

"""
    ParamStats

A structure to store statistics about hyperparameter values.

# Fields
- `sum::Float32`: Sum of all results for this parameter value
- `count::Int32`: Number of times this parameter value was used
- `min::Float32`: Minimum result value obtained with this parameter value
- `max::Float32`: Maximum result value obtained with this parameter value
- `sum_sq::Float32`: Sum of squares of results, used for calculating variance/std
"""
mutable struct ParamStats
    sum::Float32
    count::Int32
    min::Float32
    max::Float32
    sum_sq::Float32
end

"""
    top_hyperopt(ho, top = 5)

Extract the top performing hyperparameter configurations from a hyperoptimization object.

# Arguments
- `ho`: Hyperoptimizer object containing results
- `top::Int`: Number of top results to extract (default: 5)

# Returns
- Tuple of (top_hyperparameters, top_results) where smaller results are considered better, as the hyperoptimization aims at minimising. 
"""
function top_hyperopt(ho, top = 5)
    top = min(length(ho.results), top)
    sorted_res =  sort(deepcopy(ho.results))
    # The minimum values are the best ones
    top_res = sorted_res[1:top]
    top_hyper = []
    for i in 1:top
        ind = findfirst(a -> a == sorted_res[i], ho.results)
        if isa(ind, Int)
            push!(top_hyper, ho.history[ind])
        else
            push!(top_hyper, Nothing)
        end
    end
    top_hyper, top_res
end

"""
    worst_hyperopt(ho, bottom = 5)

Extract the worst performing hyperparameter configurations from a hyperoptimization object.

# Arguments
- `ho`: Hyperoptimizer object containing results
- `bottom::Int`: Number of worst results to extract (default: 5)

# Returns
- Tuple of (bottom_hyperparameters, bottom_results) where larger results are considered worse, as the hyperoptimization aims at minimising
"""

function worst_hyperopt(ho, bottom = 5)
    bottom = min(length(ho.results), bottom)
    sorted_res = sort(deepcopy(ho.results), rev=true)  # Sorting in descending order
    # The maximum values are the worst ones
    bottom_res = sorted_res[1:bottom]
    bottom_hyper = []
    for i in 1:bottom
        ind = findfirst(a -> a == sorted_res[i], ho.results)
        if isa(ind, Int)
            push!(bottom_hyper, ho.history[ind])
        else
            push!(bottom_hyper, Nothing)
        end
    end
    bottom_hyper, bottom_res
end

"""
    top_hyperopt_bson(ho_bson, top = 6)

Extract the top performing hyperparameter configurations from a BSON-loaded hyperoptimization object.

# Arguments
- `ho_bson`: BSON dictionary containing hyperoptimization results
- `top::Int`: Number of top results to extract (default: 6)

# Returns
- Tuple of (top_hyperparameters, top_results) excluding the first result
"""
function top_hyperopt_bson(ho_bson, top = 6)
    top = min(length(ho_bson[:results]), top)
    sorted_res =  sort(deepcopy(ho_bson[:results]))
    # The minimum values are the best ones
    top_res = sorted_res[2:top]
    top_hyper = []
    for i in 2:top
        ind = findfirst(a -> a == sorted_res[i], ho_bson[:results])
        push!(top_hyper, ho_bson[:history][ind])
    end
    top_hyper, top_res
end

"""
    save_hyperopt(ho; path = joinpath(@__DIR__,"..","hyperopt\\"), 
                  file = "tune_log.txt", policy_type::String = "?", 
                  n_seeds = "?", extra = nothing)

Save hyperoptimization results to files, including a log file, parameter statistics, and plots.

# Arguments
- `ho`: Hyperoptimizer object containing results
- `path::String`: Directory path to save output files (default: "../hyperopt/")
- `file::String`: Name of the log file (default: "tune_log.txt")
- `policy_type::String`: Type of policy used (default: "?")
- `n_seeds`: Number of seeds per sample (default: "?")
- `extra`: Additional information to include in the log (default: nothing)

# Effects
1. Writes detailed hyperparameter results to a log file
2. Generates and saves a plot of hyperoptimization results
3. Saves the hyperoptimizer object as a JLD2 file
4. Calculates and saves parameter statistics
"""
function save_hyperopt(ho; path = joinpath(@__DIR__,"..","hyperopt\\"), 
    file = "tune_log.txt", policy_type::String = "?", n_seeds = "?", extra = nothing)
    ref = rand(1:100000)

    open(joinpath(path,file), "a") do io
        println(io, "\n", "="^40)
        println(io, "Date & Time: $(Dates.now())")
        !isnothing(extra) && println(io, extra)
        println(io, "Policy Type: $(policy_type)")
        println(io, "Samples: $(ho.iterations)")
        println(io, "Nº of seeds per sample: $(n_seeds)")
        println(io, "Threads: ", nthreads())
        # println(io, "Cost Function Weights: ", params)
        println(io, "Hyperparameters: $(ho.params)")
        println(io, "Candidates: $(ho.candidates)")
        println(io, "Plot: $ref")
        
        println(io,"\n", "-"^40)
        println(io, "Top 5 Hyperparameter Configurations:")
        top_hyper, top_res = top_hyperopt(ho, 5)
        for i in eachindex(top_hyper)
            # println(io, "-"^40)
            println(io, "Result -> $(top_res[i])")
            println(io, "Hyperparameters: $(top_hyper[i])")
        end
        println(io, "-"^40, "\n")
        
        println(io, "-"^40)
        println(io, "Worst 5 Hyperparameter Configurations:")
        worst_hyper, worst_res = worst_hyperopt(ho, 5)
        for i in eachindex(worst_hyper)
            # println(io, "-"^40)
            println(io, "Result -> $(worst_res[i])")
            println(io, "Hyperparameters: $(worst_hyper[i])")
        end
        println(io, "="^40, "\n")
    end
    # Store plot of the hyperopt samples
    p = plot(ho)
    plot!(p, dpi = 800)
    Plots.savefig(p, joinpath(path, "Plot_$ref"))

    # Store hyperopt object
    jldsave(joinpath(path,"ho_$ref.jld2"), ho = ho)

    stats = stats_by_parameter_value(ho)
    save_param_stats(stats; path = path, file = "params_" * file)

    println("Saved Output")
end

"""
    stats_by_parameter_value(ho)

Calculate statistics for each parameter value across all hyperoptimization runs.

# Arguments
- `ho`: Hyperoptimizer object containing history and results

# Returns
- Dictionary mapping parameter names to dictionaries of parameter values,
  where each value contains statistics (:avg, :min, :max, :std, :count), see `ParamStats`.
"""
function stats_by_parameter_value(ho)
    history = ho.history
    results = ho.results
    param_names = ho.params
    
    param_stats = Dict{Symbol, Dict{Any, ParamStats}}()

    for param in param_names
        param_stats[param] = Dict{Any, ParamStats}()
    end

    for (params, result) in zip(history, results)
        result = Float32(result) 
        for (i, param_name) in enumerate(param_names)
            param_value = params[i]
            
            if haskey(param_stats[param_name], param_value)
                stats = param_stats[param_name][param_value]
                stats.sum += result
                stats.count += 1
                stats.min = min(stats.min, result)
                stats.max = max(stats.max, result)
                stats.sum_sq += result^2
            else
                param_stats[param_name][param_value] = ParamStats(result, 1, result, result, result^2)
            end
        end
    end

    final_stats = Dict{Symbol, Dict{Any, Dict{Symbol, Float32}}}()
    
    for param_name in param_names
        final_stats[param_name] = Dict{Any, Dict{Symbol, Float32}}()
        for (param_value, stats) in param_stats[param_name]
            avg = stats.sum / stats.count
            variance = (stats.sum_sq / stats.count) - avg^2
            std = sqrt(variance)  
            
            final_stats[param_name][param_value] = Dict{Symbol, Float32}(
                :avg => avg,
                :min => stats.min,
                :max => stats.max,
                :std => std,
                :count => stats.count
            )
        end
    end

    return final_stats
end

"""
    save_param_stats(param_stats; path = joinpath(@__DIR__, ".."), 
                    file = "param_log.txt")

Save parameter statistics to a formatted text file.

# Arguments
- `param_stats`: Dictionary of parameter statistics as returned by `stats_by_parameter_value`
- `path::String`: Directory path to save output file (default: parent Directory)
- `file::String`: Name of the parameter log file (default: "param_log.txt")

# Effects
- Writes a formatted table of parameter statistics to the specified file
- Each parameter's values are listed with their average, minimum, maximum, standard deviation, and count
"""
function save_param_stats(param_stats; path = joinpath(@__DIR__, ".."), file = "param_log.txt")
    open(joinpath(path, file), "a") do io
        println(io, "\n", "="^100)
        println(io, "Date & Time: $(Dates.now())")
        println(io, "Hyperparameter Statistics:")
        
        for (param, values) in param_stats
            println(io, "\nParameter: $param")
            println(io, "-"^55)
            println(io, "Value  | Average | Minimum | Maximum | Std Dev | Count")
            println(io, "-"^55)

            sorted_stats = sort(values)  # Sort by parameter value
            for (value, stats) in sorted_stats
                println(io, "$(rpad(value, 6)) | $(rpad(round(stats[:avg], digits=4), 7)) | " *
                            "$(rpad(round(stats[:min], digits=4), 7)) | $(rpad(round(stats[:max], digits=4), 7)) | " *
                            "$(rpad(round(stats[:std], digits=4), 7)) | $(stats[:count])"
                        )
            end
            println(io, "-"^55)
        end
        
        println(io, "="^100, "\n")
    end

    println("Saved parameter statistics to $file")
end
#############################################################################
# Quick stats for a collection of Hooks

"""
    hook_params(hooks::AbstractHook...; grid_weight::Real = 1.0, ev_weight::Real = 1.0, 
                init_proj_weight::Real = 1.0, op_proj_weight::Real = 1.0)

Analyze and summarize reward component statistics across multiple hooks with optional weighting.

# Arguments
- `hooks::AbstractHook...`: One or more hook objects containing reward dissection data.
- `grid_weight::Real = 1.0`: Weight factor for grid costs.
- `ev_weight::Real = 1.0`: Weight factor for EV costs.
- `init_proj_weight::Real = 1.0`: Weight factor for initial projection costs.
- `op_proj_weight::Real = 1.0`: Weight factor for operational projection costs.

# Returns
- Tuple of eight values: average mean and standard deviation for each of the four cost components.

# Details
1. Extracts and applies weights to each cost component from each hook.
2. Calculates and prints statistics for each component (mean and standard deviation).
3. For EV costs, only considers non-zero values in statistics.
4. Returns the aggregate statistics for further processing.
"""
function hook_params(hooks::AbstractHook...; grid_weight::Real = 1.0, ev_weight::Real = 1.0, init_proj_weight::Real = 1.0, op_proj_weight::Real = 1.0)
    grid_means = Float32[]
    grid_stds = Float32[]
    ev_means = Float32[]
    ev_stds = Float32[]
    init_proj_means = Float32[]
    init_proj_stds = Float32[]
    op_proj_means = Float32[]
    op_proj_stds = Float32[]
    
    for hook in hooks
        dissection = hook.reward_dissect
        grid_cost = dissection["Grid"] .* grid_weight
        ev_cost = dissection["EV"] .* ev_weight
        init_proj_cost = dissection["Initial Projection"] .* init_proj_weight
        op_proj_cost = dissection["Op. Projection"] .* op_proj_weight

        grid_mean = mean(grid_cost)
        grid_std = std(grid_cost)

        non_zero_ev_cost = filter(x -> x != 0, ev_cost)
        ev_mean = isempty(non_zero_ev_cost) ? 0.0 : mean(non_zero_ev_cost)
        ev_std = isempty(non_zero_ev_cost) ? 0.0 : std(non_zero_ev_cost) 

        init_proj_mean = mean(init_proj_cost)
        init_proj_std = std(init_proj_cost)
        
        op_proj_mean = mean(op_proj_cost)
        op_proj_std = std(op_proj_cost)

        push!(grid_means, grid_mean)
        push!(grid_stds, grid_std)
        push!(ev_means, ev_mean)
        push!(ev_stds, ev_std)
        push!(init_proj_means, init_proj_mean)
        push!(init_proj_stds, init_proj_std)
        push!(op_proj_means, op_proj_mean)
        push!(op_proj_stds, op_proj_std)
    end

    avg_grid_mean = mean(grid_means)
    avg_grid_std = mean(grid_stds)
    avg_ev_mean = mean(ev_means)
    avg_ev_std = mean(ev_stds)
    avg_init_proj_mean = mean(init_proj_means)
    avg_init_proj_std = mean(init_proj_stds)
    avg_op_proj_mean = mean(op_proj_means)
    avg_op_proj_std = mean(op_proj_stds)

    println("Grid Cost Statistics (Average):")
    println("μ: ", round(avg_grid_mean, digits=3))
    println("σ: ", round(avg_grid_std, digits=3))

    println("\nEV Cost Statistics (Average):")
    println("μ (non-zero values): ", round(avg_ev_mean, digits=3))
    println("σ (non-zero values): ", round(avg_ev_std, digits=3))

    println("\nInitial Projection Cost Statistics (Average):")
    println("μ: ", round(avg_init_proj_mean, digits=3))
    println("σ: ", round(avg_init_proj_std, digits=3))
    
    println("\nOperational Projection Cost Statistics (Average):")
    println("μ: ", round(avg_op_proj_mean, digits=3))
    println("σ: ", round(avg_op_proj_std, digits=3))

    return avg_grid_mean, avg_grid_std, avg_ev_mean, avg_ev_std, avg_init_proj_mean, avg_init_proj_std, avg_op_proj_mean, avg_op_proj_std
end



##############################################################################
# Get stats for all the hyperparameters. 
"""
    validate_parameter(hop::Hyperoptimizer, param::Symbol)::Int64

Check if a parameter exists in the Hyperoptimizer params field and return its index.

# Arguments
- `hop::Hyperoptimizer`: Hyperoptimizer object
- `param::Symbol`: Parameter name to validate

# Returns
- `Int64`: Index of the parameter in the params list

# Throws
- `ArgumentError`: If the parameter is not found in the hyperopt parameters
"""
function validate_parameter(hop::Hyperoptimizer, param::Symbol)::Int64
    param_idx = findfirst(x -> x == param, hop.params)
    if isnothing(param_idx)
        throw(ArgumentError("Parameter $param not found in hyperopt parameters"))
    end
    return param_idx
end

"""
    initialize_candidate_dicts(hop::Hyperoptimizer, param_idx::Int64)::Dict

Initialize empty dictionaries for each candidate value of a parameter.

# Arguments
- `hop::Hyperoptimizer`: Hyperoptimizer object
- `param_idx::Int64`: Index of the parameter in the params list

# Returns
- `Dict`: Dictionary mapping each candidate value to an empty vector of Float32
"""
function initialize_candidate_dicts(hop::Hyperoptimizer, param_idx::Int64)::Dict
    candidate_values = hop.candidates[param_idx]
    result_dict = Dict{Float64, Vector{Float32}}()
    
    for candidate in candidate_values
        result_dict[candidate] = Float32[]
    end
    
    return result_dict
end


"""
    collect_parameter_results(hop::Hyperoptimizer, param_idx::Int64, 
                            result_dict::Dict; correct::Bool = false)::Dict

Collects and groups performance results for each candidate value of a specified parameter based on the hyperoptimizer’s history and outcomes.

# Arguments
- `hop::Hyperoptimizer`: The hyperoptimizer instance containing optimization history (`hop.history`) and corresponding results (`hop.results`). Each history entry is an array of candidate parameter values.
- `param_idx::Int64`: The index of the parameter in the candidate parameters array to be used as the key in the results dictionary.
- `result_dict::Dict`: A pre-initialized dictionary (typically created using `initialize_candidate_dicts`) where each key corresponds to a candidate parameter value and the associated value is an array that will store the results.
- `correct::Bool`: A flag indicating whether a correction factor should be applied to the results. When set to `true`, each result is multiplied by -0.01. This correction reverses an earlier scaling where results were multiplied by -100, a transformation often used to convert a maximization objective into a minimization one or for numerical stability during optimization. This factor can only be modified within the function definition. 

# Returns
- `Dict`: The updated dictionary mapping each candidate parameter value to an array of results, with the correction applied if requested.
"""
function collect_parameter_results(
    hop::Hyperoptimizer, param_idx::Int64, 
    result_dict::Dict; correct::Bool = false
    )::Dict
    for (hist_entry, result) in zip(hop.history, hop.results)
        param_value = hist_entry[param_idx]
        result_float = Float32(result)
        if correct 
            result_float *= -0.01f0 # For Hyperoptimizer results were multiplied by -100. 
        end
        push!(result_dict[param_value], result_float)
    end
    return result_dict
end

"""
    analize_hyperopt(hop::Hyperoptimizer, param::Symbol; correct::Bool = false)::Dict

Analyze results for a specific parameter from a Hyperoptimizer object.

# Arguments
- `hop::Hyperoptimizer`: Hyperoptimizer object
- `param::Symbol`: Parameter name to analyze
- `correct::Bool`: Whether to apply a correction factor to results (default: false)

# Returns
- `Dict`: Dictionary mapping each parameter value to its collected results
"""
function analize_hyperopt(hop::Hyperoptimizer, param::Symbol; correct::Bool = false)::Dict
    param_idx = validate_parameter(hop, param)
    result_dict = initialize_candidate_dicts(hop, param_idx)
    result_dict = collect_parameter_results(hop, param_idx, result_dict, correct = correct)
    return result_dict
end

"""
    analize_hyperopt(hops::Vector, param::Symbol; correct::Bool = false)::Dict

Analyze results for a specific parameter across multiple Hyperoptimizer objects.

# Arguments
- `hops::Vector`: Vector of Hyperoptimizer objects
- `param::Symbol`: Parameter name to analyze
- `correct::Bool`: Whether to apply a correction factor to results (default: false)

# Returns
- `Dict`: Dictionary mapping each parameter value to its collected results across all optimizers

# Notes
- All results for each candidate value are collected into a single dictionary
- If different candidate values are found across optimizers, they are merged
"""
function analize_hyperopt(hops::Vector, param::Symbol; correct::Bool = false)::Dict
    param_indices = [validate_parameter(hop, param) for hop in hops]
    
    candidate_dicts = [initialize_candidate_dicts(hop, idx) for (hop, idx) in zip(hops, param_indices)]
    
    result_dict = candidate_dicts[1]
    if !all(dict -> dict == result_dict, candidate_dicts)
        @info "Hyperparameter with different candidates were found for param $param"
        result_dict = reduce(merge!, candidate_dicts)
    end
    
    for (hop, param_idx) in zip(hops, param_indices)
        result_dict = collect_parameter_results(hop, param_idx, result_dict, correct=correct)
    end
    
    return result_dict
end


# hop.results -> 600 element Vector{Any}: [-0.24001f0, -0.0f0, 0.0f0, ..., -0.01f0, -0.060000002f0, -0.0f0, -0.0f0]
# hop.params -> (:disc_rate, :gae, :init_std, :w_e_loss, :adam_a, :adam_c, :upd_freq, :actor_width, :critic_width, :actor_arch, :critic_arch, :actor_activ, :critic_activ, :clip_coef, :years, :features, :params, :rew_shape)
# hop.history -> 600-element Vector{Any}: (0.6f0, 0.0f0, 2.0f0, 0.0f0, 3.0f-5, 0.001f0, 2, 16, 32, 3, 7, 2, 2, 0.0f0, 3, 2, 1, 8) ; (0.99f0, 0.0f0, 0.5f0, 0.0f0, 0.0001f0, 0.0001f0, 1, 16, 64, 5, 9, 2, 1, 0.0f0, 3, 8, 1, 3); (0.8f0, 0.0f0, 2.0f0, 0.0f0, 3.0f-5, 0.0003f0, 2, 512, 32, 6, 11, 2, 2, 0.0f0, 6, 5, 2, 4); (0.999f0, 0.0f0, 0.5f0, 0.0f0, 0.001f0, 0.003f0, 1, 256, 32, 12, 8, 1, 2, 0.0f0, 3, 8, 1, 3)
# hop.candidates ->  (Any[0.6f0, 0.8f0, 0.9f0, 0.95f0, 0.99f0, 0.999f0], Float32[0.0], Any[0.1f0, 0.5f0, 1.0f0, 2.0f0], Float32[0.0], Any[3.0f-5, 0.0001f0, 0.0003f0, 0.001f0], Any[3.0f-5, 0.0001f0, 0.0003f0, 0.001f0, 0.003f0], Any[1, 2, 3], Any[16, 32, 64, 128, 256, 512], Any[16, 32, 64, 128, 256, 512], Any[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], Any[1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12], Any[1, 2], Any[1, 2], Float32[0.0], Any[3, 6], Any[1, 2, 3, 4, 5, 6, 7, 8, 9], Any[1, 2, 3], Any[2, 3, 4, 5, 6, 7, 8, 9])
