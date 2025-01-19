mutable struct ParamStats
    sum::Float32
    count::Int32
    min::Float32
    max::Float32
    sum_sq::Float32
end


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


function save_param_stats(param_stats; path = joinpath(@__DIR__, "..", "hyperopt"), file = "param_log_ppo.txt")
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


##############################################################################
# Get stats for all the hyperparameters. 
"""
Check if a parameter exists in the Hyperoptimizer params field
"""
function validate_parameter(hop::Hyperoptimizer, param::Symbol)::Int64
    param_idx = findfirst(x -> x == param, hop.params)
    if isnothing(param_idx)
        throw(ArgumentError("Parameter $param not found in hyperopt parameters"))
    end
    return param_idx
end

"""
Initialize empty dictionaries for each candidate value of a parameter
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
Collect results for each candidate value of the specified parameter
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
Main function to analize results for a specific parameter from a Hyperoptimizer object.
"""
function analize_hyperopt(hop::Hyperoptimizer, param::Symbol; correct::Bool = false)::Dict
    param_idx = validate_parameter(hop, param)
    result_dict = initialize_candidate_dicts(hop, param_idx)
    result_dict = collect_parameter_results(hop, param_idx, result_dict, correct = correct)
    return result_dict
end

"""
Analize results for a specific parameter across multiple Hyperoptimizer objects.
All results for each candidate value are collected into a single dictionary.
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
