####################################################################################################
# Special functions for the MCES environment:
# valid_actions, limit_battery_power, limit_hp_power, perform_safety_checks

@inline function avg_entropy(stds::Vector)
    mean(0.5f0*log.(2f0*pi*exp(1f0)*(stds.^2f0)))
end

@inline function inv_softplus(y)
    return log(exp(y) - 1)
end

function q_norm(x::AbstractArray; min_quantile=0.0, max_quantile=0.99, clip=false)
    min_val = Statistics.quantile(x, min_quantile)
    max_val = Statistics.quantile(x, max_quantile)
    
    normalized = (x .- min_val) ./ (max_val - min_val)
    
    if clip
        return clamp.(normalized, 0, 1)
    else
        return normalized
    end
end

# Function to apply smoothing if necessary
function apply_smoothing(arr::Vector, smooth::Int)
    return smooth > 0 ? Smoothing.binomial(arr, smooth) : arr
end

"""
    clamp_with_amount(x, lo, hi; abs = false) -> (clamped_x, clamped_amount)

Clamp the input `x` to the range `[lo, hi]` and return both the clamped value and the amount by which the input was clamped.

# Arguments
- `x`: The input value to be clamped.
- `lo`: The lower bound of the range.
- `hi`: The upper bound of the range.
- `abs::Bool`: A keyword argument. If `true`, return the absolute value of the clamped amount. Default is `false`.

# Returns
- `clamped_x`: The value of `x` clamped to the range `[lo, hi]`.
- `clamped_amount`: The amount by which `x` was clamped. If `abs` is `true`, this value is returned as an absolute value.

# Example
```julia
clamped_x, clamped_amount = clamp_with_amount(15, 10, 12)
# clamped_x will be 12
# clamped_amount will be 3

clamped_x, clamped_amount = clamp_with_amount(5, 10, 12, abs=true)
# clamped_x will be 10
# clamped_amount will be 5
```
"""
@inline function clamp_with_amount(x, lo, hi; abs::Bool = false)
    clamped_x = clamp(x, lo, hi)
    clamped_amount = abs ? Base.abs(x - clamped_x) : x - clamped_x
    return clamped_x, clamped_amount
end

function check_required_keys(dict::Dict, required_keys::Vector{String})
    missing_keys = filter(k -> !haskey(dict, k), required_keys)
    if !isempty(missing_keys)
        throw(KeyError("Missing required keys: $(join(missing_keys, ", "))"))
    end
end

function print_last_values(dict::Dict, shift::Int = 0)
    for (key, value) in dict
        if value isa AbstractArray && !isempty(value)
            if shift == 0
                println("$key: $(round(value[end], digits = 6))")
            else
                println("$key: $(round.(value[end-shift:end], digits = 6))")
            end
        elseif value isa AbstractString && !isempty(value)
            println("$key: $(value[end:end])")
        else
            println("Failed")
        end
    end
end

function find_dif(a::Vector , b::Vector ; first::Bool = true)
    length(a) == length(b) ? nothing : return println("Different Length")
    a = Float32.(a)
    b = Float32.(b)
    c = abs.(a - b)
    return first ? findfirst(c-> c > 0, c) : findall(c-> c > 0, c)
end

function to_dict(exog::Exogenous_BatchCollection)
    dict = Dict{String, Any}()

    for field in fieldnames(typeof(exog))
        dict[string(field)] = getfield(exog, field)
    end
    
    return dict
end

function print_dict_to_file(dict::Dict, filename::String; title="Training Information")
    open(filename, "a") do file
        write(file, "=============================\n")
        write(file, "$title \n")

        # Recursive function to handle nested dictionaries
        function print_ordered_dict(d::Dict, indent::String="  ")
            for key in sort(collect(keys(d)))  # Sort keys alphabetically
                value = d[key]
                
                if isa(value, Dict)
                    write(file, "$indent ====== \n")
                    write(file, "$indent$key : \n")
                    print_ordered_dict(value, indent * "  ")  # Increase indent for nested dictionary
                    write(file, "$indent ======\n\n")
                else
                    write(file, "$indent$key => $value \n")
                end
            end
        end

        # Print the main dictionary
        print_ordered_dict(dict)
        write(file, "=============================\n\n")
    end
end

function write_best_seeds(array::Vector, filename::String; append = false)
    top_10_indices = partialsortperm(array, 1:min(10, length(array)) , rev=true)

    open(filename, append ? "a" : "w") do file
        write(file, "Best 10 seeds are:\n")
        for i in top_10_indices
            write(file, "$(round(array[i], digits = 4)) -> $i \n" )
        end
    end
end
##################################################################################


"""
    limit_battery_power(power, battery::Battery, Δt)

Limits the power output of a battery device to safe levels based on its desing characteristics. 
Also guarantees that the SoC min and max levels are respected. 
Takes in Power in kW, turns it into W, and then outputs kW. 
The Coulombic efficiency will be <1 when discharging and =1 when charging.

# Args:
    * `power`: Proposed power level for the battery device.
    * `battery`: Battery device object containing specifications and state information.
    * `Δt`: Time step size.

# Returns:
    The limited power output for the battery device.
"""
function limit_battery_power(power_kW, bat::Battery, Δt, simple::Bool = false)
    if simple
        return clamp(power_kW, - bat.p_max, bat.p_max)
    end

    safety_factor = 1f0 # To reduce the width between the limits. It should be safety_factor <=1
    power_W = power_kW * 1000f0   # Receives Power in kW, goes to W (to obtain Amperes)
    η_c = power_W > 0 ? bat.η_c : 1f0
    i = power_W / (bat.v * bat.Np * bat.Ns * η_c)
    
    i_max = min(bat.i_max, (bat.soc - bat.soc_min/safety_factor) * bat.q * 3600 / (Δt))
    i_min = max(-bat.i_max, (bat.soc - bat.soc_max*safety_factor) * bat.q * 3600 / (Δt))

    i_safe_soc = clamp(i, i_min, i_max)
    p_lim = min(max_power(bat), bat.p_max)  # Safe check in case the predefined power limit of the battery is higher than the theoretical limit. 
    return clamp(i_safe_soc * bat.v * bat.Np * bat.Ns * η_c / 1000, -p_lim, p_lim)
end

@inline function max_power(bat::Battery)::Float32
    bat.v_rated * bat.i_max * bat.Np * bat.Ns / 1000  # Output in kW
end

"""
    limit_hp_power(power, env::MCES_Env)

Limits the heat pump (HP) power to ensure the state of charge (SoC) of the thermal energy storage system (TESS) remains within safe limits.

# Arguments
- `power::Float32`: The desired power for the heat pump.
- `env::MCES_Env`: The environment structure containing relevant properties of the heat pump, TESS, load, etc.

# Returns
- The limited power that can be demanded from the heat pump.

# Details
1. `safety_factor`: A factor (0.9) used to prevent SoC from slightly exceeding limits due to decimal precision issues or considerable changes in exogenous info.
2. `tess`: The TESS component of the environment.
3. `hp`: The heat pump component of the environment.
4. `p_tess_max`: Maximum allowable power for TESS calculated based on the SoC, minimum SoC, capacity (`q`), and time step (`Δt`).
5. `p_tess_min`: Minimum allowable power for TESS calculated similarly to `p_tess_max`.
6. `hp_th`: Theoretical HP power, clamped to ensure it's within the operational range of the heat pump.
7. `p_tess`: Actual power entering or leaving the TESS, calculated as the difference between thermal load and sum of solar thermal input and HP power.
8. `η_th`: Thermal efficiency factor for TESS, adjusted based on whether `p_tess` is positive (discharging) or negative (charging).
9. `p_tess_soc`: The power that will update the SoC, adjusted by the efficiency factor.
10. `p_tess_safe_soc`: Clamped value of `p_tess_soc` to ensure it stays within the calculated safe limits (`p_tess_min` and `p_tess_max`).
11. `hp_e_safe_soc`: The power demand for the heat pump, ensuring it remains within the allowable range for updating the TESS SoC.
12. The function returns the clamped `hp_e_safe_soc`, which will satisfy the HP limits and likely the TESS SoC limits. This last satisfaction will be guaranteed by the exogenous! update. 

This function only works on the current timestep, so a big change in the thermal load or the solar thermal output will make the defined limits unsafe. This situation is compensated by the safety factor.
"""
function limit_hp_power(power, env::MCES_Env, simple::Bool = false)
    hp = env.hp

    if simple
        return clamp(power, 0f0, hp.e_max)
    end
    
    safety_factor = 1f0  # To reduce the width between the limits. It should be safety_factor <= 1
    tess = env.tess
    
    p_tess_max = min(tess.p_max, (tess.soc - tess.soc_min / safety_factor) * tess.q * 3600 / env.Δt)  # Both in kW
    p_tess_min = max(-tess.p_max, (tess.soc - tess.soc_max * safety_factor) * tess.q * 3600 / env.Δt)

    hp_th = clamp(power, 0f0, hp.e_max) * hp.η  # Clamped here to assure the sign of p_tess is correct
    p_tess = env.load_th - env.st - hp_th  
    η_th = p_tess > 0 ? 1 / tess.η : tess.η
    p_tess_soc = p_tess * η_th 

    p_tess_safe_soc = clamp(p_tess_soc, p_tess_min, p_tess_max)
    hp_e_safe_soc = (env.load_th - env.st - p_tess_safe_soc / η_th) / hp.η
    
    return clamp(hp_e_safe_soc, 0f0, hp.e_max)
end


####################################################################################################
# SoC Updates
"""
    update_bat_soc!(bat::Battery, env::MCES_Env; power_kW::Float32 = bat.p, η::Float32 = bat.η_c)

Updates the state of charge (SoC) of the battery based on the power demanded and the Coulombic efficiency.

# Arguments
- `bat::Battery`: The battery whose SoC needs to be updated.
- `env::MCES_Env`: The MCES environment, it contains the timestep in seconds.
- `power_kW::Float32`: The power in kW (optional, defaults to `bat.p`).
- `η::Float32`: The efficiency (optional, defaults to `bat.η_c`).

# Returns
- The amount by which the SoC was clamped.

# Details
1. Converts the provided power from kW to W.
2. Determines the charging efficiency (`η_c`) based on the direction of power flow.
3. Calculates the current (`i`) using the battery parameters.
4. Updates the SoC and clamps it within the safe limits, returning the amount of clamping.
"""
function update_bat_soc!(bat::Battery, env::MCES_Env; power_kW::Float32 = bat.p, η::Float32 = bat.η_c)
    power_W = power_kW * 1000f0  # Convert from kW to W
    η_c = power_W > 0 ? η : 1/η
    bat.i = power_W / (bat.v * bat.Np * bat.Ns * η_c)
    bat.soc, extra_soc = clamp_with_amount(bat.soc - bat.i * env.Δt / (bat.q * 3600), bat.soc_min, bat.soc_max, abs = true)
    return extra_soc
end

function update_ev_soc!(ev::EV, env::MCES_Env)
    # Update EV SoC (only if car is present in the current timestep)
    if ev.γ == 1
        extra_ev_soc = update_bat_soc!(ev.bat, env)
    else
        extra_ev_soc = update_bat_soc!(ev.bat, env, power_kW = ev.p_drive, η = 1f0)
    end
    return extra_ev_soc
end


function update_tess_soc!(tess::TESS, env::MCES_Env)::Float32
    η_th = tess.p > 0 ? 1 / tess.η : tess.η
    p_tess_soc = tess.p * η_th
    tess.soc, extra_tess_soc = clamp_with_amount(tess.soc - p_tess_soc * env.Δt / (tess.q * 3600), tess.soc_min, tess.soc_max, abs = true)
    return extra_tess_soc
end

####################################################################################################
# Voltage Updates

@inline function update_voltage!(bat::Battery)
    bat.v = bat.a + bat.b * bat.soc
end



    


#########################################################################################
# Assigning Distributions from the Agent's decisions. 
@inline function action_distribution(dist::Type{T}, model_output) where {T<:ContinuousDistribution}
    map(col -> dist(col...), eachcol(model_output))
end

"""
    Gaussian_actions(model_output::Matrix{Float32})

Convert a matrix of model outputs into a vector of Gaussian distributions.

This function is similar to `Gaussian_withgrad`, but it returns a vector
instead of a matrix. Also, it doesn't use the Zygote.Buffer, so no gradients can be backpropagated.

# Arguments
- `model_output::Matrix{Float32}`: A matrix with 2 rows. The first row
  contains means, and the second row contains standard deviations for
  each action.

# Returns
- `Vector{Normal{Float32}}`: A vector where each element is a
  `Normal{Float32}` distribution corresponding to each column of the input.

# Note
Benchmarked against using map!, but a for loop is faster and has less allocations. Julia compiler magic.
"""
@inline function Gaussian_actions(model_output::Matrix{Float32}) 
    number_of_actions = size(model_output, 2)
    buf = Vector{Normal{Float32}}(undef, number_of_actions)
    for i in Base.OneTo(number_of_actions)
        column = model_output[:, i] # First row: mean, Second row: std
        buf[i] = Distributions.Normal{Float32}(column...)
    end
    buf
end

# function Gaussian_actions(model_output::Vector{Float32})
    
# end


"""
    action_distribution_withgrad(dist::Type{T}, model_output) where {T<:ContinuousDistribution}

Create an array of probability distributions from the Neural Network model output, which represents the means and standard deviations of the actions to take.

# Arguments
- `dist::Type{T}`: The type of continuous distribution to be used (e.g., `Normal`, `Uniform`).
- `model_output`: A matrix or vector representing the parameters for distributions. Means on first row and Stds on second. 

# Returns
An `Array` of probability distributions based on the input of means and standard deviations.

# Details
This function performs the following steps:
1. Reshapes `model_output` to ensure it has 2 rows (if necessary).
2. Determines the number of actions based on the number of columns in the reshaped input.
3. Uses a `Buffer` for memory allocation within the Zygote gradient calculation.
4. Creates a distribution for each column of the reshaped input.
5. Copies the buffer to a standard Array before returning.

The function is designed to handle model outputs that may not be in the exact shape required for distribution creation.
"""
function action_distribution_withgrad(dist::Type{T}, model_output) where {T<:ContinuousDistribution}
    dist_param = size(model_output)[1] != 2 ? reshape(model_output, 2, :) : model_output
    action_number = size(dist_param, 2)
    buf = Buffer([], 1, action_number)
    for i in range(1, action_number)
        col = dist_param[:, i]
        buf[i] = dist(col...)
    end
    copy(buf) #array with the distribution for every column of model output. 
end


"""
    Gaussian_withgrad(model_output::Matrix{Float32})

Convert a matrix of model outputs into an array of Gaussian distributions.

This function takes a matrix where each column represents the parameters
(mean and standard deviation) of a Gaussian distribution. It returns a
matrix of `Normal{Float32}` distributions.

# Arguments
- `model_output::Matrix{Float32}`: A matrix with 2 rows. The first row
  contains means, and the second row contains standard deviations for
  each action.

# Returns
- `Matrix{Normal{Float32}}`: A matrix where each element is a
  `Normal{Float32}` distribution corresponding to each column of the input.

# Note
This function uses a `Zygote.Buffer` for intermediate storage, allowing to backpropagate the gradients.
"""
function Gaussian_withgrad(model_output) 
    number_of_actions = size(model_output, 2) #expects model output to have 2 rows (mean and std)
    buf = Buffer([Normal(0f0, 0f0)], 1, number_of_actions)
    for i in Base.OneTo(number_of_actions)
        column = model_output[:, i] # First row: mean, Second row: std
        buf[i] = Distributions.Normal{Float32}(column...)
    end
    copy(buf) #array with the distribution for every column of model output. 
end


#########################################################################################
# Gradient Clipping
global_norm(gs, ps) = sqrt(sum(mapreduce(x -> x^2, +, gs[p]) for p in ps))

function clip_by_global_norm!(gs, ps, clip_norm::Float32; noise::Bool = false)
    gn = global_norm(gs, ps)
    if clip_norm <= gn
        for p in ps
            gs[p] .*= clip_norm / max(clip_norm, gn)
        end
    end
    for p in ps
        gs[p] .+= 1f-4 * noise
    end
    gn
end

#########################################################################################
# Hyperparameter tuning

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



####################################################################################################
# Safety checks

function perform_safety_warnings(env::MCES_Env)
    ev = env.ev           # Electric Vehicle (EV)
    bess = env.bess       # Battery Energy Storage System (BESS)
    tess = env.tess       # Thermal Energy Storage System (TESS)

    if ev.bat.soc > 1 || (ev.bat.soc) < ev.bat.soc_min
        @warn "EV Battery SoC is out of limits" maxlog = 10 ev.bat.soc  
    end
    
    if bess.soc > 1 || (bess.soc) < bess.soc_min
        @warn "BESS SoC is out of limits" maxlog = 10 bess.soc
    end
    
    if tess.soc > 1 || (tess.soc) < tess.soc_min/2    # Since the TESS is subject to exogenous information, its more likely that it will go below the limit more often. 
        @warn "Thermal ESS SoC is out of limits" maxlog = 10 tess.soc
    end
    return nothing
end

####################################################################################################
# Data from EMS Module to MCES Env
function EMSDict_to_Replay(rhDict::Dict)
    required_fields = ["Pbess", "Pev[1]", "Phpe", "γf"]
    
    # Error Checking
    for field in required_fields
        if !haskey(rhDict, field)
            error("Missing required field: $field")
        end
    end

    Pbess = get(rhDict, "Pbess", Float32[])
    Pev = get(rhDict, "Pev[1]", Float32[])
    Phpe = get(rhDict, "Phpe", Float32[])
    γf = get(rhDict, "γf", Float32[])
    
    if !isempty(γf) && !isempty(Pev) && length(γf) == length(Pev)
        processed_Pev = γf .* Pev
    else
        error("Pev and γf are not of same length")
    end
    
    return Replay{Vector{Float32}}(
        p_ev = processed_Pev,
        p_bess = Pbess,
        p_hp_e = Phpe,
        ind = 1,
        max_ind = length(Pbess)  
    )
end

function save_MCES_Hook(filepath::String, hook::AbstractHook)
    jldsave(filepath, hook = hook)
end

function load_MCES_Hook(filepath::String)
    JLD2.load(filepath, "hook")
end

function save_policy(filepath::String, policy::Union{myVPG,A2CGAE,PPO,Replay})
    jldsave(filepath, p = policy)
end

function load_policy(filepath::String)
    JLD2.load(filepath, "p")
end

function load_policy_folder(folder_path::String)
    policies = []
    
    try
        if !isdir(folder_path)
            error("The specified folder does not exist: $folder_path")
        end
        
        # Find all files that match the pattern
        matching_files = filter(file -> startswith(basename(file), "model"), readdir(folder_path, join=true, sort = true))

        if isempty(matching_files)
            @warn "No matching policy files found in the folder: $folder_path"
            return policies
        end

        println("Found $(length(matching_files)) models in folder: $(basename(folder_path))")
        
        # Load each matching file
        for file in matching_files
            try
                policy = load_policy(file)
                push!(policies, policy)
                @info "Successfully loaded policy from file: $(basename(file))"
            catch e
                @warn "Failed to load policy from file: $file" exception=(e, catch_backtrace())
            end
        end
        
    catch e
        @error "An error occurred while processing the folder: $folder_path" exception=(e, catch_backtrace())
    end
    
    return policies
end

function load_hooks_folder(folder_path::String)
    hooks = MCES_Hook[]
    
    try
        if !isdir(folder_path)
            error("The specified folder does not exist: $folder_path")
        end
        
        # Find all files that match the pattern (starting with "model")
        matching_files = filter(file -> startswith(basename(file), "hook"), readdir(folder_path, join=true, sort=true))

        if isempty(matching_files)
            @warn "No matching hook files found in the folder: $folder_path"
            return hooks
        end

        println("Found $(length(matching_files)) Hooks in folder: $(basename(folder_path))")
        
        # Load each matching file
        for file in matching_files
            try
                hook = load_MCES_Hook(file)
                push!(hooks, hook)
                @info "Successfully loaded hook from file: $(basename(file))"
            catch e
                @warn "Failed to load hook from file: $file" exception=(e, catch_backtrace())
            end
        end
        
    catch e
        @error "An error occurred while processing the folder: $folder_path" exception=(e, catch_backtrace())
    end
    
    return hooks
end


function find_differences(vec1::Vector, vec2::Vector)
    length(vec1) != length(vec2) && error("Vectors have different lengths")
    
    for i in 1:length(vec1)
        if vec1[i] != vec2[i]
            println("Difference at index $i")
        end
    end
    println("Reached the end of the vectors")
end
###################################################################################
# To Clone Behaviour
function get_actions(rhDict::Dict)
    required_fields = ["Pbess", "Pev[1]", "Phpe", "γ_cont"]
    
    # Error Checking
    for field in required_fields
        if !haskey(rhDict, field)
            error("Missing required field: $field")
        end
    end

    Pbess = get(rhDict, "Pbess", Float32[])
    Pev = get(rhDict, "Pev[1]", Float32[])
    Phpe = get(rhDict, "Phpe", Float32[])
    γ_cont = get(rhDict, "γ_cont", Float32[])
    
    if !isempty(γ_cont) && !isempty(Pev) && length(γ_cont) == length(Pev)
        processed_Pev = γ_cont .* Pev
    else
        processed_Pev = Float32[]
    end
    
    # Create a matrix with 3 rows
    action_matrix = zeros(Float32, 3, length(Pbess))
    action_matrix[1, :] = processed_Pev
    action_matrix[2, :] = Pbess
    action_matrix[3, :] = Phpe
    
    return action_matrix
end


function get_socs(rhDict::Dict)
    # List of required fields
    required_fields = ["SoCtess", "SoCbess", "SoCev[1]", "γ_cont"]
    
    # Error Checking
    for field in required_fields
        if !haskey(rhDict, field)
            error("Missing required field: $field")
        end
    end

    # Extract the relevant fields
    SoCtess = get(rhDict, "SoCtess", Float32[])
    SoCbess = get(rhDict, "SoCbess", Float32[])
    SoCev = get(rhDict, "SoCev[1]", Float32[])
    γ_cont = get(rhDict, "γ_cont", Float32[])

    # Create a matrix with 4 rows, each representing one of the state variables
    state_matrix = zeros(Float32, 4, length(SoCtess))
    state_matrix[1, :] = SoCtess
    state_matrix[2, :] = SoCbess
    state_matrix[3, :] = SoCev
    state_matrix[4, :] = γ_cont
    
    return state_matrix
end

function get_exog_normalized(exog::Exogenous_BatchCollection)
    load_e_normalized = z_score(exog.load_e, 0.342, 0.203)
    load_th_normalized = z_score(exog.load_th, 0.655, 0.405)
    pv_normalized = z_score(exog.pv, 0.528, 0.873)
    λ_buy_normalized = z_score(exog.λ_buy, 0.242, 0.128)
    λ_sell_normalized = z_score(exog.λ_sell, 0.231, 0.121)

    normalized_matrix = zeros(Float32, 5, length(exog.load_e))
    normalized_matrix[1, :] = load_e_normalized
    normalized_matrix[2, :] = load_th_normalized
    normalized_matrix[3, :] = pv_normalized
    normalized_matrix[4, :] = λ_buy_normalized
    normalized_matrix[5, :] = λ_sell_normalized

    return normalized_matrix
end

function get_t_ep(ep_length::Int, episodes::Int)
    ep = [i/ep_length for i in 1:ep_length]
    Float32.(repeat(ep, episodes))
end

function get_t_year(length::Int)
    ep = [i/35040 for i in 1:length]
    Float32.(ep)
end

function get_full_state(exog::Exogenous_BatchCollection, rhDict::Dict, ep_length::Int)
    # Extract SoCs and γ_cont
    socs_matrix = get_socs(rhDict)
    max_step = length(socs_matrix[1,:])   # The rhDict is the limiting factor

    # Extract and normalize exogenous values
    normalized_exog = get_exog_normalized(exog)[:, 1:max_step]
    
    # Calculate t_ep_ratio and t_year_ratio
    episodes = div(max_step, ep_length)
    t_ep_ratio = get_t_ep(ep_length, episodes)
    # t_year_ratio = get_t_year(max_step)
    
    # Combine all into an 11-row matrix
    full_state_matrix = vcat(
        normalized_exog, 
        socs_matrix, 
        reshape(t_ep_ratio, 1, :)
    )
    
    return full_state_matrix
end

function save_clone_policy(filepath::String, policy)
    jldsave(filepath, clone = policy)
end

function load_clone_policy(filepath::String)
    JLD2.load(filepath, "clone")
end

function save_expert_data(filepath::String, actions, states, hook)
    jldsave(filepath, actions = actions, states = states, rewards = hook) # Rewards can be found and weighted within the hook. 
end

function load_expert_data(filepath::String)
    @load filepath actions states rewards 
    println("Output is ordered -> actions, states, rewards")
    actions, states, rewards
end

function double_backslashes(path::String)
    # The string must have "raw" before, to be accepted by the function.
    # Example -> raw"C:\Users\...\file.jld2"
    return replace(path, "\\" => "\\")
end


function run_with_redirect(house, policy; exog, filename::String = "Ipopt_debug.txt")
    debug_file = joinpath(@__DIR__, "debug", filename)
    original_stdout = stdout

    local h
    open(debug_file, "w") do file
        redirect_stdout(file)
        h = run_dont_learn(house,policy; exog = exog)
        flush(file)
    end

    redirect_stdout(original_stdout)
    return h
end

#########################################################################################



@info "Loaded helper functions"

