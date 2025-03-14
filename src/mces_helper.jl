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

"""
    print_last_values(dict::Dict, shift::Int = 0)

Print the last values of arrays stored in a dictionary, with optional display of multiple trailing values.

# Arguments
- `dict::Dict`: A dictionary containing key-value pairs where values are arrays.
- `shift::Int = 0`: Number of additional trailing values to display before the last value. Default is 0 (only shows the last value).

# Details
- For numeric arrays: Rounds values to 6 decimal places.
- For string values: Displays only the last character.
- Prints "Failed" for empty arrays or non-array values.
"""
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

"""
    find_dif(a::Vector, b::Vector; first::Bool = true)

Find positions where two vectors have different values.

# Arguments
- `a::Vector`: First vector to compare.
- `b::Vector`: Second vector to compare.
- `first::Bool = true`: If true, returns only the first difference position; if false, returns all difference positions.

# Returns
- If `first=true`: Index of the first difference found, or nothing if vectors are identical.
- If `first=false`: Array of indices where differences exist, or empty array if vectors are identical.

# Notes
- Converts inputs to Float32 for comparison.
- Prints "Different Length" and returns nothing if vectors have different lengths.
"""

function find_dif(a::Vector , b::Vector ; first::Bool = true)
    length(a) == length(b) ? nothing : return println("Different Length")
    a = Float32.(a)
    b = Float32.(b)
    c = abs.(a - b)
    return first ? findfirst(c-> c > 0, c) : findall(c-> c > 0, c)
end


"""
    print_dict_to_file(dict::Dict, filename::String; title="Training Information")

Write dictionary contents to a file in a structured, hierarchical format with sorted keys.

# Arguments
- `dict::Dict`: The dictionary to write to file.
- `filename::String`: Path to the output file (will be appended to if file exists).
- `title::String = "Training Information"`: Optional title for the output section.

# Details
- Writes nested dictionaries with increased indentation and section markers.
- Sorts dictionary keys alphabetically for consistent output.
- Formats output with section markers (===) for better readability.
"""
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

"""
    double_backslashes(path::String)

Replace single backslashes with double backslashes in a file path string.

# Arguments
- `path::String`: Input file path string.

# Returns
- `String`: Path with all single backslashes replaced by double backslashes.

# Notes
- Intended for use with raw string literals, e.g., `double_backslashes(raw"C:\\Users\\...\\file.jld2")`.
"""
function double_backslashes(path::String)
    # The string must have "raw" before, to be accepted by the function.
    # Example -> raw"C:\Users\...\file.jld2"
    return replace(path, "\\" => "\\")
end

##################################################################################


"""
    limit_battery_power(power_kW, bat::Battery, Δt, simple::Bool = false)

Limits the power output of a battery device to safe levels based on its design characteristics.
Also guarantees that the SoC min and max levels are respected.
Takes in power in kW and outputs power in kW.
The Coulombic efficiency will be <1 when charging and =1 when discharging.

# Arguments
- `power_kW::Float32`: Proposed power level for the battery device in kW.
- `bat::Battery`: Battery device object containing specifications and state information.
- `Δt::Float32`: Time step size in seconds.
- `simple::Bool`: If true, performs basic power clamping without SoC considerations (optional, defaults to `false`).

# Returns
- The limited power output for the battery device in kW.

# Details
1. If `simple` is true, simply clamps power between negative and positive `p_max` (known as "simple projection").
2. Converts power from kW to W for internal calculations.
3. Determines charging efficiency (`η_c`) based on power direction (<0 is charging, >0 is discharging).
4. Calculates current (`i`) using battery parameters and efficiency.
5. Computes safe current limits (`i_max`, `i_min`) based on SoC constraints and safety factor.
6. Clamps the current and converts back to power in kW, ensuring it's within theoretical and rated limits.
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

"""
    max_power(bat::Battery)::Float32

Calculates the theoretical maximum power capability of a battery based on its electrical characteristics.

# Arguments
- `bat::Battery`: Battery device object containing specifications.

# Returns
- The theoretical maximum power output in kW.

# Details
Multiplies the rated voltage, maximum current, and number of cells in parallel and series,
then divides by 1000 to convert from W to kW.
"""
@inline function max_power(bat::Battery)::Float32
    bat.v_rated * bat.i_max * bat.Np * bat.Ns / 1000  # Output in kW
end


"""
    limit_hp_power(power, env::MCES_Env, simple::Bool = false)

Limits the heat pump (HP) power to ensure the state of charge (SoC) of the thermal energy storage system (TESS) remains within safe limits.

# Arguments
- `power::Float32`: The desired power for the heat pump in kW.
- `env::MCES_Env`: The environment structure containing relevant properties of the heat pump, TESS, load, etc.
- `simple::Bool`: If true, performs basic power clamping without SoC considerations (optional, defaults to `false`).

# Returns
- The limited power that can be demanded from the heat pump in kW.

# Details
1. If `simple` is true, simply clamps power between 0 and `hp.e_max`.
2. `safety_factor`: A factor (default 1.0) used to adjust the SoC limits for safety margin.
3. Calculates maximum and minimum allowable TESS power based on SoC constraints.
4. Computes theoretical HP thermal output (`hp_th`) based on input power and efficiency.
5. Determines power entering/leaving the TESS (`p_tess`) as the difference between thermal load and the input to MCES environment by HP and solar thermal.
6. Applies appropriate thermal efficiency based on whether TESS is charging or discharging.
7. Clamps the TESS power within safe limits and back-calculates the corresponding HP electrical input power.
8. Returns the HP input power, clamped to its operational range.

This function operates on the current timestep only. Significant changes in thermal load or solar thermal output between steps may impact the effectiveness of the SoC limits, which is partly mitigated by the safety factor.
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
- `env::MCES_Env`: The MCES environment, containing the timestep in seconds.
- `power_kW::Float32`: The power in kW (optional, defaults to `bat.p`).
- `η::Float32`: The efficiency (optional, defaults to `bat.η_c`).

# Returns
- The amount by which the SoC was clamped.

# Details
1. Converts the provided power from kW to W.
2. Determines the charging efficiency (`η_c`) based on the direction of power flow.
   When power is positive (discharging), efficiency is applied as is; when negative (charging), 
   the inverse of efficiency is used.
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

@inline function update_voltage!(bat::Battery)
    bat.v = bat.a + bat.b * bat.soc
end


#########################################################################################
# Assigning Distributions from the Agent's decisions. 
"""
    action_distribution(dist::Type{T}, model_output) where {T<:ContinuousDistribution}

Create a vector of probability distributions from model outputs. Imported directly from ReinforcementLearning.jl

# Arguments
- `dist::Type{T}`: The type of continuous distribution to use (e.g., `Normal`, `Uniform`).
- `model_output`: Matrix where each column contains parameters for a distribution.

# Returns
- Vector of probability distributions, one for each column in the input matrix.

# Details
Maps the distribution constructor to each column of model_output using the column values as distribution parameters.
"""
@inline function action_distribution(dist::Type{T}, model_output) where {T<:ContinuousDistribution}
    map(col -> dist(col...), eachcol(model_output))
end

"""
    Gaussian_actions(model_output::Matrix{Float32})

Convert a matrix of model outputs into a vector of Gaussian distributions.

# Arguments
- `model_output::Matrix{Float32}`: A matrix with 2 rows where the first row contains means and the second row contains standard deviations.

# Returns
- `Vector{Normal{Float32}}`: A vector where each element is a Gaussian distribution corresponding to each column of the input.

# Details
- Creates Normal distributions without using Zygote.Buffer, meaning no gradients can be backpropagated.
- More efficient than using map! due to compiler optimizations.
- This function is similar to `Gaussian_withgrad`, but it returns a vector instead of a matrix.
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



"""
    action_distribution_withgrad(dist::Type{T}, model_output) where {T<:ContinuousDistribution}

Create an array of probability distributions from neural network outputs with gradient support.

# Arguments
- `dist::Type{T}`: The type of continuous distribution to use (e.g., `Normal`).
- `model_output`: Matrix where each column contains distribution parameters (means in first row, standard deviations in second).

# Returns
- Array of probability distributions with gradient tracking preserved.

# Details
1. Ensures input has correct shape (2 rows) by reshaping if necessary.
2. Uses Zygote.Buffer for memory allocation within gradient calculations.
3. Creates distributions for each column of parameters.
4. Returns a gradient-compatible array of distributions.
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
    Gaussian_withgrad(model_output)

Convert model outputs into an array of Gaussian distributions with gradient support.

# Arguments
- `model_output`: Matrix with 2 rows containing means (row 1) and standard deviations (row 2).

# Returns
- Matrix of Normal{Float32} distributions with gradient tracking preserved.

# Details
Uses Zygote.Buffer for intermediate storage to allow gradient backpropagation through the operation.
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
"""
    global_norm(gs, ps)

Calculate the global norm of gradients across multiple parameters. Imported directly from ReinforcementLearning.jl

# Arguments
- `gs`: Gradient dictionary/collection.
- `ps`: Parameter collection to include in the norm calculation.

# Returns
- The square root of the sum of squared gradient values across all parameters.
"""
global_norm(gs, ps) = sqrt(sum(mapreduce(x -> x^2, +, gs[p]) for p in ps))


"""
    clip_by_global_norm!(gs, ps, clip_norm::Float32; noise::Bool = false)

Clip gradients by their global norm in-place. Imported from ReinforcementLearning.jl and added noise. 

# Arguments
- `gs`: Gradient dictionary/collection to modify.
- `ps`: Parameter collection to include in clipping.
- `clip_norm::Float32`: Maximum allowed global norm value.
- `noise::Bool = false`: Whether to add small noise to gradients.

# Returns
- The global norm value before clipping.

# Details
1. Calculates the global norm of gradients.
2. If the norm exceeds `clip_norm`, scales all gradients by the ratio clip_norm/global_norm.
3. Optionally adds small noise (1e-4) to gradients if `noise` is true.
"""
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
# Data from EMS Module to MCES Env, for direct comparison with RL agent 
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

#############################################
# Save and Load functionality
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





#########################################################################################



@info "Loaded helper functions"

