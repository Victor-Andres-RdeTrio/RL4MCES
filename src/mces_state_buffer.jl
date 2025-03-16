"""
   state_buffer_update!(ag::Agent, env::MCES_Env; training::Bool = true)

Updates the state buffer with the current state of the environment and applies normalization.

# Arguments
- `ag::Agent`: The agent object that contains the policy with online statistics for normalization.
- `env::MCES_Env`: The MCES environment containing the current state variables.
- `training::Bool = true`: Flag indicating whether to update the running statistics during training.

# Details
1. Constructs a `curr_state` vector with the following environment variables:
  - Electrical load (`env.load_e`)
  - Thermal load (`env.load_th`)
  - Photovoltaic generation (`env.pv`)
  - Buy price (`env.λ_buy`)
  - Sell price (`env.λ_sell`)
  - EV availability factor (`env.ev.γ`)
  - Grid power (`env.grid`)
  - Battery power (`env.bess.p`)
  - EV battery power (`env.ev.bat.p`)
  - Heat pump electrical power (`env.hp.e`)
  - Battery state of charge (`env.bess.soc`)
  - EV battery state of charge (`env.ev.bat.soc`)
  - Thermal energy storage state of charge (`env.tess.soc`)
  - Normalized time step within the day (`mod1(env.t_ep, 96)/96`)

2. If `training` is true, updates the running mean and standard deviation in the agent's policy for each of the state variables.
3. Applies z-score normalization to the `curr_state` vector using the agent's online statistics.
4. Pushes the normalized state to the environment's state buffer.

# Note
- The state vector is of type `Float32` for computational efficiency.
- The state includes 14 variables representing the full state of the MCES environment.
- The time step is normalized to a value between 0 and 1 representing the progress through the day.
"""
function state_buffer_update!(ag::Agent, env::MCES_Env; training::Bool = true)
    curr_state = Float32[
        env.load_e,
        env.load_th,
        env.pv,
        env.λ_buy, 
        env.λ_sell, 
        env.ev.γ,
        env.grid,
        env.bess.p,
        env.ev.bat.p,
        env.hp.e,
        env.bess.soc,
        env.ev.bat.soc, 
        env.tess.soc,
        mod1(env.t_ep, 96)/96
    ]    

    training && update!(ag.policy.online_stats, curr_state) # Update the running mean and std
    z_score!(curr_state, ag.policy.online_stats) # Z-score normalisation on the curr_state
    push!(env.state_buffer, curr_state)
end


###################################################################################
# State construction for Agent
"""
    matrix_select_states(daylength::Int = 96; kwargs...)

Generate a Bit Matrix that efficiently represents the selection of states from a buffer based on specified lags (in days).

This function creates a matrix where each row corresponds to a state variable, and the columns represent
time steps within a day. The matrix is used to extract relevant values from a state buffer matrix.

# Arguments
- `daylength::Int = 96`: The number of time steps in a day. Defaults to 96 (15-minute intervals).
- `kwargs...`: Keyword arguments representing state variables. Each argument should be a 
  `Tuple{<:Integer, Vector{<:Real}}`, where the first element is the row index and the second 
  is a vector of lags in days and day fractions.

# Returns
- `BitMatrix`: A BitMatrix of size `(max_row, daylength)` where max_row is the maximum row index provided.
  Elements are set to true where state information should be selected, and false elsewhere.


# Example
```julia
state_buffer_dict = Dict(
    :state1 => (1, [0.0, 0.5, 1.0]),
    :state2 => (2, [0.25, 0.75])
)
matrix = matrix_select_states(; state_buffer_dict...)
```
This creates a matrix where row 1 is true at indices corresponding to lags of 0, 0.5, and 1 day,
and row 2 is true at indices corresponding to lags of 0.25 and 0.75 days.

# Notes
- The function uses `day_to_index` to convert day lags to column indices, output is always within [1, daylength].
- Arguments with row indices outside the range 1:max_row or with empty day vectors are skipped.
- Throws an ArgumentError if any keyword argument value isn't a Tuple{<:Integer, Vector{<:Real}}.
- This function is crucial for defining the field `state_buffer_ind` within the environment cnstructor function `build_MCES`.
- This function is designed to work with a state buffer indexing system where:
    - State information is stored in a buffer of size (n_state_variables, daylength).

"""
function matrix_select_states(daylength::Int = 96; kwargs...)
    # Function to turn lag in days to an index within [1,daylength]. 
    day_to_index(day) = daylength - round(Int32, (daylength - 1) * clamp(abs(day), 0, 1))

    max_row = maximum(v[1] for (_,v) in kwargs) 
    matrix = falses(max_row, daylength)
    for (key, value) in kwargs
        row, days_vector = value
        if !(row ∈ 1:max_row) || isempty(days_vector)
            continue
        end

        if !(value isa Tuple{<:Integer, Vector{<:Real}})
            throw(ArgumentError("'$key' must be paired with Tuple{Integer, Vector{<:Real}}"))
        end

        matrix[row, day_to_index.(days_vector)] .= true
    end
    
    return matrix
end

function matrix_select_states(dict::Dict)
    matrix_select_states(;dict...)
end


function count_features(dict::Dict)
    sum(length(t[2]) for (_, t) in dict)
end

###############################################
# These are the feature vector configurations used within the Master thesis project.
"""
    what_features(type::Integer)

Returns a dictionary mapping feature variables to an index and an array of time lags (max 1 day of lag),
selecting the configuration based on the provided type. Raises an error if the type is unrecognized.
"""
function what_features(type::Integer)
    if type == 1 # Only exog, SoCs and time. 
        d = Dict(
            :load_e => (1, [0.0]),
            :load_th => (2, [0.0]),
            :pv => (3, [0.0]),
            :λ_buy => (4, [0.0]),
            :λ_sell => (5, [0.0]), 
            :γ_ev => (6, [0.0]),
            :grid => (7, []),
            :p_bess => (8, []),
            :p_ev => (9, []),
            :p_hp_e => (10, []),
            :soc_bess => (11, [0.0]),
            :soc_ev => (12, [0.0]),
            :soc_tess => (13, [0.0]),
            :t_ep_ratio => (14, [0.0])
        )

    elseif type == 2 # Last timestep
        d = Dict(
            :load_e => (1, [0.0]),
            :load_th => (2, [0.0]),
            :pv => (3, [0.0]),
            :λ_buy => (4, [0.0]),
            :λ_sell => (5, [0.0]), 
            :γ_ev => (6, [0.0]),
            :grid => (7, [0.0]),
            :p_bess => (8, [0.0]),
            :p_ev => (9, [0.0]),
            :p_hp_e => (10, [0.0]),
            :soc_bess => (11, [0.0]),
            :soc_ev => (12, [0.0]),
            :soc_tess => (13, [0.0]),
            :t_ep_ratio => (14, [0.0])
        )
        

    elseif type == 3 # Most relevant correlations
        d = Dict(
            :load_e => (1, [0.0, 0.5, 1.0]),
            :load_th => (2, [0.0, 0.5, 1.0]),
            :pv => (3, [0.0, 0.5, 0.65, 1.0]),
            :λ_buy => (4, [0.0, 0.1, 0.45, 0.65, 1.0]),
            :λ_sell => (5, [0.0, 0.1, 0.45, 0.65, 1.0]), 
            :γ_ev => (6, [0.0, 1.0]),
            :grid => (7, [0.0, 0.5, 1.0]),
            :p_bess => (8, [0.0, 1.0]),
            :p_ev => (9, [0.0, 0.5, 1.0]),
            :p_hp_e => (10, [0.0, 0.5, 1.0]),
            :soc_bess => (11, [0.0, 0.15, 0.4, 0.8, 1.0]),
            :soc_ev => (12, [0.0, 0.25, 0.75, 1.0]),
            :soc_tess => (13, [0.0, 0.25, 0.5, 0.75]),
            :t_ep_ratio => (14, [0.0, 0.3, 0.9])
        )

    elseif type == 4 # Most relevant correlations (without decisions and λ_sell)
        d = Dict(
            :load_e => (1, [0.0, 0.5, 1.0]),
            :load_th => (2, [0.0, 0.5, 1.0]),
            :pv => (3, [0.0, 0.5, 0.65, 1.0]),
            :λ_buy => (4, [0.0, 0.1, 0.45, 0.65, 1.0]),
            :λ_sell => (5, []), 
            :γ_ev => (6, [0.0, 1.0]),
            :grid => (7, [0.0, 0.5, 1.0]),
            :p_bess => (8, []),
            :p_ev => (9, []),
            :p_hp_e => (10, []),
            :soc_bess => (11, [0.0, 0.15, 0.4, 0.8, 1.0]),
            :soc_ev => (12, [0.0, 0.25, 0.75, 1.0]),
            :soc_tess => (13, [0.0, 0.25, 0.5, 0.75]),
            :t_ep_ratio => (14, [0.0, 0.3, 0.9])
        )

    elseif type == 5 # Relevant correlations (simplified)
        d = Dict(
            :load_e => (1, [0.0, 0.5, 1.0]),
            :load_th => (2, [0.0, 0.5, 1.0]),
            :pv => (3, [0.0, 0.25, 0.65]),
            :λ_buy => (4, [0.1, 0.45, 1.0]),
            :λ_sell => (5, []), 
            :γ_ev => (6, [0.0, 1.0]),
            :grid => (7, [0.0, 0.5, 1.0]),
            :p_bess => (8, [0.0, 1.0]),
            :p_ev => (9, [0.0, 1.0]),
            :p_hp_e => (10, [0.0, 1.0]),
            :soc_bess => (11, [0.1, 0.8]),
            :soc_ev => (12, [0.0, 0.25, 0.75]),
            :soc_tess => (13, [0.0, 0.25, 0.8]),
            :t_ep_ratio => (14, [0.0])
        )
    elseif type == 6 # Randomized
        d = Dict(
            :load_e => (1, [0.17, 0.58, 0.92]),
            :load_th => (2, [0.03, 0.41, 0.79]),
            :pv => (3, [0.22, 0.55, 0.88]),
            :λ_buy => (4, [0.09, 0.36, 0.71]),
            :λ_sell => (5, [0.14, 0.47, 0.83]), 
            :γ_ev => (6, [0.06, 0.39, 0.75]),
            :grid => (7, [0.28, 0.61, 0.95]),
            :p_bess => (8, [0.11, 0.44, 0.80]),
            :p_ev => (9, [0.19, 0.52, 0.87]),
            :p_hp_e => (10, [0.08, 0.33, 0.69]),
            :soc_bess => (11, [0.25, 0.58, 0.91]),
            :soc_ev => (12, [0.05, 0.38, 0.72]),
            :soc_tess => (13, [0.31, 0.64, 0.97]),
            :t_ep_ratio => (14, [0.13, 0.45, 0.78])
        )

    elseif type == 7 # Educated guess (hourly, 6 hours and daily patterns)
        d = Dict(
            :load_e => (1, [0.0, 0.04, 0.25, 1.0]), 
            :load_th => (2, [0.0, 0.04, 0.25, 1.0]),
            :pv => (3, [0.0, 0.04, 0.25, 1.0]),  
            :λ_buy => (4, [0.0, 0.04, 0.25, 1.0]),
            :λ_sell => (5, []), 
            :γ_ev => (6, [0.0, 0.25, 1.0]),  
            :grid => (7, [0.0, 0.04, 0.25, 1.0]), 
            :p_bess => (8, [0.0, 0.04, 0.25, 1.0]),  
            :p_ev => (9, [0.0, 0.04, 0.25, 1.0]),  
            :p_hp_e => (10, [0.0, 0.04, 0.25, 1.0]),  
            :soc_bess => (11, [0.0, 0.04, 0.25, 1.0]), 
            :soc_ev => (12, [0.0, 0.25, 1.0]),  
            :soc_tess => (13, [0.0, 0.04, 0.25, 1.0]),  
            :t_ep_ratio => (14, [0.0, 0.25]) 
        )

    elseif type == 8 # As much info as could be useful
        d = Dict(
            :load_e => (1, [0.0, 0.1, 0.5, 0.65, 1.0]),
            :load_th => (2, [0.0, 0.4, 0.5, 0.75, 1.0]),
            :pv => (3, [0.0, 0.1, 0.25, 0.5, 0.65, 1.0]),
            :λ_buy => (4, [0.0, 0.04, 0.1, 0.2, 0.45, 0.65, 1.0]),
            :λ_sell => (5, [0.0]), 
            :γ_ev => (6, [0.0, 0.25, 1.0]),
            :grid => (7, [0.0, 0.04, 0.25, 0.5, 1.0]),
            :p_bess => (8, [0.0, 0.04, 0.65, 1.0]),
            :p_ev => (9, [0.0, 0.5, 0.7, 1.0]),
            :p_hp_e => (10, [0.0, 0.5, 0.95]),
            :soc_bess => (11, [0.0, 0.15, 0.25, 0.4, 0.8, 1.0]),
            :soc_ev => (12, [0.0, 0.04, 0.25, 0.75, 1.0]),
            :soc_tess => (13, [0.0, 0.25, 0.4, 0.65, 0.8]),
            :t_ep_ratio => (14, [0.0, 0.25, 0.4, 0.9])
        )

    elseif type == 9 # Highest correlation for the most related decisions
        d = Dict(
            :load_e => (1, [0.0, 0.1, 0.5, 0.9]),
            :load_th => (2, [0.0, 0.4, 0.75]),
            :pv => (3, [0.0, 0.25, 0.5, 0.75]),
            :λ_buy => (4, [0.0, 0.1, 0.4, 0.65]),
            :λ_sell => (5, [0.0]), 
            :γ_ev => (6, [0.0, 1.0]),
            :grid => (7, [0.0, 0.5, 1.0]),
            :p_bess => (8, [0.0, 1.0]),
            :p_ev => (9, [0.0, 0.5, 1.0]),
            :p_hp_e => (10, [0.0, 0.5, 1.0]),
            :soc_bess => (11, [0.0, 0.1, 0.25, 0.5, 0.8]),
            :soc_ev => (12, [0.0, 0.25, 0.8]),
            :soc_tess => (13, [0.0, 0.2, 0.8]),
            :t_ep_ratio => (14, [0.0, 0.25, 0.4])
        )
    else
        error("Feature Selection Dictionary you were looking for was not found.")
    end
    return d 
end

@info "State Buffer is operational."