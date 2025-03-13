"""
    StopAfterEpisode_or_Early

A stopping criterion for reinforcement learning training that terminates either after a fixed 
number of episodes or when improvement falls below a threshold.

# Fields
- `episode::Int32 = Int32(1)`: Maximum number of episodes to run
- `cur::Int32 = Int32(0)`: Current episode count
- `patience::Int32 = Int32(1)`: Number of initial episodes to ignore for early stopping
- `reference::Float32 = 0f0`: Reference value for comparing improvement
- `size::Int32 = Int32(20)`: Window size for calculating the average loss
- `min_δ::Union{Float32, Nothing} = nothing`: Minimum change threshold; if loss improvement is below this, training stops
- `stop::Bool = false`: Flag indicating whether training should stop

# Details
This structure provides a mechanism for early stopping when training progress plateaus,
helping to prevent overfitting and save computational resources. When `min_δ` is set to `nothing`,
only the episode count limit is used for stopping.

# Warning
This function was not included in the master thesis and was only tested during initial stages of development.
"""
@kwdef mutable struct StopAfterEpisode_or_Early
    episode::Int32 = Int32(1)
    cur::Int32 = Int32(0)
    patience::Int32 = Int32(1)
    reference::Float32 = 0f0
    size::Int32 = Int32(20)
    min_δ::Union{Float32, Nothing} = nothing
    stop::Bool = false
end

"""
    StopAfterEpisode_or_Early(; episode::Integer = 1, cur::Integer = 0, patience::Integer = 1, 
                              reference::Real = 0f0, size::Integer = 20, 
                              min_δ::Union{Real, Nothing} = nothing, stop::Bool = false)

Constructor for the `StopAfterEpisode_or_Early` stopping criterion with type conversion.

# Arguments
- `episode::Integer = 1`: Maximum number of episodes to run
- `cur::Integer = 0`: Current episode count
- `patience::Integer = 1`: Number of initial episodes to ignore for early stopping
- `reference::Real = 0f0`: Reference value for comparing improvement
- `size::Integer = 20`: Window size for calculating the average loss
- `min_δ::Union{Real, Nothing} = nothing`: Minimum change threshold
- `stop::Bool = false`: Flag indicating whether training should stop

# Returns
- A new `StopAfterEpisode_or_Early` instance with the provided parameters converted to appropriate types

# Details
This constructor ensures that all numeric parameters are converted to the correct types
(Int32 for integer parameters and Float32 for floating-point parameters) regardless of
the input types.

# Warning
This function was not included in the master thesis and was only tested during initial stages of development.
"""
function StopAfterEpisode_or_Early(; episode::Integer = 1,
    cur::Integer = 0,
    patience::Integer = 1,
    reference::Real = 0f0,
    size::Integer = 20,
    min_δ::Union{Real, Nothing} = nothing,
    stop::Bool = false)

    StopAfterEpisode_or_Early(
        Int32(episode), 
        Int32(cur), 
        Int32(patience), 
        Float32(reference), 
        Int32(size),
        min_δ === nothing ? nothing : Float32(min_δ), 
        stop
    )
end

"""
    (s::StopAfterEpisode_or_Early)(agent, env)

Functor implementation for the stopping criterion. Determines if training should stop.

# Arguments
- `s::StopAfterEpisode_or_Early`: The stopping criterion
- `agent`: The learning agent
- `env`: The environment

# Returns
- `Bool`: `true` if training should stop, `false` otherwise

# Warning
This function was not included in the master thesis and was only tested during initial stages of development.
"""
function (s::StopAfterEpisode_or_Early)(agent, env)

    if is_terminated(env)
        s.cur += 1
        s.stop = stop_early(s, agent.policy)
        s.stop && println("Early Stop!!! On Episode $(s.cur), with last reference value: $(s.reference)")
    end
    
    s.cur >= s.episode || s.stop
end


"""
    stop_early(s::StopAfterEpisode_or_Early, p::Union{A2CGAE, PPO, myVPG})::Bool

Determines if training should stop early based on the policy's recent loss values.

# Arguments
- `s::StopAfterEpisode_or_Early`: The stopping criterion
- `p::Union{A2CGAE, PPO, myVPG}`: The policy being trained (must be one of the supported policy types)

# Returns
- `Bool`: `true` if training should stop early, `false` otherwise

# Details
This function implements the early stopping logic by:

1. Returning `false` during the initial patience period or if `min_δ` is not set
2. Calculating the mean of the absolute raw loss values over the most recent window (defined by `s.size`)
3. Storing this mean in the policy's `δ` vector for tracking
4. Updating the reference value for future comparisons
5. Returning `true` if the mean loss delta falls below the minimum threshold (`s.min_δ`)

Early stopping helps prevent overfitting and saves computational resources by terminating
training when improvements become marginal.

# Warning
This function was not included in the master thesis and was only tested during initial stages of development.
"""
function stop_early(s::StopAfterEpisode_or_Early, p::Union{A2CGAE, PPO, myVPG})::Bool
    if s.cur <= s.patience || isnothing(s.min_δ)
        return false
    end
    
    
    abs_loss = abs.(p.raw_loss)
    size = min(length(abs_loss) - 1, s.size)
    δ = mean(abs_loss[end-size:end])
    # δ = abs(s.reference - val)/ ((s.reference + val)/2)
    push!(p.δ, δ)
    s.reference = δ
    return δ < s.min_δ
end


@info "Stop when you wish"