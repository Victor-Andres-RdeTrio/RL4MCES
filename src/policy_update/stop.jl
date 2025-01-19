@kwdef mutable struct StopAfterEpisode_or_Early
    episode::Int32 = Int32(1)
    cur::Int32 = Int32(0)
    patience::Int32 = Int32(1)
    reference::Float32 = 0f0
    size::Int32 = Int32(20)
    min_δ::Union{Float32, Nothing} = nothing
    stop::Bool = false
end

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

function (s::StopAfterEpisode_or_Early)(agent, env)

    if is_terminated(env)
        s.cur += 1
        s.stop = stop_early(s, agent.policy)
        s.stop && println("Early Stop!!! On Episode $(s.cur), with last reference value: $(s.reference)")
    end
    
    s.cur >= s.episode || s.stop
end

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