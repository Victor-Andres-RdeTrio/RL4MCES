
"""
   (p::Replay)(env::MCES_Env)

Action selection function for the Replay policy. Returns prerecorded actions from historical data.

# Arguments
- `p::Replay`: The replay policy containing the historical action data
- `env::MCES_Env`: The Multi-Carrier Energy System environment (not used for action selection)

# Returns
- A 3-element vector containing the next set of actions: `[p_ev, p_bess, p_hp_e]`

# Details
1. Retrieves the action at the current index position
2. Increments the index for the next call
3. Returns zeros if all actions have been replayed (i.e., when index exceeds max_ind)
4. Prints a message when there are no more actions to replay

The function returns three control values representing:
- Electric vehicle power (p_ev)
- Battery energy storage system power (p_bess)
- Heat pump electrical power (p_hp_e)
"""
@inline function (p::Replay)(env::MCES_Env)
    i = p.ind
    if i > p.max_ind 
        println("The are no more actions to replay") 
        return zeros(3)
    end
    p.ind += Int32(1)
    
    [p.p_ev[i], p.p_bess[i], p.p_hp_e[i]]
end

# @inline function (p::Replay)(env::MCES_Env)
#     # Erasing the element after extraction. 
#     [popfirst!(p.p_ev), popfirst!(p.p_bess), popfirst!(p.p_hp_e)]
# end

function (ag::Agent{<:Replay})(::PreExperimentStage, env::MCES_Env)
    ag.policy.ind = 1
end

RLBase.optimise!(::Agent{<:Replay}) = nothing

function (ag::Agent{<:Replay})(::PostEpisodeStage, env::MCES_Env)
    ag.trajectory.container[] = true 
    empty!(ag.trajectory.container)
    nothing
end

function (ag::Agent{<:Replay})(::PostEpisodeStage, string::String)
    ag.trajectory.container[] = true 
    empty!(ag.trajectory.container)
    nothing
end

state_buffer_update!(ag::Agent{<:Replay}, env::MCES_Env) = nothing




@info "You can Replay"