
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


function fake_replay(length)
    Replay(
        rand(-5.0:5.0, length),
        rand(-5.0:5.0, length),
        rand(-5.0:5.0, length),
        1,
        length
    )
end

function fake_badreplay(length)
    Replay(
        rand(-15.0:-5.0, length),
        rand(-15.0:-5.0, length),
        rand(-15.0:-5.0, length),
        1,
        length
    )
end


@info "You can Replay"