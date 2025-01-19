# Official Name: VPG with Critic (unofficially it is already an Advantage Actor Critic)

##########################################################################################
# Modifying the behaviour of VPG policy. 

@inline function (π::myVPG)(env::MCES_Env)
    res = env |> state |> π.actor 
    rand.(π.rng, Gaussian_actions(res))
end

RLBase.optimise!(::Agent{<:myVPG}) = nothing

function (ag::Agent{<:myVPG})(::PostEpisodeStage, env::MCES_Env)
    ag.trajectory.container[] = true # to say it is filled
    optimise!(ag.policy, ag.trajectory.container)
    empty!(ag.trajectory.container)
end

function (ag::Agent{<:myVPG})(::PostEpisodeStage, string::String) # For running without learning
    ag.trajectory.container[] = true # to say it is filled
    empty!(ag.trajectory.container)
end

function RLBase.optimise!(π::myVPG, episode::Episode)
    states, actions, rewards = map(x -> Array(episode[x][:]), (:state, :action, :reward))
    for _ in 1:π.epochs
        V_last_state = π.critic(states[:, end])
        gain = mydiscount_rewards(rewards, π.γ, V_last_state[])
            for inds in Iterators.partition(shuffle(π.rng, 1:length(episode)), π.batch_size)
                optimise!(π, states[:, inds], actions[:, inds], gain[inds])
            end
    end
end

function RLBase.optimise!(p::myVPG, states, actions, gain)
    A = p.actor
    C = p.critic
    
    local δ
    gs_c = gradient(Flux.params(C)) do
        δ = gain - vec(C(states))
        mean(δ .^ 2)
    end
    
    gs_a = policy_gradient_estimate(p, states, actions, δ)

    if !p.mem_safe
        push!(p.critic_loss, mean(δ .^ 2))
        push!(p.actor_norm, global_norm(gs_a, Flux.params(A)))
        push!(p.critic_norm, global_norm(gs_c, Flux.params(C)))
    end

    optimise!(C, gs_c)
    optimise!(A, gs_a)

end

function RLBase.optimise!(A::Approximator, gs) 
    Flux.Optimise.update!(A.optimiser, Flux.params(A), gs)
end

function policy_gradient_estimate(p::myVPG, states, actions, advantage)
    # states, actions and advantage should be Float32. The second dimension of states and actions (and the only dimension of advantage) depends on the batch size.
    # example of sizes: states (11,24), actions (3,24), advantage (24,)
    advantage_norm = normalise(advantage)
    
    if !p.mem_safe
        local actor_loss
        local logprob
        local res
    end
    
    gs = gradient(Flux.params(p.actor)) do
        res = p.actor(states)
        normal_d_reshaped = reshape(Gaussian_withgrad(res), size(actions)[1], :) # size(3, batchsize)
        logprob = logpdf.(normal_d_reshaped, actions)
        actor_loss = -mean(logprob .* advantage_norm')
    end

    # Tracking
    if !p.mem_safe
        push!(p.actor_loss, actor_loss)
        push!(p.raw_loss, -mean(logprob .* advantage'))
        push!(p.avg_adv, mean(advantage))
        push!(p.mean, res[1,:]...)
        push!(p.std, res[2,:]...)
    end

    gs
end

@info "myVPG can run"