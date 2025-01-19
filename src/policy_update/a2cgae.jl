@inline function (p::A2CGAE)(env::MCES_Env)
    res = env |> state |> p.actor 
    rand.(p.rng, Gaussian_actions(res))
end

RLBase.optimise!(::Agent{<:A2CGAE}) = nothing

function (ag::Agent{<:A2CGAE})(::PostEpisodeStage, env::MCES_Env)
    ag.trajectory.container[] = true # to indicate it is filled
    optimise!(ag.policy, ag.trajectory.container)
    empty!(ag.trajectory.container)
end

function (ag::Agent{<:A2CGAE})(::PostEpisodeStage, string::String) # For running without learning
    ag.trajectory.container[] = true 
    empty!(ag.trajectory.container)
end

function RLBase.optimise!(p::A2CGAE, episode::Episode)
    states, actions, rewards = map(x -> Array(episode[x][:]), (:state, :action, :reward))

    for _ in 1:p.epochs # Advantage is recomputed per epoch, with updated critic. 
        values = vec(p.critic(states))
        advantages = gae_advantage(rewards[1: end-1], values, p.γ, p.λ)
        # Last step is not used to allow for GAE calculation with current episode.
        for inds in Iterators.partition(shuffle(p.rng, 1: length(advantages)), p.batch_size)
           optimise!(p, states[:, inds], actions[:, inds], advantages[inds], values[inds])
        end
    end
end

function RLBase.optimise!(p::A2CGAE, states, actions, advantages, values)
    A = p.actor
    C = p.critic
    critic_targets = advantages + values 
    
    local critic_loss
    gs_c = gradient(Flux.params(C)) do
        δ = critic_targets - vec(C(states))
        critic_loss = mean(δ .^ 2)
    end
    
    if !p.mem_safe
    push!(p.critic_loss, critic_loss) # Just Logging
    end

    gs_a = policy_gradient_estimate(p, states, actions, advantages)

    if !isnothing(p.max_grad_norm) && !p.mem_safe # Applies gradient clipping and stores the global norm
        push!(p.actor_norm, clip_by_global_norm!(gs_a, Flux.params(A), p.max_grad_norm))
        push!(p.critic_norm, clip_by_global_norm!(gs_c, Flux.params(C), p.max_grad_norm))
    elseif !isnothing(p.max_grad_norm) && p.mem_safe
        clip_by_global_norm!(gs_a, Flux.params(A), p.max_grad_norm)
        clip_by_global_norm!(gs_c, Flux.params(C), p.max_grad_norm)
    end

    optimise!(A, gs_a)
    optimise!(C, gs_c)
end

function policy_gradient_estimate(p::A2CGAE, states, actions, advantage)
    # states, actions and advantage should be Float32. The second dimension of states and actions (and the only dimension of advantage) depends on the batch size.
    # example of sizes: states (11,24), actions (3,24), advantage (24,)
    w_a = p.actor_loss_weight
    w_e = p.entropy_loss_weight
    
    advantage_norm = normalise(advantage)
    
    if !p.mem_safe
        local actor_loss
        local entropy_loss
        local means_and_stds
        local loss
        local logprob
    end
    
    gs = gradient(Flux.params(p.actor)) do
        means_and_stds = p.actor(states) # size (2, batchsize*3)
        normal_d_reshaped = reshape(Gaussian_withgrad(means_and_stds), size(actions)[1], :)  # size(3, batchsize)
        logprob = logpdf.(normal_d_reshaped, actions)
        sum_of_loss = logprob .* advantage_norm'   # size (3,batchsize) .* (1,batchsize) = (3,batchsize)
        actor_loss = - mean(sum_of_loss) # average actor loss per batch
        entropy_loss = avg_entropy(means_and_stds[2,:])
        loss = w_a * actor_loss - w_e * entropy_loss
    end
    
    # Tracking
    if !p.mem_safe
        push!(p.avg_adv, mean(advantage))
        push!(p.actor_loss, actor_loss)
        push!(p.entropy_loss, entropy_loss)
        push!(p.loss, loss)
        push!(p.raw_loss, w_a * mean(-logprob .* advantage') - w_e * entropy_loss)
        push!(p.mean, means_and_stds[1,:]...)
        push!(p.std, means_and_stds[2,:]...)
    end

    gs
end

@info "A2CGAE can run"

