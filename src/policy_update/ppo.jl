"""
    (p::PPO)(env::MCES_Env)

Sample actions from the actor network for the current state of the environment.

# Arguments
- `p::PPO`: The PPO policy.
- `env::MCES_Env`: The MCES environment.

# Returns
- A vector of sampled actions from Gaussian distributions defined by the actor network.

# Details
- Extracts the current state from the environment.
- Passes the state through the actor network to get mean and standard deviation values.
- Samples actions from the resulting Gaussian distributions using the policy's random number generator.
- Uses a Normal distribution by default. Custom distributions can be used via "action_distribution".
"""
@inline function (p::PPO)(env::MCES_Env)
    res = env |> state |> p.actor 
    rand.(p.rng, Gaussian_actions(res))  # Uses a Normal(). Use "action_distribution" to use a custom distribution. 
end


"""
    RLBase.optimise!(::Agent{<:PPO})

Empty method for optimizing a PPO agent. Actual optimization occurs after episode completion.

# Arguments
- `::Agent{<:PPO}`: The agent with a PPO policy.

# Returns
- `nothing`
"""
RLBase.optimise!(::Agent{<:PPO}) = nothing


"""
    (ag::Agent{<:PPO})(::PostEpisodeStage, env::MCES_Env)

Handles post-episode optimization for a PPO agent.

# Arguments
- `ag::Agent{<:PPO}`: The agent with a PPO policy.
- `::PostEpisodeStage`: Dispatch tag indicating post-episode processing.
- `env::MCES_Env`: The MCES environment.

# Details
- Marks the trajectory container as filled (ready for optimization).
- Optimizes the policy using the collected trajectory data.
- Empties the trajectory container for the next episode.
"""
function (ag::Agent{<:PPO})(::PostEpisodeStage, env::MCES_Env)
    ag.trajectory.container[] = true # to say it is filled
    optimise!(ag.policy, ag.trajectory.container)
    empty!(ag.trajectory.container)
end

"""
    (ag::Agent{<:PPO})(::PostEpisodeStage, string::String)

Handles post-episode stage for a PPO agent when running without learning.

# Arguments
- `ag::Agent{<:PPO}`: The agent with a PPO policy.
- `::PostEpisodeStage`: Dispatch tag indicating post-episode processing.
- `string::String`: A string parameter (likely used for identification or logging).

# Details
- Marks the trajectory container as filled.
- Empties the trajectory container without performing optimization.
"""
function (ag::Agent{<:PPO})(::PostEpisodeStage, string::String) # For running without learning
    ag.trajectory.container[] = true # to say it is filled
    empty!(ag.trajectory.container)
end


"""
    RLBase.optimise!(p::PPO, episode::Episode)

Optimizes the PPO policy using data from a completed episode.

# Arguments
- `p::PPO`: The PPO policy to optimize.
- `episode::Episode`: The completed episode containing states, actions, and rewards.

# Details
- Extracts states, actions, and rewards from the episode data.
- Performs multiple optimization epochs as specified by `p.epochs`.
- For each epoch:
  - Computes values for all states using the critic network.
  - Calculates Generalized Advantage Estimation (GAE) using rewards, values, and discount factors.
  - Shuffles and batches the data according to `p.batch_size`.
  - Calls the inner optimization function for each batch.
- Updates the old actor model with the current actor model state for future policy ratio calculations.
"""
function RLBase.optimise!(p::PPO, episode::Episode)
    states, actions, rewards = map(x -> Array(episode[x][:]), (:state, :action, :reward))

    for _ in 1:p.epochs # Advantage is recomputed per epoch, with updated critic. 
        values = vec(p.critic(states))
        advantages = gae_advantage(rewards[1: end-1], values, p.γ, p.λ)
        # Last step is not used to allow for GAE calculation with current episode.
        for inds in Iterators.partition(shuffle(p.rng, 1: length(advantages)), p.batch_size)
           optimise!(p, states[:, inds], actions[:, inds], advantages[inds], values[inds])
        end
    end
    Flux.loadmodel!(p.old_actor.model, Flux.state(p.actor.model))
end


"""
    RLBase.optimise!(p::PPO, states, actions, advantages, values)

Performs a single optimization step for both actor and critic networks.

# Arguments
- `p::PPO`: The PPO policy to optimize.
- `states`: Batch of states.
- `actions`: Batch of actions.
- `advantages`: Batch of advantage values.
- `values`: Batch of critic values for the states.

# Details
- Calculates critic targets by adding advantages to values.
- Computes gradient for the critic network to minimize mean squared error between targets and predictions.
- Logs critic loss if memory-safe mode is disabled.
- Computes policy gradient for the actor network using PPO's clipped objective.
- Applies gradient clipping if `p.max_grad_norm` is specified.
- Updates both actor and critic networks using their respective optimizers.
"""
function RLBase.optimise!(p::PPO, states, actions, advantages, values)
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
        push!(p.actor_norm, clip_by_global_norm!(gs_a, Flux.params(A), p.max_grad_norm, noise = false))
        push!(p.critic_norm, clip_by_global_norm!(gs_c, Flux.params(C), p.max_grad_norm, noise = false))
    elseif !isnothing(p.max_grad_norm) && p.mem_safe
        clip_by_global_norm!(gs_a, Flux.params(A), p.max_grad_norm)
        clip_by_global_norm!(gs_c, Flux.params(C), p.max_grad_norm)
    end

    optimise!(A, gs_a)
    optimise!(C, gs_c)
end

"""
    policy_gradient_estimate(p::PPO, states, actions, advantage)

Computes the policy gradient for the actor network using PPO's clipped objective function.

# Arguments
- `p::PPO`: The PPO policy.
- `states`: Batch of states (first dimension is state dimension, second is batch size).
- `actions`: Batch of actions (first dimension is action dimension, second is batch size).
- `advantage`: Batch of advantage values (vector with length equal to batch size).

# Returns
- The computed gradients for the actor network parameters.

# Details
- Normalizes advantage values.
- Computes log probabilities of actions under the old policy.
- Computes log probabilities of actions under the current policy.
- Calculates probability ratios between new and old policies.
- Applies PPO's clipped surrogate objective:
  - Unclipped objective: -ratio * advantage
  - Clipped objective: -clamp(ratio, 1-clip, 1+clip) * advantage
  - Takes the maximum (worse) of these two objectives
- Adds entropy bonus to encourage exploration.
- Logs various metrics if not in memory-safe mode, including:
  - Approximate KL divergence
  - Clipping fraction
  - Actor and entropy losses
  - Means and standard deviations
- Returns the gradients for the actor network parameters.
"""
function policy_gradient_estimate(p::PPO, states, actions, advantage)
    # states, actions and advantage should be Float32. The second dimension of states and actions (and the only dimension of advantage) depends on the batch size.
    # example of sizes: states (11,24), actions (3,24), advantage (24,)
    advantage_norm = normalise(advantage)
    
    w_a = p.actor_loss_weight
    w_e = p.entropy_loss_weight
    clip = p.clip_coef

    # Log Probabilities of actions for not updated policy.
    old_means_and_stds = p.old_actor(states)
    old_normal_d_reshaped = reshape(Gaussian_actions(old_means_and_stds), size(actions)[1], :)  # size(3, batchsize)
    old_logprob = logpdf.(old_normal_d_reshaped, actions)
    
    if !p.mem_safe
        local actor_loss
        local entropy_loss
        local means_and_stds
        local loss
        local logratio
        local ratio
    end

    gs = gradient(Flux.params(p.actor)) do
        means_and_stds = p.actor(states) # size (2, batchsize*3)
        normal_d_reshaped = reshape(Gaussian_withgrad(means_and_stds), size(actions)[1], :)  # size(3, batchsize)
        # Assumes actions will be sampled from a Gaussian. Use "action_distribution_withgrad" to use a custom distribution. 
        logprob = logpdf.(normal_d_reshaped, actions)
        logratio = logprob .- old_logprob
        ratio = exp.(logratio)
        act_loss_1 = - ratio .* advantage_norm'   # size (3,batchsize) .* (1,batchsize) = (3,batchsize)
        act_loss_2 = - clamp.(ratio, 1f0 - clip, 1f0 + clip) .* advantage_norm'   # size (3,batchsize) .* (1,batchsize)
        actor_loss = mean(max.(act_loss_1, act_loss_2)) # average actor loss per batch
        entropy_loss = avg_entropy(means_and_stds[2,:])
        loss = w_a * actor_loss - w_e * entropy_loss
    end
    
    # Tracking
    if !p.mem_safe
        push!(p.avg_adv, mean(advantage))
        push!(p.actor_loss, actor_loss)
        push!(p.entropy_loss, entropy_loss)
        push!(p.loss, loss)
        a_loss1 = - ratio .* advantage' 
        a_loss2 = - clamp.(ratio, 1f0 - clip, 1f0 + clip) .* advantage'
        push!(p.raw_loss, mean(max.(a_loss1, a_loss2)))
        push!(p.mean, means_and_stds[1,:]...)
        push!(p.std, means_and_stds[2,:]...)
        push!(p.approx_kl, mean((ratio .- 1) - logratio))
        push!(p.clip_fracs, mean(abs.(ratio .- 1) .> clip))
    end

    gs
end

@info "PPO ready to run"

