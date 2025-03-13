"""
    (p::A2CGAE)(env::MCES_Env)

Sample actions from the actor network for the current state of the environment.

# Arguments
- `p::A2CGAE`: The A2C-GAE policy.
- `env::MCES_Env`: The MCES environment.

# Returns
- A vector of sampled actions from Gaussian distributions defined by the actor network.

# Details
1. Extracts the current state from the environment.
2. Passes the state through the actor network to get mean and standard deviation values.
3. Samples actions from the resulting Gaussian distributions using the policy's random number generator.
"""
@inline function (p::A2CGAE)(env::MCES_Env)
    res = env |> state |> p.actor 
    rand.(p.rng, Gaussian_actions(res))
end

RLBase.optimise!(::Agent{<:A2CGAE}) = nothing

"""
    (ag::Agent{<:A2CGAE})(::PostEpisodeStage, env::MCES_Env)

Handles post-episode optimization for an A2C-GAE agent.

# Arguments
- `ag::Agent{<:A2CGAE}`: The agent with an A2C-GAE policy.
- `::PostEpisodeStage`: Dispatch tag indicating post-episode processing.
- `env::MCES_Env`: The MCES environment.

# Details
1. Marks the trajectory container as filled (ready for optimization).
2. Optimizes the policy using the collected trajectory data.
3. Empties the trajectory container for the next episode.
"""
function (ag::Agent{<:A2CGAE})(::PostEpisodeStage, env::MCES_Env)
    ag.trajectory.container[] = true 
    optimise!(ag.policy, ag.trajectory.container)
    empty!(ag.trajectory.container)
end

"""
    (ag::Agent{<:A2CGAE})(::PostEpisodeStage, string::String)

Handles post-episode stage for an A2C-GAE agent when running without learning.

# Arguments
- `ag::Agent{<:A2CGAE}`: The agent with an A2C-GAE policy.
- `::PostEpisodeStage`: Dispatch tag indicating post-episode processing.
- `string::String`: A string parameter (likely used for identification or logging).

# Details
1. Marks the trajectory container as filled.
2. Empties the trajectory container without performing optimization.
"""
function (ag::Agent{<:A2CGAE})(::PostEpisodeStage, string::String) # For running without learning
    ag.trajectory.container[] = true 
    empty!(ag.trajectory.container)
end


"""
    RLBase.optimise!(p::A2CGAE, episode::Episode)

Optimizes the A2C-GAE policy using data from a completed episode.

# Arguments
- `p::A2CGAE`: The A2C-GAE policy to optimize.
- `episode::Episode`: The completed episode containing states, actions, and rewards.

# Details
1. Extracts states, actions, and rewards from the episode data.
2. Performs multiple optimization epochs as specified by `p.epochs`.
3. For each epoch:
   - Computes values for all states using the critic network.
   - Calculates Generalized Advantage Estimation (GAE) using rewards, values, and discount factors.
   - Shuffles and batches the data according to `p.batch_size`.
   - Calls the inner optimization function for each batch.
"""
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


"""
    RLBase.optimise!(p::A2CGAE, states, actions, advantages, values)

Performs a single optimization step for both actor and critic networks.

# Arguments
- `p::A2CGAE`: The A2C-GAE policy to optimize.
- `states`: Batch of states.
- `actions`: Batch of actions.
- `advantages`: Batch of advantage values.
- `values`: Batch of critic values for the states.

# Details
1. Calculates critic targets by adding advantages to values.
2. Computes gradient for the critic network to minimize mean squared error between targets and predictions.
3. Logs critic loss if memory-safe mode is disabled.
4. Computes policy gradient for the actor network.
5. Applies gradient clipping if `p.max_grad_norm` is specified.
6. Updates both actor and critic networks using their respective optimizers.
"""
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


"""
    policy_gradient_estimate(p::A2CGAE, states, actions, advantage)

Computes the policy gradient for the actor network.

# Arguments
- `p::A2CGAE`: The A2C-GAE policy.
- `states`: Batch of states (first dimension is state dimension, second is batch size).
- `actions`: Batch of actions (first dimension is action dimension, second is batch size).
- `advantage`: Batch of advantage values (vector with length equal to batch size).

# Returns
- The computed gradients for the actor network parameters.

# Details
1. Normalizes advantage values.
2. Computes actor network outputs (means and standard deviations) for the input states.
3. Constructs Gaussian distributions and calculates log probabilities of the actions.
4. Computes actor loss as the negative mean of log probabilities weighted by advantages.
5. Calculates entropy loss.
6. Combines actor and entropy losses according to their weights.
7. Logs various metrics if not in memory-safe mode.
8. Returns the gradients for the actor network parameters.
"""
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

