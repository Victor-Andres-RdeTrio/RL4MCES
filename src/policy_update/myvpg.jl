# Official Name: VPG with Critic (unofficially it is already an Advantage Actor Critic)

##########################################################################################
# Modifying the behaviour of VPG policy. 
"""
    (π::myVPG)(env::MCES_Env)

Sample actions from the actor network for the current state of the environment.

# Arguments
- `π::myVPG`: The myVPG policy.
- `env::MCES_Env`: The MCES environment.

# Returns
- A vector of sampled actions from Gaussian distributions defined by the actor network.

# Details
- Extracts the current state from the environment.
- Passes the state through the actor network to get mean and standard deviation values.
- Samples actions from the resulting Gaussian distributions using the policy's random number generator.
"""
@inline function (π::myVPG)(env::MCES_Env)
    res = env |> state |> π.actor 
    rand.(π.rng, Gaussian_actions(res))
end

"""
    RLBase.optimise!(::Agent{<:myVPG})

Empty method for optimizing a myVPG agent. Actual optimization occurs after episode completion.

# Arguments
- `::Agent{<:myVPG}`: The agent with a myVPG policy.

# Returns
- `nothing`
"""
RLBase.optimise!(::Agent{<:myVPG}) = nothing

"""
    (ag::Agent{<:myVPG})(::PostEpisodeStage, env::MCES_Env)

Handles post-episode optimization for a myVPG agent.

# Arguments
- `ag::Agent{<:myVPG}`: The agent with a myVPG policy.
- `::PostEpisodeStage`: Dispatch tag indicating post-episode processing.
- `env::MCES_Env`: The MCES environment.

# Details
- Marks the trajectory container as filled (ready for optimization).
- Optimizes the policy using the collected trajectory data.
- Empties the trajectory container for the next episode.
"""
function (ag::Agent{<:myVPG})(::PostEpisodeStage, env::MCES_Env)
    ag.trajectory.container[] = true # to say it is filled
    optimise!(ag.policy, ag.trajectory.container)
    empty!(ag.trajectory.container)
end

"""
    (ag::Agent{<:myVPG})(::PostEpisodeStage, string::String)

Handles post-episode stage for a myVPG agent when running without learning.

# Arguments
- `ag::Agent{<:myVPG}`: The agent with a myVPG policy.
- `::PostEpisodeStage`: Dispatch tag indicating post-episode processing.
- `string::String`: A string parameter (likely used for identification or logging).

# Details
- Marks the trajectory container as filled.
- Empties the trajectory container without performing optimization.
"""
function (ag::Agent{<:myVPG})(::PostEpisodeStage, string::String) # For running without learning
    ag.trajectory.container[] = true # to say it is filled
    empty!(ag.trajectory.container)
end

"""
    RLBase.optimise!(π::myVPG, episode::Episode)

Optimizes the myVPG policy using data from a completed episode.

# Arguments
- `π::myVPG`: The myVPG policy to optimize.
- `episode::Episode`: The completed episode containing states, actions, and rewards.

# Details
- Extracts states, actions, and rewards from the episode data.
- Performs multiple optimization epochs as specified by `π.epochs`.
- For each epoch:
  - Uses the critic to evaluate the value of the final state.
  - Calculates discounted returns using `mydiscount_rewards` with the final state value.
  - Shuffles and batches the data according to `π.batch_size`.
  - Calls the inner optimization function for each batch.
"""
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

"""
    RLBase.optimise!(p::myVPG, states, actions, gain)

Performs a single optimization step for both actor and critic networks.

# Arguments
- `p::myVPG`: The myVPG policy to optimize.
- `states`: Batch of states.
- `actions`: Batch of actions.
- `gain`: Batch of discounted returns.

# Details
- Computes the TD error (δ) as the difference between discounted returns and critic predictions.
- Updates the critic to minimize the squared TD error.
- Updates the actor using the policy gradient with the TD error as advantage estimates.
- Tracks metrics if memory-safe mode is disabled, including:
  - Critic loss
  - Actor and critic gradient norms
"""
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

"""
    RLBase.optimise!(A::Approximator, gs)

Updates the parameters of an approximator using its optimizer and the provided gradients.

# Arguments
- `A::Approximator`: The approximator (actor or critic network).
- `gs`: The computed gradients for the approximator's parameters.

# Details
- Applies the optimizer's update rule to the approximator's parameters using the provided gradients.
"""
function RLBase.optimise!(A::Approximator, gs) 
    Flux.Optimise.update!(A.optimiser, Flux.params(A), gs)
end

"""
    policy_gradient_estimate(p::myVPG, states, actions, advantage)

Computes the policy gradient for the actor network using the REINFORCE algorithm with advantage baseline.

# Arguments
- `p::myVPG`: The myVPG policy.
- `states`: Batch of states (first dimension is state dimension, second is batch size).
- `actions`: Batch of actions (first dimension is action dimension, second is batch size).
- `advantage`: Batch of TD errors used as advantage estimates (vector with length equal to batch size).

# Returns
- The computed gradients for the actor network parameters.

# Details
- Normalizes advantage values.
- Computes the actor network outputs (means and standard deviations) for the input states.
- Constructs Gaussian distributions and calculates log probabilities of the actions.
- Computes actor loss as the negative mean of log probabilities weighted by normalized advantages.
- Logs various metrics if not in memory-safe mode, including:
  - Actor loss
  - Raw loss (unnormalized advantages)
  - Average advantage
  - Mean and standard deviation outputs from the actor
"""
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