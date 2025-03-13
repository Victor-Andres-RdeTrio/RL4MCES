"""
    clone_bhv!(bc::Clone_Policy, epochs::Int = 1)

Train a policy to imitate expert behavior through supervised learning.

# Arguments
- `bc::Clone_Policy`: The behavior cloning policy wrapper.
- `epochs::Int`: Number of training epochs (default: 1).

# Returns
- The updated `Clone_Policy` object.

# Details
1. Verifies that expert states and actions have matching dimensions.
2. For each epoch:
   - Shuffles expert data.
   - Batches the data according to `bc.batch_size`.
   - Trains the actor to mimic expert actions via `clone_actor!`.
3. Verifies that expert states and rewards have matching dimensions.
4. For each epoch:
   - Batches the data.
   - Trains the critic to predict expected returns via `clone_critic!`.

# Warning
This behavior cloning implementation was not included in the master thesis and was only tested during initial stages of development.
"""
function clone_bhv!(bc::Clone_Policy, epochs::Int = 1)
    S = bc.expert_states
    A = bc.expert_actions
    rew = bc.expert_rewards
    @assert size(S)[2] == size(A)[2] "Different number of examples of states and actions"
    
    for _ in Base.OneTo(epochs)
        rand_ind = randperm(Int(size(S)[2]))  # Shuffle the columns of expert data. 
        for inds in Iterators.partition(rand_ind, bc.batch_size)
            states = S[:, inds]
            actions = A[:, inds]
            clone_actor!(bc, states, actions)
        end
    end

    @assert size(S)[2] == length(rew) "Different number of examples of states and rewards"
    for _ in Base.OneTo(epochs)
        for inds in Iterators.partition(eachindex(rew), bc.batch_size)
            states = S[:, inds]
            rewards = rew[inds]
            clone_critic!(bc, states, rewards)
        end
    end

    bc
end

"""
    clone_actor!(bc::Clone_Policy{T}, states, actions) where{T<:Union{A2CGAE,PPO,myVPG}}

Trains the actor network to predict expert actions from states.

# Arguments
- `bc::Clone_Policy{T}`: The behavior cloning policy wrapper.
- `states`: Batch of expert states.
- `actions`: Batch of expert actions corresponding to the states.

# Details
1. Computes the actor network outputs (means and standard deviations) for the expert states.
2. Constructs Gaussian distributions and calculates log probabilities of the expert actions.
3. Computes actor loss as the negative mean of log probabilities (maximum likelihood).
4. Adds entropy regularization to encourage exploration.
5. Applies gradient clipping using `bc.max_grad_norm`.
6. Updates the actor network parameters.
7. Logs various metrics including actor loss, entropy loss, means, and standard deviations.

# Warning
This function was not included in the master thesis and was only tested during initial stages of development.
"""
function clone_actor!(bc::Clone_Policy{T}, states, actions) where{T<:Union{A2CGAE,PPO,myVPG}}  
    actor = bc.learner.actor
    w_e = bc.entropy_loss_weight
    
    local means_and_stds
    local actor_loss
    local entropy_loss
    gs = gradient(Flux.params(actor)) do
        means_and_stds = actor(states)
        normal_d_reshaped = reshape(
            Gaussian_withgrad(means_and_stds), 
            size(actions)[1], :)                      # size(3, batchsize)
        logprob = logpdf.(normal_d_reshaped, actions) # size(3, batchsize)
        actor_loss = - mean(logprob)                  # actor loss per batch
        entropy_loss = avg_entropy(means_and_stds[2,:])
        actor_loss - w_e * entropy_loss
    end

    clip_by_global_norm!(gs, Flux.params(actor), bc.max_grad_norm)
    optimise!(actor, gs)

    # Bookkeeping
    push!(bc.actor_loss, actor_loss)
    push!(bc.entropy_loss, entropy_loss)
    push!(bc.mean, means_and_stds[1,:]...)
    push!(bc.std, means_and_stds[2,:]...)

    return nothing
end

"""
    clone_critic!(bc::Clone_Policy{T}, states, rewards::Vector) where{T<:Union{A2CGAE,PPO,myVPG}}

Trains the critic network to predict the expected return from states.

# Arguments
- `bc::Clone_Policy{T}`: The behavior cloning policy wrapper.
- `states`: Batch of expert states.
- `rewards`: Vector of rewards corresponding to the states.

# Details
1. Extracts the discount factor (`γ`) from the wrapped learning policy if available.
2. Calculates discounted returns using `discount_rewards`.
3. Computes the critic network predictions for the expert states.
4. Computes critic loss as the mean squared error between predictions and discounted returns.
5. Applies gradient clipping using `bc.max_grad_norm`.
6. Updates the critic network parameters.
7. Logs the critic loss.

# Warning
This function was not included in the master thesis and was only tested during initial stages of development.
"""
function clone_critic!(bc::Clone_Policy{T}, states, rewards::Vector)  where{T<:Union{A2CGAE,PPO,myVPG}}
    critic = bc.learner.critic
    γ = hasfield(typeof(bc.learner), :γ) ? bc.learner.γ : 0
    gains = discount_rewards(rewards, γ)

    local δ
    gs = gradient(Flux.params(critic)) do
        δ = gains - vec(critic(states))
        mean(δ .^ 2)
    end

    clip_by_global_norm!(gs, Flux.params(critic), bc.max_grad_norm)
    optimise!(critic, gs)

    push!(bc.critic_loss, mean(δ .^ 2))
    return nothing
end

@info "Learning to Imitate"