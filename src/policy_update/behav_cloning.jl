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

######################################################################
# Some functionality to clone the behaviour of the MPC expert
# rhDict file is the common output of the Expert, which concatenates the states and decisions over a simulation. 

function get_actions(rhDict::Dict)
    required_fields = ["Pbess", "Pev[1]", "Phpe", "γ_cont"]
    
    # Error Checking
    for field in required_fields
        if !haskey(rhDict, field)
            error("Missing required field: $field")
        end
    end

    Pbess = get(rhDict, "Pbess", Float32[])
    Pev = get(rhDict, "Pev[1]", Float32[])
    Phpe = get(rhDict, "Phpe", Float32[])
    γ_cont = get(rhDict, "γ_cont", Float32[])
    
    if !isempty(γ_cont) && !isempty(Pev) && length(γ_cont) == length(Pev)
        processed_Pev = γ_cont .* Pev
    else
        processed_Pev = Float32[]
    end
    
    # Create a matrix with 3 rows
    action_matrix = zeros(Float32, 3, length(Pbess))
    action_matrix[1, :] = processed_Pev
    action_matrix[2, :] = Pbess
    action_matrix[3, :] = Phpe
    
    return action_matrix
end


function get_socs(rhDict::Dict)
    # List of required fields
    required_fields = ["SoCtess", "SoCbess", "SoCev[1]", "γ_cont"]
    
    # Error Checking
    for field in required_fields
        if !haskey(rhDict, field)
            error("Missing required field: $field")
        end
    end

    # Extract the relevant fields
    SoCtess = get(rhDict, "SoCtess", Float32[])
    SoCbess = get(rhDict, "SoCbess", Float32[])
    SoCev = get(rhDict, "SoCev[1]", Float32[])
    γ_cont = get(rhDict, "γ_cont", Float32[])

    # Create a matrix with 4 rows, each representing one of the state variables
    state_matrix = zeros(Float32, 4, length(SoCtess))
    state_matrix[1, :] = SoCtess
    state_matrix[2, :] = SoCbess
    state_matrix[3, :] = SoCev
    state_matrix[4, :] = γ_cont
    
    return state_matrix
end

function get_exog_normalized(exog::Exogenous_BatchCollection)
    load_e_normalized = z_score(exog.load_e, 0.342, 0.203)
    load_th_normalized = z_score(exog.load_th, 0.655, 0.405)
    pv_normalized = z_score(exog.pv, 0.528, 0.873)
    λ_buy_normalized = z_score(exog.λ_buy, 0.242, 0.128)
    λ_sell_normalized = z_score(exog.λ_sell, 0.231, 0.121)

    normalized_matrix = zeros(Float32, 5, length(exog.load_e))
    normalized_matrix[1, :] = load_e_normalized
    normalized_matrix[2, :] = load_th_normalized
    normalized_matrix[3, :] = pv_normalized
    normalized_matrix[4, :] = λ_buy_normalized
    normalized_matrix[5, :] = λ_sell_normalized

    return normalized_matrix
end

function get_t_ep(ep_length::Int, episodes::Int)
    ep = [i/ep_length for i in 1:ep_length]
    Float32.(repeat(ep, episodes))
end

function get_t_year(length::Int)
    ep = [i/35040 for i in 1:length]
    Float32.(ep)
end

function get_full_state(exog::Exogenous_BatchCollection, rhDict::Dict, ep_length::Int)
    # Extract SoCs and γ_cont
    socs_matrix = get_socs(rhDict)
    max_step = length(socs_matrix[1,:])   # The rhDict is the limiting factor

    # Extract and normalize exogenous values
    normalized_exog = get_exog_normalized(exog)[:, 1:max_step]
    
    # Calculate t_ep_ratio and t_year_ratio
    episodes = div(max_step, ep_length)
    t_ep_ratio = get_t_ep(ep_length, episodes)
    # t_year_ratio = get_t_year(max_step)
    
    # Combine all into an 11-row matrix
    full_state_matrix = vcat(
        normalized_exog, 
        socs_matrix, 
        reshape(t_ep_ratio, 1, :)
    )
    
    return full_state_matrix
end

function save_expert_data(filepath::String, actions, states, hook)
    jldsave(filepath, actions = actions, states = states, rewards = hook) # Rewards can be found and weighted within the hook. 
end

function load_expert_data(filepath::String)
    @load filepath actions states rewards 
    println("Output is ordered -> actions, states, rewards")
    actions, states, rewards
end




@info "Learning to Imitate"