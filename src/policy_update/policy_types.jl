


#########################################################################################
# Custom NN

"""
    Custom_NN{M,S}

A custom neural network structure for reinforcement learning policies that outputs both mean and standard deviation.

# Fields
- `π_mean::M`: Neural network module for predicting action means
- `π_std::S`: Neural network module for predicting action standard deviations

# Details
This neural network structure is designed for stochastic policies in reinforcement learning. 
It outputs concatenated mean and standard deviation values, which can be used to parameterize 
probability distributions for action selection.

The network processes inputs differently for mean and standard deviation:
- Mean values are calculated by applying `π_mean` to the input state `x`
- Standard deviation values use constant input of ones with the same shape as the state. This effectively disconnects the standard deviation from the current state of the environment.

The outputs are concatenated into a single array with dimensions `[2, batch_size]`.
"""
@kwdef struct Custom_NN{M,S}
    π_mean::M
    π_std::S
end

@functor Custom_NN

@inline function (nn::Custom_NN)(x::AbstractArray)
    Float32.(vcat(
        reshape(nn.π_mean(x),1,:), 
        reshape(nn.π_std(ones32(size(x))),1,:)
    ))
end



#########################################################################################
# MyVPG

"""
    myVPG{A,C,D,V} <: AbstractPolicy

Vanilla Policy Gradient with Critic implementation.

# Fields
- `actor::A`: Neural network that outputs action distribution parameters
- `critic::C`: Neural network that estimates state values
- `γ::Float32 = 0.99f0`: Discount factor for future rewards
- `dist::D = Normal`: Probability distribution for stochastic policy (defaults to Normal)
- `batch_size::Int64 = 24`: Number of samples to process in each training batch
- `epochs::Int64 = 1`: Number of epochs to train on each batch of data
- `rng::AbstractRNG = Random.default_rng()`: Random number generator
- `online_stats::Normaliser = OnlineNorm(14)`: Tracks statistics for state normalization

## Logging fields
- `mem_safe::Bool = false`: Flag for memory efficient operation
- `actor_loss::V`: Stores actor loss values during training
- `critic_loss::V`: Stores critic loss values during training
- `avg_adv::V`: Stores average advantage values
- `raw_loss::V`: Stores raw loss values before normalization
- `actor_norm::V`: Stores actor gradient norms
- `critic_norm::V`: Stores critic gradient norms
- `mean::V`: Stores mean values of action distributions
- `std::V`: Stores standard deviation values of action distributions
- `δ::V`: Stores temporal difference values

# Details
This implementation of Vanilla Policy Gradient maintains separate networks for the policy (actor) 
and value function (critic). It uses the specified probability distribution for action selection 
and includes fields for tracking various metrics during training.
"""
@kwdef mutable struct myVPG{A,C,D,V} <: AbstractPolicy
    actor::A
    critic::C
    γ::Float32 = 0.99f0
    dist::D = Normal
    batch_size::Int64 = 24
    epochs::Int64 = 1
    rng::AbstractRNG = Random.default_rng()
    online_stats::Normaliser = OnlineNorm(14) # Keeps track of the current mean and variance of the state information.

    # for logging
    mem_safe::Bool = false
    actor_loss::V = sizehint!(Float32[], 5000)
    critic_loss::V = sizehint!(Float32[], 5000)
    avg_adv::V = sizehint!(Float32[], 5000)
    raw_loss::V = sizehint!(Float32[], 5000)
    actor_norm::V = sizehint!(Float32[], 5000)
    critic_norm::V = sizehint!(Float32[], 5000)
    mean::V = sizehint!(Float32[], 350000)
    std::V = sizehint!(Float32[], 350000)
    δ::V = sizehint!(Float32[], 5000)
end

#########################################################################################
# A2CGAE

"""
    A2CGAE{A,C,D,V} <: AbstractPolicy

Advantage Actor-Critic with Generalized Advantage Estimation implementation.

# Fields
- `actor::A`: Neural network that outputs action distribution parameters
- `critic::C`: Neural network that estimates state values
- `γ::Float32 = 0.99f0`: Discount factor for future rewards
- `λ::Float32 = 0.9f0`: GAE lambda parameter for advantage estimation
- `dist::D = Normal`: Probability distribution for stochastic policy (defaults to Normal)
- `max_grad_norm::Union{Nothing,Float32} = nothing`: Maximum gradient norm for clipping
- `actor_loss_weight::Float32 = 1f0`: Weight for the actor loss component
- `entropy_loss_weight::Float32 = 0.5f0`: Weight for the entropy loss component
- `batch_size::Int64 = 24`: Number of samples to process in each training batch
- `epochs::Int64 = 1`: Number of epochs to train on each batch of data
- `rng::AbstractRNG = Random.default_rng()`: Random number generator
- `online_stats::Normaliser = OnlineNorm(14)`: Tracks statistics for state normalization

## Logging fields
- `mem_safe::Bool = false`: Flag for memory efficient operation
- `actor_loss::V`: Stores actor loss values during training
- `critic_loss::V`: Stores critic loss values during training
- `entropy_loss::V`: Stores entropy loss values
- `loss::V`: Stores combined loss values
- `raw_loss::V`: Stores raw loss values before normalization
- `actor_norm::V`: Stores actor gradient norms
- `critic_norm::V`: Stores critic gradient norms
- `avg_adv::V`: Stores average advantage values
- `mean::V`: Stores mean values of action distributions
- `std::V`: Stores standard deviation values of action distributions
- `δ::V`: Stores temporal difference values

# Details
A2CGAE combines the actor-critic architecture with generalized advantage estimation for more
stable policy gradient updates. It includes weights for balancing the actor, critic, and entropy
loss components, and supports gradient norm clipping.
"""
@kwdef mutable struct A2CGAE{A,C,D,V} <: AbstractPolicy
    actor::A
    critic::C
    γ::Float32 = 0.99f0
    λ::Float32 = 0.9f0
    dist::D = Normal
    max_grad_norm::Union{Nothing,Float32} = nothing
    actor_loss_weight::Float32 = 1f0
    entropy_loss_weight::Float32 = 0.5f0
    batch_size::Int64 = 24
    epochs::Int64 = 1
    rng::AbstractRNG = Random.default_rng()
    online_stats::Normaliser = OnlineNorm(14) # Keeps track of the current mean and variance of the state information.
    
    # for logging
    mem_safe::Bool = false
    actor_loss::V = sizehint!(Float32[], 5000)
    critic_loss::V = sizehint!(Float32[], 5000)
    entropy_loss::V = sizehint!(Float32[], 5000)
    loss::V = sizehint!(Float32[], 5000)
    raw_loss::V = sizehint!(Float32[], 5000)
    actor_norm::V = sizehint!(Float32[], 5000)
    critic_norm::V = sizehint!(Float32[], 5000)
    avg_adv::V = sizehint!(Float32[], 5000)
    mean::V = sizehint!(Float32[], 350000)
    std::V = sizehint!(Float32[], 350000)
    δ::V = sizehint!(Float32[], 5000)

end
#########################################################################################
# PPO
"""
    PPO{A,C,D,V} <: AbstractPolicy

Proximal Policy Optimization implementation.

# Fields
- `actor::A`: Neural network that outputs action distribution parameters
- `old_actor::A`: Copy of actor network for computing importance ratios
- `critic::C`: Neural network that estimates state values
- `γ::Float32 = 0.99f0`: Discount factor for future rewards
- `λ::Float32 = 0.9f0`: GAE lambda parameter for advantage estimation
- `clip_coef::Float32 = 0.25f0`: Clipping coefficient for PPO's objective
- `dist::D = Normal`: Probability distribution for stochastic policy (defaults to Normal)
- `max_grad_norm::Union{Nothing,Float32} = nothing`: Maximum gradient norm for clipping
- `actor_loss_weight::Float32 = 1f0`: Weight for the actor loss component
- `entropy_loss_weight::Float32 = 0.5f0`: Weight for the entropy loss component
- `batch_size::Int64 = 24`: Number of samples to process in each training batch
- `epochs::Int64 = 1`: Number of epochs to train on each batch of data
- `rng::AbstractRNG = Random.default_rng()`: Random number generator
- `online_stats::Normaliser = OnlineNorm(14)`: Tracks statistics for state normalization

## Logging fields
- `mem_safe::Bool = false`: Flag for memory efficient operation
- `actor_loss::V`: Stores actor loss values during training
- `critic_loss::V`: Stores critic loss values during training
- `entropy_loss::V`: Stores entropy loss values
- `loss::V`: Stores combined loss values
- `raw_loss::V`: Stores raw loss values before normalization
- `actor_norm::V`: Stores actor gradient norms
- `critic_norm::V`: Stores critic gradient norms
- `avg_adv::V`: Stores average advantage values
- `δ::V`: Stores temporal difference values
- `clip_fracs::V`: Stores clipping fractions during training
- `approx_kl::V`: Stores approximate KL divergence values
- `mean::V`: Stores mean values of action distributions
- `std::V`: Stores standard deviation values of action distributions

# Details
PPO is a policy gradient method that uses a clipped surrogate objective to constrain policy updates,
improving stability. This implementation includes GAE for advantage estimation and maintains an old
copy of the policy network for calculating importance sampling ratios.
"""
@kwdef mutable struct PPO{A,C,D,V} <: AbstractPolicy
    actor::A
    old_actor::A 
    critic::C
    γ::Float32 = 0.99f0
    λ::Float32 = 0.9f0
    clip_coef::Float32 = 0.25f0
    dist::D = Normal
    max_grad_norm::Union{Nothing,Float32} = nothing
    actor_loss_weight::Float32 = 1f0
    entropy_loss_weight::Float32 = 0.5f0
    batch_size::Int64 = 24
    epochs::Int64 = 1
    rng::AbstractRNG = Random.default_rng()
    online_stats::Normaliser = OnlineNorm(14) # Keeps track of the current mean and variance of the state information.
    
    # for logging
    mem_safe::Bool = false
    actor_loss::V = sizehint!(Float32[], 5000)
    critic_loss::V = sizehint!(Float32[], 5000)
    entropy_loss::V = sizehint!(Float32[], 5000)
    loss::V = sizehint!(Float32[], 5000)
    raw_loss::V = sizehint!(Float32[], 5000)
    actor_norm::V = sizehint!(Float32[], 5000)
    critic_norm::V = sizehint!(Float32[], 5000)
    avg_adv::V = sizehint!(Float32[], 5000)
    δ::V = sizehint!(Float32[], 5000)
    clip_fracs::V = sizehint!(Float32[], 5000)
    approx_kl::V = sizehint!(Float32[], 5000)
    mean::V = sizehint!(Float32[], 350000)
    std::V = sizehint!(Float32[], 350000)
    
end


"""
    reset_stats!(pol::AbstractPolicy)

Resets the normalization statistics for a policy.

# Arguments
- `pol::AbstractPolicy`: The policy whose statistics should be reset

# Details
Attempts to reset the `online_stats` field of the policy, which is responsible for state normalization.
Catches and reports any errors that occur during the reset process.
"""
function reset_stats!(pol::AbstractPolicy)
    # Function implementation...
end
function reset_stats!(pol::AbstractPolicy)
    try 
        reset_stats!(pol.online_stats)
    catch e
        println("Could not reset the Normaliser object")
        println("The error is: $e")
    end
end

#########################################################################################
# Replay
"""
    Replay{V} <: AbstractPolicy

A replay policy that returns predefined actions.

# Fields
- `p_ev::V`: Vector of historical electric vehicle power values
- `p_bess::V`: Vector of historical battery energy storage system power values
- `p_hp_e::V`: Vector of historical heat pump electrical power values
- `ind::Int32`: Current index in the replay data
- `max_ind::Int32`: Maximum index (length of the shortest data vector)

# Details
This policy doesn't learn but instead replays pre-recorded actions, useful for imitation
learning or as a baseline. It maintains separate vectors for different components of the
energy management system.
"""
@kwdef mutable struct Replay{V} <: AbstractPolicy
    p_ev::V = Float32[]
    p_bess::V = Float32[]
    p_hp_e::V = Float32[]
    ind::Int32 = Int32(1)
    max_ind::Int32 = Int32(35040)
end

function Replay(p_ev::V, p_bess::V, p_hp_e::V) where V
    max_ind = minimum(length, (p_ev, p_bess, p_hp_e))
    Replay{V}(p_ev, p_bess, p_hp_e, 1, max_ind)
end

"""
    Replay32(p_ev::Vector, p_bess::Vector, p_hp_e::Vector)

Constructs a Replay policy with Float32 vectors from the provided data.

# Arguments
- `p_ev::Vector`: Vector of electric vehicle power values
- `p_bess::Vector`: Vector of battery energy storage system power values
- `p_hp_e::Vector`: Vector of heat pump electrical power values

# Returns
- A new `Replay{Vector{Float32}}` instance

# Details
Similar to the standard `Replay` constructor but ensures the data is stored as Float32.
"""
function Replay32(p_ev::Vector, p_bess::Vector, p_hp_e::Vector) 
    max_ind = minimum(length, (p_ev, p_bess, p_hp_e))
    Replay{Vector{Float32}}(p_ev, p_bess, p_hp_e, 1, max_ind)
end

"""
    Replay(data::Matrix{V}) where V

Constructs a Replay policy from a matrix of historical actions.

# Arguments
- `data::Matrix{V}`: Matrix where each row represents a different component:
  - Row 1: Electric vehicle power values
  - Row 2: Battery energy storage system power values
  - Row 3: Heat pump electrical power values

# Returns
- A new `Replay{Vector{V}}` instance

# Throws
- `ArgumentError`: If the input matrix doesn't have exactly 3 rows

# Details
Creates a replay policy by extracting component data from rows of the provided matrix.
Useful when historical data is stored in a consolidated matrix format.
"""
function Replay(data::Matrix{V}) where V
    if size(data, 1) != 3
        throw(ArgumentError("Input matrix must have exactly 3 rows"))
    end
    
    p_ev = data[1, :]
    p_bess = data[2, :]
    p_hp_e = data[3, :]
    
    max_ind = size(data, 2)
    Replay{Vector{V}}(p_ev, p_bess, p_hp_e, 1, max_ind)
end

###############################################################################################
# Imitation
"""
    Clone_Policy{L,S,A}

Policy for behavioral cloning (imitation learning).

# Fields
- `learner::L`: The policy being trained to imitate the expert
- `expert_states::S`: States from expert demonstrations
- `expert_actions::A`: Actions from expert demonstrations
- `expert_rewards::Vector{Float32}`: Rewards from expert demonstrations
- `batch_size::Int64 = 96`: Number of samples to process in each training batch
- `entropy_loss_weight::Float32 = 1f-3`: Weight for the entropy loss component
- `max_grad_norm::Float32 = 1f0`: Maximum gradient norm for clipping

## Logging fields
- `actor_loss::Vector{Float32}`: Stores actor loss values during training
- `critic_loss::Vector{Float32}`: Stores critic loss values during training
- `entropy_loss::Vector{Float32}`: Stores entropy loss values
- `mean::Vector{Float32}`: Stores mean values of action distributions
- `std::Vector{Float32}`: Stores standard deviation values of action distributions
"""
@kwdef mutable struct Clone_Policy{L,S,A}
    learner::L
    expert_states::S
    expert_actions::A
    expert_rewards::Vector{Float32}
    batch_size::Int64 = 96  # It should match the batch size the agent will use in the RL Env. 
    entropy_loss_weight::Float32 = 1f-3
    max_grad_norm::Float32 = 1f0

    # Bookkeeping
    actor_loss::Vector{Float32} = sizehint!(Float32[], 5000)
    critic_loss::Vector{Float32} = sizehint!(Float32[], 5000)
    entropy_loss::Vector{Float32} = sizehint!(Float32[], 5000)
    mean::Vector{Float32} = sizehint!(Float32[], 350000)
    std::Vector{Float32} = sizehint!(Float32[], 350000)
end



@info "Policy Types Added"