


#########################################################################################
# Custom NN

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

# @kwdef mutable struct Tracker{V}
#     avg_adv::V = sizehint!(Float32[], 5000) 
#     avg_loss::V = sizehint!(Float32[], 5000) 
#     raw_avg_loss::V = sizehint!(Float32[], 5000)
#     mean::V = sizehint!(Float32[], 350000) 
#     std::V = sizehint!(Float32[], 350000) 
#     δ::V = sizehint!(Float32[], 5000) 
# end

# @kwdef mutable struct Tracker_NoHint{V}
#     avg_adv::V = Float32[] 
#     avg_loss::V = Float32[] 
#     raw_avg_loss::V = Float32[]
#     mean::V = Float32[]
#     std::V = Float32[] 
# end

# @kwdef mutable struct myVPG{V<:VPG, T} <: AbstractPolicy
#     vpg::V
#     debug::T = Tracker()
#     std_min::Float32 = 1f-3
#     epochs::Int64 = 1
# end

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

function Replay32(p_ev::Vector, p_bess::Vector, p_hp_e::Vector) 
    max_ind = minimum(length, (p_ev, p_bess, p_hp_e))
    Replay{Vector{Float32}}(p_ev, p_bess, p_hp_e, 1, max_ind)
end

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