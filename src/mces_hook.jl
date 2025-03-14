# Defining Hook objects
"""
    MCES_Hook <: AbstractHook

A hook for recording rewards during episodes in the MCES environment. 
Designed for debugging purposes, it keeps track of individual rewards and their components.

# Fields
- `episode_rewards::Vector{Real}`: Vector storing the rewards obtained in each episode.
- `reward::Real`: Accumulated reward for the current episode.
- `reward_dissect::Dict{String, Vector{Real}}`: Dictionary to dissect and store different 
  components of rewards (costs with inverted signs).
- `energy::Dict{String, Vector{Real}}`: Dictionary to store energy-related metrics during episodes.
- `debug::Dict{String, Vector{Real}}`: Dictionary to store debugging information.
"""
@kwdef mutable struct MCES_Hook{V,D} <: AbstractHook
    episode_rewards::V = sizehint!(Float32[], 5000)
    reward::Float32 = 0.0f0
    reward_dissect::D = Dict{String, Vector{Float32}}()
    energy::D = Dict{String, Vector{Float32}}()
    debug::D = Dict{String,Vector{Float32}}()
end

# Empty Hook 
"""
    VoidHook <: AbstractHook

A minimal hook implementation that doesn't record any information.
Used when no monitoring is needed to reduce memory overhead.
"""
struct VoidHook <: AbstractHook end

"""
    MCES_Moderate_Hook{V, D} <: AbstractHook

A memory-efficient version of MCES_Hook that uses Float16 to reduce memory usage. Focused exclusively on tracking rewards. 
Suitable for longer training runs (or multithreading) where memory efficiency is more important than precision.

# Fields
- `episode_rewards::V`: Vector storing the rewards obtained in each episode.
- `reward::Float16`: Accumulated reward for the current episode.
- `reward_dissect::D`: Dictionary to dissect and store different components of rewards.
"""
@kwdef mutable struct MCES_Moderate_Hook{V, D} <: AbstractHook # Memory efficient hook.
    episode_rewards::V = sizehint!(Float16[], 3500)
    reward::Float16 = Float16(0.0)
    reward_dissect::D = Dict{String, Vector{Float16}}()
end

####################################################
# Hook Specific Functions
"""
    create_reward_dissect(; mem_safe = false)

Creates a dictionary to store different components of rewards for analysis.

# Arguments
- `mem_safe::Bool`: If true, uses Float16 for memory efficiency; otherwise uses Float32 for precision.

# Returns
- `Dict{String, Vector{Float{16|32}}}`: Dictionary with pre-allocated vectors for different reward components.

# Components
The dictionary contains the following keys:
- "Initial Projection": Initial projection-related rewards
- "Op. Projection": Operational projection-related rewards
- "Grid": Grid-related rewards
- "EV": Electric vehicle-related rewards
- "Degradation": Degradation-related rewards
- "Clamped Reward": Rewards after clamping operations
- "Real Reward": Actual rewards received
"""
function create_reward_dissect(;mem_safe = false)
    if mem_safe
        return Dict{String, Vector{Float16}}(
            key => sizehint!(Float16[], 200000) for key in [
                "Initial Projection", "Op. Projection", "Grid", "EV", "Degradation", "Clamped Reward", "Real Reward"]
            )
    else
        return Dict{String, Vector{Float32}}(
            key => sizehint!(Float32[], 250000) for key in [
                "Initial Projection", "Op. Projection", "Grid", "EV", "Degradation", "Clamped Reward", "Real Reward"]
            )
    end
end

"""
    create_energy_dict()

Creates a dictionary to store energy-related metrics during simulation.

# Returns
- `Dict{String, Vector{Float32}}`: Dictionary with pre-allocated vectors for different energy components.

# Components
The dictionary contains keys for various energy-related measurements including:
- Load-related: "load_e", "load_th"
- Generation: "pv"
- Storage: "st", "p_tess", "soc_tess", "p_bess", "p_bess_raw", "soc_bess"
- Electric vehicle: "p_ev", "p_ev_raw", "γ_ev", "soc_ev"
- Heat pump: "p_hp_e", "p_hp_e_raw", "p_hp_th"
- Grid: "grid"
- Time: "t"
- Electrical: "i_bess", "v_bess", "i_ev", "v_ev"
- Constraint violations: "ξp_ev", "ξp_bess", "ξp_hp_e", "ξsoc_ev", 
  "ξsoc_bess", "ξsoc_tess", "ξp_tess", "ξp_grid"
"""
function create_energy_dict()
    Dict{String, Vector{Float32}}(
        key => sizehint!(Float32[], 250000) for key in [
            "load_e", "load_th", "pv", "st", "grid", "p_tess", "soc_tess", "p_hp_e", 
            "p_hp_e_raw", "p_hp_th", "p_bess", "p_bess_raw", "soc_bess", "p_ev", 
            "p_ev_raw", "γ_ev", "soc_ev", "t",
            "i_bess", "v_bess", "i_ev", "v_ev",
            # New logging elements for constraint violations
            "ξp_ev", "ξp_bess", "ξp_hp_e",
            "ξsoc_ev", "ξsoc_bess", "ξsoc_tess", "ξp_tess", "ξp_grid"
        ]
    )
end

"""
    create_debug_vpg_dict()

Creates a dictionary to store debugging information for the Vanilla Policy Gradient (VPG) algorithm.

# Returns
- `Dict{String, Vector{Float32}}`: Dictionary with pre-allocated vectors for different VPG metrics.

# Components
The dictionary contains keys for various VPG-specific metrics including:
- Training statistics: "avg_adv", "avg_loss", "raw_avg_loss", "actor_loss", "critic_loss", "δ"
- Gradient information: "actor_norm", "critic_norm"
- Distribution metrics: "mean", "std" (with larger pre-allocated buffers)
"""
function create_debug_vpg_dict()
    d = Dict{String, Vector{Float32}}(
        key => sizehint!(Float32[], 16000) for key in [
            "avg_adv", "avg_loss", "raw_avg_loss", "actor_loss", "critic_loss", 
            "actor_norm", "critic_norm", "δ"]
    )
    d["mean"] = sizehint!(Float32[], 2500000)
    d["std"] = sizehint!(Float32[], 2500000)
    d
end

"""
    create_debug_a2c_dict()

Creates a dictionary to store debugging information for the Advantage Actor-Critic (A2C) algorithm.

# Returns
- `Dict{String, Vector{Float32}}`: Dictionary with pre-allocated vectors for different A2C metrics.

# Components
The dictionary contains keys for various A2C-specific metrics including:
- Training statistics: "avg_adv", "avg_loss", "raw_avg_loss", "actor_loss", "critic_loss", "entropy_loss", "δ"
- Gradient information: "actor_norm", "critic_norm"
- Distribution metrics: "mean", "std" (with larger pre-allocated buffers)
"""
function create_debug_a2c_dict()
    d = Dict{String, Vector{Float32}}(
        key => sizehint!(Float32[], 16000) for key in [
            "avg_adv", "avg_loss", "raw_avg_loss", "actor_loss", "critic_loss", 
            "entropy_loss", "actor_norm", "critic_norm", "δ"]
    )
    d["mean"] = sizehint!(Float32[], 2500000)
    d["std"] = sizehint!(Float32[], 2500000)
    d
end

"""
    create_debug_ppo_dict()

Creates a dictionary to store debugging information for the Proximal Policy Optimization (PPO) algorithm.

# Returns
- `Dict{String, Vector{Float32}}`: Dictionary with pre-allocated vectors for different PPO metrics.

# Components
The dictionary contains keys for various PPO-specific metrics including:
- Training statistics: "avg_adv", "avg_loss", "raw_avg_loss", "actor_loss", "critic_loss", "entropy_loss", "δ"
- Gradient information: "actor_norm", "critic_norm"
- PPO-specific metrics: "clip_fracs" (clipping fraction), "approx_kl" (approximate KL divergence)
- Distribution metrics: "mean", "std" (with larger pre-allocated buffers)
"""
function create_debug_ppo_dict()
    d = Dict{String, Vector{Float32}}(
        key => sizehint!(Float32[], 16000) for key in [
            "avg_adv", "avg_loss", "raw_avg_loss", "actor_loss", "critic_loss", 
            "entropy_loss", "actor_norm", "critic_norm", "δ", "clip_fracs", "approx_kl"]
    )
    d["mean"] = sizehint!(Float32[], 2500000)
    d["std"] = sizehint!(Float32[], 2500000)
    d
end
#############################################
# MCES_Hook functionality
# Each function represents the actions performed with the Hook ina particular stage of the run (e.g. PreActStage, PostEpisodeStage...)

Base.getindex(h::MCES_Hook) = h.episode_rewards

function (h::MCES_Hook)(::PreExperimentStage, agent, env)
    if isempty(h.reward_dissect)
        h.reward_dissect = create_reward_dissect()
    end

    if isempty(h.energy)
        h.energy = create_energy_dict()
    end

    nothing
end

function (h::MCES_Hook)(::PostActStage, agent, env) 
    h.reward += reward(env)
    
    # Dissection of Reward 
    rd = h.reward_dissect 
    push!(rd["Initial Projection"], -env.init_projection)   # Unweighted
    push!(rd["Op. Projection"], -env.op_projection)   # Unweighted
    push!(rd["Grid"], -env.cost_grid)               # Unweighted
    push!(rd["EV"], -env.cost_ev)                   # Unweighted
    push!(rd["Degradation"], -env.cost_degradation) # Unweighted
    push!(rd["Clamped Reward"], reward(env)) # Clamped and Weighted Reward
    push!(rd["Real Reward"], -(env.init_projection + env.op_projection + env.cost_grid + env.cost_ev + env.cost_degradation)) #UnWeighted reward


    # Power and SoCs for Plotting
    e = h.energy
    push!(e["load_e"], env.load_e)
    push!(e["load_th"], env.load_th)
    push!(e["pv"], env.pv)
    push!(e["st"], env.st)
    push!(e["grid"], env.grid)
    push!(e["p_tess"], env.tess.p)
    push!(e["soc_tess"], env.tess.soc)
    push!(e["p_hp_e"], env.hp.e)
    push!(e["p_hp_e_raw"], env.p_hp_e_raw)
    push!(e["p_hp_th"], env.hp.th)
    push!(e["p_bess"], env.bess.p)
    push!(e["p_bess_raw"], env.p_bess_raw)
    push!(e["soc_bess"], env.bess.soc)
    push!(e["p_ev"], env.ev.bat.p)
    push!(e["p_ev_raw"], env.p_ev_raw)
    push!(e["γ_ev"], env.ev.γ)
    push!(e["soc_ev"], env.ev.bat.soc)
    push!(e["t"], isempty(e["t"]) ? 1f0 : e["t"][end]+1)
    push!(e["i_bess"], env.bess.i)
    push!(e["v_bess"], env.bess.v)
    push!(e["i_ev"], env.ev.bat.i)
    push!(e["v_ev"], env.ev.bat.v)

    # New logging elements for constraint violations
    push!(e["ξp_ev"], env.ξp_ev)
    push!(e["ξp_bess"], env.ξp_bess)
    push!(e["ξp_hp_e"], env.ξp_hp_e)
    push!(e["ξsoc_ev"], env.ξsoc_ev)
    push!(e["ξsoc_bess"], env.ξsoc_bess)
    push!(e["ξsoc_tess"], env.ξsoc_tess)
    push!(e["ξp_tess"], env.ξp_tess)
    push!(e["ξp_grid"], env.ξp_grid)
    
    nothing
end

function (h::MCES_Hook)(::PostEpisodeStage, agent, env)
    push!(h.episode_rewards, h.reward)
    h.reward = 0.0f0
    nothing
end

#####################################################################################################
# Hook extra functionality for myVPG
function (h::MCES_Hook)(::PreExperimentStage, agent::Agent{<:myVPG}, env)
    if isempty(h.reward_dissect)
        h.reward_dissect = create_reward_dissect()
    end

    if isempty(h.energy)
        h.energy = create_energy_dict()
    end

    if isempty(h.debug)
        h.debug = create_debug_vpg_dict()
    end
    nothing
end

function (h::MCES_Hook)(::PostExperimentStage, agent::Agent{<:myVPG}, env)
    p = agent.policy
    d = h.debug
    push!(d["avg_adv"], p.avg_adv...)
    push!(d["raw_avg_loss"], p.raw_loss...)
    push!(d["avg_loss"], p.actor_loss...)
    push!(d["mean"], p.mean...)
    push!(d["std"], p.std...)
    push!(d["δ"], p.δ...)

    empty!(p.avg_adv)
    empty!(p.raw_loss)
    empty!(p.mean)
    empty!(p.std)
    empty!(p.δ)

    push!(d["actor_loss"], p.actor_loss...)
    push!(d["critic_loss"], p.critic_loss...)
    push!(d["actor_norm"], p.actor_norm...)
    push!(d["critic_norm"], p.critic_norm...)
    
    empty!(p.actor_loss)
    empty!(p.critic_loss)
    empty!(p.actor_norm)
    empty!(p.critic_norm)

    nothing
end

#####################################################################################################
# Hook extra functionality for A2CGAE

function (h::MCES_Hook)(::PreExperimentStage, agent::Agent{<:A2CGAE}, env)
    if isempty(h.reward_dissect)
        h.reward_dissect = create_reward_dissect()
    end

    if isempty(h.energy)
        h.energy = create_energy_dict()
    end

    if isempty(h.debug)
        h.debug = create_debug_a2c_dict()
    end
    nothing
end

function (h::MCES_Hook)(::PostExperimentStage, agent::Agent{<:A2CGAE}, env)

    p = agent.policy
    d = h.debug
    push!(d["avg_adv"], p.avg_adv...)
    push!(d["avg_loss"], p.loss...)
    push!(d["raw_avg_loss"], p.raw_loss...)
    push!(d["mean"], p.mean...)
    push!(d["std"], p.std...)
    push!(d["δ"], p.δ...)

    empty!(p.avg_adv)
    empty!(p.loss)
    empty!(p.raw_loss)
    empty!(p.mean)
    empty!(p.std)
    empty!(p.δ)

    # Push additional fields exclusive to A2CGAE
    push!(d["actor_loss"], p.actor_loss...)
    push!(d["critic_loss"], p.critic_loss...)
    push!(d["entropy_loss"], p.entropy_loss...)
    push!(d["actor_norm"], p.actor_norm...)
    push!(d["critic_norm"], p.critic_norm...)

    # Empty all pushed fields
    empty!(p.actor_loss)
    empty!(p.critic_loss)
    empty!(p.entropy_loss)
    empty!(p.actor_norm)
    empty!(p.critic_norm)

    nothing
end


#####################################################################################################
# Hook extra functionality for A2CGAE

function (h::MCES_Hook)(::PreExperimentStage, agent::Agent{<:PPO}, env)
    if isempty(h.reward_dissect)
        h.reward_dissect = create_reward_dissect()
    end

    if isempty(h.energy)
        h.energy = create_energy_dict()
    end

    if isempty(h.debug)
        h.debug = create_debug_ppo_dict()
    end
    nothing
end

function (h::MCES_Hook)(::PostExperimentStage, agent::Agent{<:PPO}, env)
    p = agent.policy
    d = h.debug
    push!(d["avg_adv"], p.avg_adv...)
    push!(d["avg_loss"], p.loss...)
    push!(d["raw_avg_loss"], p.raw_loss...)
    push!(d["mean"], p.mean...)
    push!(d["std"], p.std...)
    push!(d["δ"], p.δ...)

    empty!(p.avg_adv)
    empty!(p.loss)
    empty!(p.raw_loss)
    empty!(p.mean)
    empty!(p.std)
    empty!(p.δ)

    push!(d["actor_loss"], p.actor_loss...)
    push!(d["critic_loss"], p.critic_loss...)
    push!(d["entropy_loss"], p.entropy_loss...)
    push!(d["actor_norm"], p.actor_norm...)
    push!(d["critic_norm"], p.critic_norm...)
    push!(d["clip_fracs"], p.clip_fracs...)
    push!(d["approx_kl"], p.approx_kl...)
    
    empty!(p.actor_loss)
    empty!(p.critic_loss)
    empty!(p.entropy_loss)
    empty!(p.actor_norm)
    empty!(p.critic_norm)
    empty!(p.clip_fracs)
    empty!(p.approx_kl)

    nothing
end

#####################################################
# VoidHook and Moderate Hook functionality

Base.getindex(h::MCES_Moderate_Hook) = h.episode_rewards

function (h::MCES_Moderate_Hook)(::PreExperimentStage, agent, env)
    if isempty(h.reward_dissect)
        h.reward_dissect = Dict{String, Vector{Float16}}(
            key => sizehint!(Float16[], 200000) for key in [
                 "Clamped Reward", "Real Reward"]
            )
    end
    nothing
end

@inline function (h::MCES_Moderate_Hook)(::PostActStage, agent, env)
    h.reward += Float16(reward(env))

    rd = h.reward_dissect 
    push!(rd["Clamped Reward"], reward(env)) # Clamped and Weighted Reward
    push!(rd["Real Reward"], -(env.init_projection + env.op_projection + env.cost_grid + env.cost_ev + env.cost_degradation)) #UnWeighted reward

end

function (h::MCES_Moderate_Hook)(::PostEpisodeStage, agent, env)
    push!(h.episode_rewards, h.reward)
    h.reward = Float16(0.0)
end

function (h::Union{MCES_Moderate_Hook, VoidHook})(::PostExperimentStage, agent::Agent{<:myVPG}, env)
    p = agent.policy
    empty!(p.avg_adv)
    empty!(p.raw_loss)
    empty!(p.actor_loss)
    empty!(p.critic_loss)
    empty!(p.actor_norm)
    empty!(p.critic_norm)
    empty!(p.mean)
    empty!(p.std)
    empty!(p.δ)
    nothing
end

function (h::Union{MCES_Moderate_Hook, VoidHook})(::PostExperimentStage, agent::Agent{<:A2CGAE}, env)
    p = agent.policy
    empty!(p.avg_adv)
    empty!(p.loss)
    empty!(p.raw_loss)
    empty!(p.mean)
    empty!(p.std)
    empty!(p.δ)
    empty!(p.actor_loss)
    empty!(p.critic_loss)
    empty!(p.entropy_loss)
    empty!(p.actor_norm)
    empty!(p.critic_norm)
    nothing
end

function (h::Union{MCES_Moderate_Hook, VoidHook})(::PostExperimentStage, agent::Agent{<:PPO}, env)
    p = agent.policy
    empty!(p.avg_adv)
    empty!(p.loss)
    empty!(p.raw_loss)
    empty!(p.mean)
    empty!(p.std)
    empty!(p.δ)
    empty!(p.actor_loss)
    empty!(p.critic_loss)
    empty!(p.entropy_loss)
    empty!(p.actor_norm)
    empty!(p.critic_norm)
    empty!(p.clip_fracs)
    empty!(p.approx_kl)
    nothing
end


@info "Ready to hook"

