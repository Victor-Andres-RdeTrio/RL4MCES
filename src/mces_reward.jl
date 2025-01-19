# Rewards. The cost has to be inverted to make negative rewards.
@inline function RLBase.reward(env::MCES_Env)
    p = env.params
    return clamp(- p.w_init_proj * env.init_projection 
                 - p.w_op_proj * env.op_projection 
                 - p.w_soc * env.cost_ev 
                 - p.w_grid * env.cost_grid, 
                 - 50f0, 50f0)
end 

function reward_update!(env::MCES_Env)

    push!(env.grid_buffer, [env.grid, env.λ_buy, env.λ_sell])

    if env.reward_shape == 1
        env.init_projection = init_proj_cost_v1(env)
        env.op_projection = op_proj_cost_v1(env)
        env.cost_ev = ev_cost_abs(env) 
        env.cost_grid = grid_cost(env) + (env.cum_cost_grid * env.cost_grid)

    elseif env.reward_shape == 2
        env.init_projection = init_proj_cost_v1(env)
        env.op_projection = op_proj_cost_v1(env)
        env.cost_ev = ev_cost_abs(env)
        env.cost_grid = grid_cost_per_x_episodes(env, Int32(2))
        
    elseif env.reward_shape == 3
        env.init_projection = init_proj_cost_v1(env)
        env.op_projection = op_proj_cost_v1(env)
        env.cost_ev = ev_cost_abs(env)
        env.cost_grid = clamp(arbitrage_cost_2days(env), -3f0, 3f0)

    elseif env.reward_shape == 4
        env.init_projection = init_proj_cost_v1(env)
        env.op_projection = op_proj_cost_v1(env)
        env.cost_ev = ev_cost_sigmoid(env)
        env.cost_grid = grid_cost_per_x_episodes(env, Int32(2))

    elseif env.reward_shape == 5
        env.init_projection = init_proj_cost_v1(env)
        env.op_projection = op_proj_cost_v1(env)
        env.cost_ev = ev_cost_sigmoid(env)
        env.cost_grid = clamp(arbitrage_cost_2days(env), -3f0, 3f0)

    elseif env.reward_shape == 6
        env.init_projection = init_proj_cost_v1(env)
        env.op_projection = op_proj_cost_v1(env) + all_border_penalties(env)
        env.cost_ev = ev_cost_abs(env)
        env.cost_grid = grid_cost_per_x_episodes(env, Int32(2))

    elseif env.reward_shape == 7
        env.init_projection = init_proj_cost_v1(env)
        env.op_projection = op_proj_cost_v1(env) + all_border_penalties(env)
        env.cost_ev = ev_cost_abs(env)
        env.cost_grid = clamp(arbitrage_cost_2days(env), -3f0, 3f0)

    elseif env.reward_shape == 8
        env.init_projection = init_proj_cost_v1(env)
        env.op_projection = op_proj_cost_v1(env) + all_border_penalties(env)
        env.cost_ev = ev_cost_sigmoid(env)
        env.cost_grid = grid_cost_per_x_episodes(env, Int32(2))

    elseif env.reward_shape == 9
        env.init_projection = init_proj_cost_v1(env)
        env.op_projection = op_proj_cost_v1(env) + all_border_penalties(env)
        env.cost_ev = ev_cost_sigmoid(env)
        env.cost_grid = clamp(arbitrage_cost_2days(env), -3f0, 3f0)

    else
        error("Reward Shape value of the Environment Object not recognized")
    end

    # Not implemented
    # env.cost_degradation = 0f0

    nothing
end


####################################################################################################
# Penalty calculation
# Keep in mind that costs will be positive but turned negative when fed to the agent as rewards. 

# Initial Projection
@inline function init_proj_cost_v1(env::MCES_Env)
    norm((env.ξp_ev, env.ξp_bess, env.ξp_hp_e))
end

@inline function init_proj_cost_v2(env::MCES_Env)  # Higher incentives for reducing the cost to 0.
    p = (env.ξp_ev, env.ξp_bess, env.ξp_hp_e)
    return sum(sqrt.(p) .+ p)
end


# Operational Porjection
@inline function op_proj_cost_v1(env::MCES_Env)
    (1f0/(4f0 + env.ev.in_EMS)) * (   
        env.params.w_ξsoc_bess * env.ξsoc_bess + 
        env.params.w_ξsoc_ev   * env.ξsoc_ev + 
        env.params.w_ξsoc_tess * env.ξsoc_tess + 
        env.params.w_ξp_grid   * env.ξp_grid + 
        env.params.w_ξp_tess   * env.ξp_tess
    )
end

@inline function op_proj_cost_v2(
    env::MCES_Env,
    weights::NamedTuple{(:soc_bess, :soc_ev, :soc_tess, :p_tess, :p_grid), NTuple{5, Float32}} = (soc_bess = 0.2f0, soc_ev = 0.3f0, soc_tess = 0.2f0, p_tess = 0.15f0, p_grid = 0.15f0),
    β::NamedTuple{(:soc_bess, :soc_ev, :soc_tess, :p_tess, :p_grid), NTuple{5, Float32}} = (soc_bess = 600f0, soc_ev = 150f0, soc_tess = 1800f0, p_tess = 150f0, p_grid = 150f0)
)
    cost_soc_bess = 1f0 - exp(-β.soc_bess * env.ξsoc_bess)
    cost_soc_ev = 1f0 - exp(-β.soc_ev * env.ξsoc_ev)
    cost_soc_tess = 1f0 - exp(-β.soc_tess * env.ξsoc_tess)
    cost_p_tess = 1f0 - exp(-β.p_tess * env.ξp_tess)
    cost_p_grid = 1f0 - exp(-β.p_grid * env.ξp_grid)

    M_proj = (
        weights.soc_bess * cost_soc_bess +
        weights.soc_ev * cost_soc_ev +
        weights.soc_tess * cost_soc_tess +
        weights.p_tess * cost_p_tess +
        weights.p_grid * cost_p_grid
    )

    return M_proj
end

@inline function op_proj_cost_v3(env::MCES_Env) # v1 with no penalty for socev violation if EV not in house. 
    (1f0/(4f0 + env.ev.in_EMS)) * (   
        env.params.w_ξsoc_bess * env.ξsoc_bess + 
        env.params.w_ξsoc_ev   * env.ξsoc_ev * env.ev.γ + 
        env.params.w_ξsoc_tess * env.ξsoc_tess + 
        env.params.w_ξp_grid   * env.ξp_grid + 
        env.params.w_ξp_tess   * env.ξp_tess
    )
end

function all_border_penalties(env)
    return (
        border_penalty_bess(env, 0.35f0) + # max cost = 0.35  
        border_penalty_tess(env, 0.25f0) + # max cost = 0.25
        border_penalty_socdep(env, 1f0) + # max cost = 1    
        border_penalty_ptess(env, 0.25f0) + # max cost = 0.25
        border_penalty_pgrid(env, 1.25f0) # max cost = 1.25
    )
end


# EV
"""
    ev_cost_abs(env::MCES_Env)

Calculates the cost associated with the EV's SoC being different from the desired SoC at departure.

If the EV has departed, the cost is the absolute difference between the current SoC and the desired SoC.
If the EV has not departed, the cost is 0.0.
If the EV is not available as part of the EMS, then it is always 0.

# Arguments:
    env::MCES_Env: The environment object containing the EV and its current state.

# Returns:
    Float32: The value of the cost.
"""
@inline function ev_cost_abs(env::MCES_Env)
    (env.ev.departs && env.ev.in_EMS) ? abs(env.ev.bat.soc - env.ev.soc_dep) : 0f0 
end

@inline function ev_cost_sigmoid(env::MCES_Env; slope::Float32 = 15f0, mid::Float32 = 0.1f0, max_cost::Float32 = 0.65f0)
    ev = env.ev
    return (ev.departs && ev.in_EMS) ? 
    1f0 - ev_metric(abs(ev.bat.soc - ev.soc_dep), slope, mid, max_cost) : 0f0 
end

@inline function border_penalty_ev(env::MCES_Env, penalty = 0.5f0)
    bat = env.ev.bat
    return env.ev.γ * env.ev.in_EMS * (
        border_penalty(bat.soc, 0.95f0 * bat.soc_min, 0.5f0 * bat.soc_max, penalty) +
        border_penalty(bat.soc, 1.05f0 * bat.soc_max, 0.97f0 * bat.soc_max, penalty) 
    )
end

@inline function border_penalty_socdep(env::MCES_Env, penalty = 0.5f0)
    soc_dep = env.ev.soc_dep
    bat = env.ev.bat
    return env.ev.γ * env.ev.in_EMS * (
        border_penalty(bat.soc, 0.95f0 * bat.soc_min, 1.03f0 * soc_dep, penalty) +
        border_penalty(bat.soc, 1.05f0 * bat.soc_max, 0.97f0 * bat.soc_max, penalty) 
    )
end


# Grid
"""
    penalty_cost_grid(env)

Calculates the cost or benefit of exhanging energy with the grid. 

If positive its a cost, if negative its money earned. 
The signs will be inverted when turned into rewards. 

# Returns:
    Value of penalty. 
""" 
@inline function grid_cost(env::MCES_Env)
    λ = env.grid > 0 ? env.λ_buy : env.λ_sell   #[€/kWh]
    
    env.grid * λ * env.Δt/3.6f3  # [kW * (€/kWh) * h] 
end

@inline function grid_cost_per_x_episodes(env::MCES_Env, x::Integer = 1)::Float32
    env.t_ep != (env.episode_length - 1) && return 0f0

    # grid_buffer has 7 days of information
    max_x = round(Int32, 7 * env.daylength / env.episode_length, RoundDown)

    buf = env.grid_buffer ::CircularArrayBuffer{Float32, 2, Matrix{Float32}}
    data = buf[:, max(1, 1 + end - (env.episode_length * min(Int32(x),max_x))) : end] ::Matrix{Float32}
    
    p_grid = data[1, :]
    p_grid_pos = max.(0f0, p_grid) # Grid interactions where energy was bought [kW]
    p_grid_neg = min.(0f0, p_grid) # Grid interactions where energy was sold (all negative kW)
    
    λ_buy = data[2, :] ; λ_sell = data[3, :]
    C_grid = sum((λ_buy .* p_grid_pos) .+ (λ_sell .* p_grid_neg)) * env.Δt/3.6f3

    return C_grid / min(Int32(x),max_x)
end

"""
    arbitrage_cost(
        grid_power::Float32,
        price::Float32,
        abs_grid_power_mean::Float32,
        mean_buy_price::Float32,
        mean_sell_price::Float32
    )::Float32

Calculate the cost score for a single arbitrage action.

This function evaluates the cost of a single buy or sell action in an energy arbitrage scenario.
It considers the current grid power, price, and historical means to compute a scaled cost score,
which is the inverse of the performance score.

# Arguments
- `grid_power::Float32`: Current grid power. Positive for buying, negative for selling (in kW).
- `price::Float32`: Current price for the action (in currency units per kWh).
- `abs_grid_power_mean::Float32`: Mean of the absolute values of historical grid powers (in kW).
- `mean_buy_price::Float32`: Mean historical buying price (in currency units per kWh).
- `mean_sell_price::Float32`: Mean historical selling price (in currency units per kWh).

# Returns
- `Float32`: A cost score for the action. Positive scores indicate unfavorable actions.

# Details
- For buy actions (grid_power > 0), a positive score indicates buying at a price higher than the mean.
- For sell actions (grid_power < 0), a positive score indicates selling at a price lower than the mean.
- The score is scaled by the ratio of the current grid power to the mean absolute grid power.
- This function is marked as `@inline` for potential performance optimization.
- All inputs and the return value are `Float32` for potential performance and memory saving benefits.

# Example
```julia
score = arbitrage_cost(50.0f0, 0.1f0, 30.0f0, 0.12f0, 0.08f0)
```
This calculates the cost score for buying 50 kW at a price of 0.1, 
given a mean absolute grid power of 30 kW, mean buy price of 0.12, and mean sell price of 0.08.
"""
@inline function arbitrage_cost(
    grid_power::Float32,
    price::Float32,
    abs_grid_power_mean::Float32, 
    mean_buy_price::Float32,
    mean_sell_price::Float32
    )::Float32

    mean_buy_price = max(mean_buy_price, 1f-2) # To avoid sending wrong reward signals when the buffer is still not filled with data. 
    mean_sell_price = max(mean_sell_price, 1f-2)

    scaling_factor = abs(grid_power / max(abs_grid_power_mean, 1f-1))
    if grid_power > 0  # Buy action
        return ((price - mean_buy_price) / mean_buy_price) * scaling_factor
    else  # Sell action
        return ((mean_sell_price - price) / mean_sell_price) * scaling_factor
    end
end

@inline function arbitrage_cost_from_buffer(env::MCES_Env, data::AbstractMatrix{Float32})::Float32
    abs_grid_power_mean = mean(abs, data[1, :])
    mean_buy_price = mean(data[2, :])
    mean_sell_price = mean(data[3, :])

    current_price = env.grid > 0 ? env.λ_buy : env.λ_sell

    return arbitrage_cost(
        env.grid,
        current_price,
        abs_grid_power_mean,
        mean_buy_price,
        mean_sell_price
    )
end

@inline function arbitrage_cost_daily(env::MCES_Env)::Float32
    buf = env.grid_buffer ::CircularArrayBuffer{Float32, 2, Matrix{Float32}}
    data = buf[:, max(1, end-env.daylength+1) : end] ::Matrix{Float32}
    return arbitrage_cost_from_buffer(env, data)
end

@inline function arbitrage_cost_2days(env::MCES_Env)::Float32
    buf = env.grid_buffer ::CircularArrayBuffer{Float32, 2, Matrix{Float32}}
    data = buf[:, max(1, end-2*env.daylength+1) : end]
    return arbitrage_cost_from_buffer(env, data)
end

@inline function arbitrage_cost_3days(env::MCES_Env)::Float32
    buf = env.grid_buffer ::CircularArrayBuffer{Float32, 2, Matrix{Float32}}
    data = buf[:, max(1, end-3*env.daylength+1) : end]
    return arbitrage_cost_from_buffer(env, data)
end

@inline function arbitrage_cost_weekly(env::MCES_Env)::Float32
    buf = env.grid_buffer ::CircularArrayBuffer{Float32, 2, Matrix{Float32}}
    data = buf[:, max(1, end-7*env.daylength+1) : end]
    return arbitrage_cost_from_buffer(env, data)
end



@inline function border_penalty_pgrid(env::MCES_Env, penalty = 0.1f0)
    grid_max = env.pei.p_max
    border_penalty(abs(env.grid), 1.05f0*grid_max, 0.90f0 * grid_max, penalty)
end

###################################################################
# BESS penalties
@inline function border_penalty_bess(env::MCES_Env, penalty = 0.25f0)
    bess = env.bess
    return border_penalty(bess.soc, 0.95f0 * bess.soc_min, 1.25f0 * bess.soc_min, penalty) +
           border_penalty(bess.soc, 1.05f0 * bess.soc_max, 0.95f0 * bess.soc_max, penalty)
end

###################################################################
# TESS Penalties
@inline function border_penalty_tess(env::MCES_Env, penalty = 0.1f0)
    tess = env.tess
    return border_penalty(tess.soc, 0.95f0 * tess.soc_min, 1.25f0 * tess.soc_min, penalty) +
           border_penalty(tess.soc, 1.05f0 * tess.soc_max, 0.95f0 * tess.soc_max, penalty)
end

@inline function border_penalty_ptess(env::MCES_Env, penalty = 0.1f0)
    tess = env.tess
    border_penalty(abs(tess.p), 1.05f0*tess.p_max, 0.90f0 * tess.p_max, penalty)
end

############################################################
# Multipurpose penalties. 

@inline function border_penalty(x::Float32, bad_state::Float32, good_state::Float32, penalty::Float32)
    direction = sign(bad_state - good_state) # direction in which x becomes worse.
    
    normalized_x = (x - good_state) / abs(bad_state - good_state)
    # If input is within the good_state:bad_state range, then normalized x will share the sign of "direction". 
    # If sharing the sing, then direction * normalized_x is always positive. 
    !(0 < direction * normalized_x < 1) && return 0f0

    return penalty * (exp(4f0 * direction * normalized_x) - 1f0) / (exp(4f0) - 1f0)
end



##########################################################################
# Reward Discounting
function mydiscount_rewards(rewards, γ, V_last_state)
    gain = 0
    rewards[end] = V_last_state
    new_rewards = similar(rewards)
    for i in length(rewards):-1:1
        gain = rewards[i] + γ * gain 
        new_rewards[i] = gain
    end
    new_rewards
end

@inline function gae_advantage(rewards::Vector, values::Vector, γ, λ) 
    gae = 0
    advantages = similar(rewards)
    for i in length(rewards):-1:1
        delta = rewards[i] + γ * values[i+1] - values[i]
        gae = delta + γ * λ * gae
        advantages[i] = gae
    end
    advantages
end


##########################################################################
# Choosing a set of parameters for weighting the rewards. 

function what_parameters(type::Integer; ev_in_EMS::Bool = true)
    if type == 1
        p = MCES_Params(
            w_init_proj = 1,
            w_op_proj = 3,
            w_grid= 1,  
            w_soc = 3 * ev_in_EMS,  
            w_ξsoc_ev = 1/0.025 * ev_in_EMS,
            w_ξsoc_bess = 1/0.015,
            w_ξsoc_tess = 1/0.005,
            w_ξp_grid = 1/0.01, 
            w_ξp_tess = 1/5 
        )

    elseif type == 2 
        p = MCES_Params(
            w_init_proj = 2,
            w_op_proj = 4,
            w_grid= 1.5,  
            w_soc = 4 * ev_in_EMS,  
            w_ξsoc_ev = 1/0.025 * ev_in_EMS,
            w_ξsoc_bess = 1/0.015,
            w_ξsoc_tess = 1/0.005,
            w_ξp_grid = 1/0.01, 
            w_ξp_tess = 1/5 
        )

    elseif type == 3
        p = MCES_Params(
            w_init_proj = 0.75,
            w_op_proj = 5,
            w_grid = 0.75,
            w_soc = 2.5 * ev_in_EMS,
            w_ξsoc_ev = 1/0.025 * ev_in_EMS,
            w_ξsoc_bess = 1/0.015,
            w_ξsoc_tess = 1/0.005,
            w_ξp_grid = 1/0.01, 
            w_ξp_tess = 1/5 
        )
    elseif type == 4
        p = MCES_Params(
            w_init_proj = 0.8,
            w_op_proj = 4.2,
            w_grid = 1.1,
            w_soc = 3.2 * ev_in_EMS,
            w_ξsoc_ev = 1/0.025 * ev_in_EMS,
            w_ξsoc_bess = 1/0.015,
            w_ξsoc_tess = 1/0.005,
            w_ξp_grid = 1/0.01, 
            w_ξp_tess = 1/5 
        )
    elseif type == 5
        p = MCES_Params(
            w_init_proj = 1.1,
            w_op_proj = 3.8,
            w_grid = 0.95,
            w_soc = 2.6 * ev_in_EMS,
            w_ξsoc_ev = 32 * ev_in_EMS,
            w_ξsoc_bess = 1/0.03,
            w_ξsoc_tess = 1/0.0065,
            w_ξp_grid = 1/0.0012,
            w_ξp_tess = 1/4.8
        )
    elseif type == 6
        p = MCES_Params(
            w_init_proj = 0.9,
            w_op_proj = 2.0,
            w_grid = 1.15,
            w_soc = 1.5 * ev_in_EMS,
            w_ξsoc_ev = 0.8 * ev_in_EMS,
            w_ξsoc_bess = 1/0.02,
            w_ξsoc_tess = 1/0.005,
            w_ξp_grid = 1/0.01,
            w_ξp_tess = 1/5.2
        )
    elseif type == 7
        p = MCES_Params(
            w_init_proj = 2,
            w_op_proj = 4,
            w_grid = 0.75,
            w_soc = 1.6 * ev_in_EMS,
            w_ξsoc_ev = 50 * ev_in_EMS,
            w_ξsoc_bess = 1/0.03,
            w_ξsoc_tess = 1/0.0045,
            w_ξp_grid = 1/0.002,
            w_ξp_tess = 1/3.8
        )
    else
        error("MCES Params you were looking for was not found.")
    end
end

##########################################
# Computing average reward
"""
    compute_reward(hook::AbstractHook, total_reward::Bool = false, num_steps::Int = 10000, rand_w::Float32 = 0.5f0) -> Float32

Compute the reward as the weighted average of the mean of the last `num_steps` rewards 
and the mean of another `num_steps` randomly chosen rewards.

# Arguments
- `hook::MCES_Hook`: An AbstractHook containing the reward data.
- `total_reward::Bool`: A flag indicating whether to use "Real Reward" or "Clamped Reward". The "Real Reward" is unweighted and un clamped. 
- `num_steps::Int`: The number of steps to consider for the reward calculation. Default is `10000`.
- `rand_w::Float32`: The weight for the random steps mean in the weighted average. Default is `0.5f0`.

# Returns
- A `Float32` representing the computed reward.
"""
@inline function compute_reward(hook::AbstractHook, total_reward::Bool = false, num_steps::Integer = Int32(10000), rand_w::Float32 = 0.5f0)
    rwds = total_reward ? hook.reward_dissect["Real Reward"] : hook.reward_dissect["Clamped Reward"]
    rand_inds = randperm(length(rwds))
    random_steps_mean = mean(rwds[rand_inds])
    last_steps_mean = mean(rwds[end - min(num_steps, length(rwds) - 1):end])
    result = (1f0 - rand_w) * last_steps_mean + rand_w * random_steps_mean
    
    if isnan(result)
        @warn "Reward Computation resulted in NaN"
        return -4f0
    end

    return result
end

##########################################################################
# Measuring Performance

############
# Combined Metrics
function compute_comprehensive_performance(
    hook::MCES_Hook,
    exog::Exogenous_BatchCollection,
    daylength::Integer = Int32(96),
    ev_soc_dep::Float32 = 0.85f0,
    soc_weights::NamedTuple{(:soc_ev, :soc_bess, :soc_tess), NTuple{3, Float32}} = (soc_ev = 0.4f0, soc_bess = 0.4f0, soc_tess = 0.2f0),
    proj_weights::NamedTuple{(:soc_bess, :soc_ev, :soc_tess, :p_tess, :p_grid), NTuple{5, Float32}} = (soc_bess = 0.25f0, soc_ev = 0.3f0, soc_tess = 0.15f0, p_tess = 0.15f0, p_grid = 0.15f0),
    proj_β::NamedTuple{(:soc_bess, :soc_ev, :soc_tess, :p_tess, :p_grid), NTuple{5, Float32}} = (soc_bess = 100.0f0, soc_ev = 10.0f0, soc_tess = 100.0f0, p_tess = 100.0f0, p_grid = 100.0f0),
    metric_weights::NamedTuple{(:ev, :grid_mean, :C_grid, :soc_iqr, :soc_std, :operational_proj), NTuple{6, Float32}} = (ev = 0.6f0, grid_mean = 0.0f0, C_grid = 0.4f0, soc_iqr = 0f0, soc_std = 0f0, operational_proj = 0.5f0);
    io::IO = stdout
    )

    n_days = hook.energy["t"][end] / daylength
    
    # EV
    ev_perf, ev_avg, pDep = ev_performance(hook, daylength, ev_soc_dep, slope = 15f0, mid = 0.1f0, max_cost = 0.65f0)
    
    # Grid
    grid_perf_margin_orig, grid_perf_margin_sigmoid = grid_performance_margin(hook, exog)
    grid_perf_mean_orig, grid_perf_mean_sigmoid, total_grid_cost = grid_performance_mean(hook, exog)
    
    C_grid = total_grid_cost * Int32(24)/daylength # Adjusts Grid cost (assumed to have 1 hour timesteps) to the actual timestep length. 
    C_grid_per_day = C_grid / n_days
    C_grid_perf = C_grid_performance(C_grid_per_day, slope = 1f0, mid = 14.5f0)
    
    # SoC
    soc_perf_iqr, soc_perf_iqr_weighted = soc_performance_iqr(hook, soc_weights)
    soc_perf_std, soc_perf_std_weighted = soc_performance_std(hook, soc_weights)

    # Safety
    op_proj_metrics, op_proj_perf, _ = operational_proj_metric(hook, daylength, proj_weights, proj_β)
    op_proj_min = operational_proj_metric(
        hook, daylength, 
        (soc_bess = 1f0, soc_ev = 1f0, soc_tess = 1f0, p_tess = 1f0, p_grid = 1f0), 
        (soc_bess = 10f0, soc_ev = 10f0, soc_tess = 10f0, p_tess = 10f0, p_grid = 10f0)
        )[3]

    
    # Normalize weights for safety sum
    metric_weights_sum = metric_weights.ev + metric_weights.grid_mean + metric_weights.C_grid + metric_weights.soc_iqr + metric_weights.soc_std
    safety_weights = (
        ev = metric_weights.ev / metric_weights_sum,
        grid_mean = metric_weights.grid_mean / metric_weights_sum,
        C_grid = metric_weights.C_grid / metric_weights_sum,
        soc_iqr = metric_weights.soc_iqr / metric_weights_sum,
        soc_std = metric_weights.soc_std / metric_weights_sum
    )

    println(io, "┌───────────────────────────────────────────────┐")
    println(io, "│         Performance Evaluation                │")
    println(io, "├───────────────────────────────────────────────┤")
    println(io, "│ EV Performance:                  │ ", lpad(round(ev_perf, digits=4), 10), " │")
    println(io, "│ Grid Performance (Daily Cost):   │ ", lpad(round(C_grid_perf, digits=4), 10), " │")
    println(io, "│ Grid Performance (Mean):         │ ", lpad(round(grid_perf_mean_sigmoid, digits=4), 10), " │")
    println(io, "│ Grid Performance (Margin):       │ ", lpad(round(grid_perf_margin_sigmoid, digits=4), 10), " │")
    println(io, "│ SoC Performance (IQR):           │ ", lpad(round(soc_perf_iqr_weighted, digits=4), 10), " │")
    println(io, "│ SoC Performance (STD):           │ ", lpad(round(soc_perf_std_weighted, digits=4), 10), " │")
    println(io, "│ Operational Projection Metric:   │ ", lpad(round(op_proj_perf, digits=4), 10), " │")
    println(io, "├───────────────────────────────────────────────┤")
    println(io, "│ Operational Projection Breakdown:             │")
    for (key, value) in op_proj_metrics
        println(io, "│ - ", lpad(key, 9), ":                     │ ", lpad(round(value, digits=4), 10), " │")
    end
    println(io, "├───────────────────────────────────────────────┤")
    println(io, "│ Raw EV Performance:                           │")
    println(io, "│ - Avg. Missing Soc:              │ ", lpad(round(ev_avg, digits=4), 10), " │")
    println(io, "│ Raw Grid Performance:                         │")
    println(io, "│ - Mean:                          │ ", lpad(round(grid_perf_mean_orig, digits=4), 10), " │")
    println(io, "│ - Margin:                        │ ", lpad(round(grid_perf_margin_orig, digits=4), 10), " │")
    println(io, "└───────────────────────────────────────────────┘")

    weighted_sum = (
        metric_weights.ev * ev_perf +
        metric_weights.grid_mean * grid_perf_mean_sigmoid +
        metric_weights.C_grid * C_grid_perf +
        metric_weights.soc_iqr * soc_perf_iqr_weighted +
        metric_weights.soc_std * soc_perf_std_weighted +
        metric_weights.operational_proj * op_proj_perf
    )

    no_proj_sum = (
        safety_weights.ev * ev_perf +
        safety_weights.grid_mean * grid_perf_mean_sigmoid +
        safety_weights.C_grid * C_grid_perf +
        safety_weights.soc_iqr * soc_perf_iqr_weighted +
        safety_weights.soc_std * soc_perf_std_weighted
    )

    safety_performance = no_proj_sum * op_proj_min

    safe_x_ev = op_proj_min * ev_perf 
    safe_x_ev_x_grid = safe_x_ev * C_grid_perf
    
    grid_bonus_1st = 0f0
    # First Batch of Tests (less strict)
    if ev_perf ≥ 0.65f0 && op_proj_min ≥ 0.6f0
        grid_bonus_1st += (ev_perf - 0.65f0) * (1f0 + C_grid_perf)
        grid_bonus_1st += (op_proj_min - 0.6f0) * (0.25f0 + C_grid_perf)
    end

    grid_bonus_2nd = 0f0
    # Second Batch of Tests (more strict)
    if ev_perf ≥ 0.75f0 && op_proj_min ≥ 0.8f0
        grid_bonus_2nd += (ev_perf - 0.75f0) * (1f0 + C_grid_perf)
        grid_bonus_2nd += (op_proj_min - 0.8f0) * (0.5f0 + C_grid_perf)
    end

    # function true_performance(hook, C_grid, pDep, daylength)
    #     w_grid = 1f0; 
    #     w_pDep = 1000f0;
    #     w_tess = 1000f0;
    #     sum_Dt = 1.66752f7

    #     components =  ["ξsoc_bess", "ξsoc_ev", "ξsoc_tess", "ξp_tess", "ξp_grid"];
    #     projections = Dict(i => hook.energy[i] for i in components)
    #     seconds_per_timestep = 24 * 3600 / daylength
    #     auxTess_x_seconds = sum(projections["ξsoc_tess"]) * seconds_per_timestep

    #     weighted_sum = (w_grid) * C_grid + w_pDep * pDep + (w_tess) * auxTess_x_seconds
        
    #     # I can add also other projections. 
    #     return auxTess_x_seconds, weighted_sum
    # end

    _, auxtess_daily, auxtess_total = compute_auxtess(hook, daylength)

    println(io, "┌───────────────────────────────────────────────┐")
    println(io, "│               Final Metrics                   │")
    println(io, "├───────────────────────────────────────────────┤")
    println(io, "│ Weighted Sum of Performances:      ", lpad(round(weighted_sum, digits=4), 10), " │")
    println(io, "│ Safety Performance:                           │")
    println(io, "│(Worst Proj. Metric x Performance)             │")
    println(io, "│", lpad(round(op_proj_min, digits=5), 10), lpad("x",11), lpad(round(no_proj_sum, digits=5), 9), "    = ", lpad(round(safety_performance, digits=4), 10), " │")
    println(io, "│ Safe x EV:                         ", lpad(round(safe_x_ev, digits=4), 10), " │")
    println(io, "│ Safe x EV x Grid Perf (daily):     ", lpad(round(safe_x_ev_x_grid, digits=4), 10), " │")
    println(io, "│ Safe x EV + Grid Perf Bonus (1st): ", lpad(round(safe_x_ev + grid_bonus_1st, digits=4), 10), " │")
    println(io, "│ Safe x EV + Grid Perf Bonus (2nd): ", lpad(round(safe_x_ev + grid_bonus_2nd, digits=4), 10), " │")
    println(io, "├───────────────────────────────────────────────┤")
    println(io, "│ True Perf Metric (MPC Obj Funct):             │")
    println(io, "│ - C_grid (total):                  ", lpad(round(C_grid, digits=1), 10), " │")
    println(io, "│ - C_grid (per day):                ", lpad(round(C_grid_per_day, digits=3), 10), " │")
    println(io, "│ - pDep (total):                    ", lpad(round(pDep, digits=3), 10), " │")
    println(io, "│ - pDep (per day):                  ", lpad(round(pDep/n_days, digits=3), 10), " │")
    println(io, "│ - auxTess x time (s) (total):      ", lpad(round(auxtess_total, digits=3), 10), " │")
    println(io, "│ - auxTess x time (s) (per day):    ", lpad(round(mean(auxtess_daily), digits=3), 10), " │")
    # println(io, "│ - True Perf Weighted Sum :         ", lpad(round(true_performance(hook, C_grid, pDep, daylength)[2], digits=3), 10), " │")
    println(io, "└───────────────────────────────────────────────┘")

    return weighted_sum, safety_performance
end


function compute_performance(
    hook::MCES_Hook,
    exog::Exogenous_BatchCollection,
    daylength::Integer = Int32(96),
    ev_soc_dep::Float32 = 0.85f0;
    mem_safe::Bool = false
    )::Float32
    
    n_days = hook.energy["t"][end] / daylength
    C_grid = grid_performance_mean(hook, exog)[3] * Int32(24)/daylength
    C_grid_per_day = C_grid / n_days
    C_grid_perf = C_grid_performance(C_grid_per_day, slope = 1f0, mid = 14.5f0)
    ev_perf = ev_performance(hook, daylength, ev_soc_dep, slope = 15f0, mid = 0.1f0, max_cost = 0.65f0)[1]

    op_proj_min = operational_proj_metric(
        hook, daylength, 
        (soc_bess = 1f0, soc_ev = 1f0, soc_tess = 1f0, p_tess = 1f0, p_grid = 1f0), 
        (soc_bess = 10f0, soc_ev = 10f0, soc_tess = 10f0, p_tess = 10f0, p_grid = 10f0)
    )[3]
    
    safe_x_ev = op_proj_min * ev_perf 
    grid_bonus = 0f0
    
    # First Bacth of Tests ( more strict)
    if ev_perf ≥ 0.65f0 && op_proj_min ≥ 0.6f0
        grid_bonus += (ev_perf - 0.65f0) * (1f0 + C_grid_perf)
        grid_bonus += (op_proj_min - 0.6f0) * (0.25f0 + C_grid_perf)
    end

    if isnan(safe_x_ev + grid_bonus)
        !mem_safe && @warn "Performance Computation resulted in NaN"
        return -1f-2
    end

    return round(safe_x_ev + grid_bonus, digits = 4)
end

function compute_performance(env::MCES_Env, hook::MCES_Hook, exog::Exogenous_BatchCollection; mem_safe::Bool = false)
    compute_performance(hook, exog, env.daylength, env.ev.soc_dep; mem_safe = mem_safe) 
end

function compute_objective_metrics(
    hook::MCES_Hook,
    exog::Exogenous_BatchCollection,
    daylength::Integer = Int32(96),
    ev_soc_dep::Float32 = 0.85f0
    )
   
    _, _, pDep = ev_performance(hook, daylength, ev_soc_dep, slope = 15f0, mid = 0.1f0, max_cost = 0.65f0)

    _, _, total_grid_cost = grid_performance_mean(hook, exog)
    C_grid = total_grid_cost * Int32(24) / daylength

    _, _, auxtess_total = compute_auxtess(hook, daylength)

    return C_grid, pDep, auxtess_total
end


############
# EV Metrics

function ev_metric(x::AbstractFloat, 
    a::Float32 = 15.0f0, # Steepness of the descent
    b::Float32 = 0.3f0,  # Midpoint
    c::Float32 = 0.65f0  # Worst possible infraction
    ) # 0 is worst outcome. 

    # Logistic function 
    f(x) = 1f0 / (1f0 + exp(a * (x - b)))
    
    (f(x) - f(c)) / (f(0) - f(c)) # Normalized to 0 - 1
end

function ev_performance(
    hook::MCES_Hook, daylength::Integer = 96, 
    soc_dep::Float32 = 0.85f0; 
    slope = 15f0, mid = 0.3f0, max_cost = 0.65f0
    )
    γ_ev = hook.energy["γ_ev"]
    soc_ev = hook.energy["soc_ev"]
    
    n = length(γ_ev)
    departures = falses(n)
    days = div(n, daylength)
    for i in 2:n
        departures[i] = (γ_ev[i-1] == 1f0) && (γ_ev[i] == 0f0)
    end
    
    departure_soc = Float32[]
    for i in 1:n
        if departures[i]
            push!(departure_soc, soc_ev[i])
        end
    end
    
    missing_socs = abs.(departure_soc .- soc_dep)
    missing_socs_sq = (departure_soc .- soc_dep).^2
    total_metric = sum(ev_metric(diff, Float32(slope), Float32(mid), Float32(max_cost)) for diff in missing_socs)
    
    return total_metric / days, sum(missing_socs) / days, sum(missing_socs_sq)
end

##############
# Grid metrics

function arbitrage_performance(p_grid::Vector{Float32}, buy_prices::Vector{Float32}, sell_prices::Vector{Float32})
    total_steps = length(p_grid)
    @assert length(buy_prices) == total_steps && length(sell_prices) == total_steps "p_grid, buy_prices, and sell_prices must have the same length"

    performance_scores = Vector{Float32}(undef, total_steps)
    
    # Separate p_grid into buy and sell actions
    buy_actions = [max(0f0, p) for p in p_grid]   # in kW
    sell_actions = [max(0f0, -p) for p in p_grid] # in kW
    
    total_bought = sum(buy_actions)
    total_sold = sum(sell_actions)

    mean_buy_price = mean(buy_prices)
    mean_sell_price = mean(sell_prices)

    for t in 1:total_steps
        buy_score = buy_actions[t] > 0 ? (mean_buy_price - buy_prices[t])/mean_buy_price * 
                    (buy_actions[t]/total_bought) : 0f0
                    
        sell_score = sell_actions[t] > 0 ? (sell_prices[t] - mean_sell_price)/mean_sell_price * 
                    (sell_actions[t]/total_sold) : 0f0 # I can use the mean of the sell actions instead. 

        performance_scores[t] = buy_score + sell_score
    end

    return performance_scores, sum(performance_scores) / total_steps
end

function C_grid_performance(C_grid_per_day; slope = 1f0, mid = 14.5f0)
    return 1 - my_sigmoid(C_grid_per_day, slope = slope, mid = mid)
end

function grid_performance_mean(hook::MCES_Hook, exog::Exogenous_BatchCollection)
    # Assumption: All timesteps have the same length. 
    λ_buy = exog.λ_buy
    T = min(length(λ_buy), length(hook.energy["grid"]))
    λ_sell = exog.λ_sell
    grid = hook.energy["grid"][end + 1 - T: end]
    
    p_grid_pos = [max(0f0, P) for P in grid]  # Grid interactions where energy was bought [kW]
    p_grid_neg = [min(0f0, P) for P in grid]  # Grid interactions where energy was sold (all negative kW)
    
    total_grid_cost = sum((λ_buy[t] * p_grid_pos[t] + λ_sell[t] * p_grid_neg[t]) for t in 1:T) # Assumes timesteps are 1 hour long
    λ_grid = total_grid_cost / max(sum(p_grid_pos), 1f-4)
    λ_buy_mean = mean(λ_buy[1:T])

    m = (λ_buy_mean - λ_grid) / λ_buy_mean

    return m, my_sigmoid(m; slope = 12f0, mid = 0.25f0), total_grid_cost
end

function grid_performance_mean_v2(hook::MCES_Hook, exog::Exogenous_BatchCollection)
    # Assumption: All timesteps have the same length. 
    λ_buy = exog.λ_buy
    T = min(length(λ_buy), length(hook.energy["grid"]))
    λ_sell = exog.λ_sell
    grid = hook.energy["grid"][end + 1 - T: end]
    
    p_grid_pos = [max(0f0, P) for P in grid]  # Grid interactions where energy was bought [kW]
    p_grid_neg = [min(0f0, P) for P in grid]  # Grid interactions where energy was sold (all negative kW)
    
    C_grid = sum((λ_buy[t] * p_grid_pos[t] + λ_sell[t] * p_grid_neg[t]) for t in 1:T)
    λ_grid = C_grid / max(sum((λ_buy[t] * p_grid_pos[t]) for t in 1:T), 1f-4)
    λ_grid = 1 + sum((λ_sell[t] * p_grid_neg[t]) for t in 1:T) / max(sum((λ_buy[t] * p_grid_pos[t]) for t in 1:T), 1f-4)
    # λ_grid equivalent to Cgrid/sum(λ_buy*p_grid_pos) 

    m = 1 - λ_grid
    return m, my_sigmoid(m; slope = 12f0, mid = 0.25f0)
end

function grid_performance_margin(hook::MCES_Hook, exog::Exogenous_BatchCollection)
    λ_buy = exog.λ_buy
    T = min(length(λ_buy), length(hook.energy["grid"]))
    λ_sell = exog.λ_sell
    grid = hook.energy["grid"][end + 1 - T: end]
    scale = mean(λ_buy)/mean(λ_sell) 
    
    p_grid_pos = [max(0f0, P) for P in grid]  # Grid interactions where energy was bought [kW]
    p_grid_neg = [min(0f0, P) for P in grid]  # Grid interactions where energy was sold (all negative kW)
    
    λ_grid_pos = sum(λ_buy[t] * p_grid_pos[t] for t in 1:T) / max(sum(p_grid_pos), 1f-4)
    λ_grid_neg = sum(λ_sell[t] * abs(p_grid_neg[t]) for t in 1:T) / max(abs(sum(p_grid_neg)), 1f-4) # Check in case no energy is sold. 

    m = (λ_grid_neg * scale - λ_grid_pos) / max(λ_grid_neg*scale, λ_grid_pos)
    return m, my_sigmoid(m; slope = 12f0, mid = 0.25f0)
end

@inline function my_sigmoid(x; slope = 12f0, mid = 0.15f0)
    1f0 / (1f0 + exp(-slope * (x - mid)))
end

#################
# SoC Performance

function soc_performance_iqr(
    hook::MCES_Hook,
    weights::NamedTuple{(:soc_ev, :soc_bess, :soc_tess), NTuple{3, Float32}} = (soc_ev = 0.33f0, soc_bess = 0.33f0, soc_tess = 0.33f0)
)

    function normalize_iqr(values::Vector{Float32})
        q1, q3 = quantile(values, [0.25f0, 0.75f0])
        iqr = q3 - q1
        clamped_values = clamp.(values, q1 - 1.5f0 * iqr, q3 + 1.5f0 * iqr)
        return (clamped_values .- q1) ./ iqr
    end

    function performance_metric(soc::Vector{Float32})
        soc_changes = abs.(diff(soc))
        normalized_soc_changes = normalize_iqr(soc_changes)
        return mean(normalized_soc_changes)
    end

    metrics = (
        performance_metric(hook.energy["soc_ev"]),
        performance_metric(hook.energy["soc_bess"]),
        performance_metric(hook.energy["soc_tess"])        
    )

    metrics, sum(metrics .* (weights.soc_ev, weights.soc_bess, weights.soc_tess))
end

function soc_performance_std(
    hook::MCES_Hook,
    weights::NamedTuple{(:soc_ev, :soc_bess, :soc_tess), NTuple{3, Float32}} = (soc_ev = 0.4f0, soc_bess = 0.4f0, soc_tess = 0.2f0)
)
    
    function performance_metric(soc::Vector{Float32})
        std_soc = std(soc)
        max_std = 0.5f0
        return std_soc / max_std  # Normalization
    end

    metrics = (
        performance_metric(hook.energy["soc_ev"]),
        performance_metric(hook.energy["soc_bess"]),
        performance_metric(hook.energy["soc_tess"])        
    )

    metrics, sum(metrics .* (weights.soc_ev, weights.soc_bess, weights.soc_tess))
end

####################
# Projection Metrics

function operational_proj_metric(
    hook::MCES_Hook,
    daylength::Integer = 96,
    weights::NamedTuple{(:soc_bess, :soc_ev, :soc_tess, :p_tess, :p_grid), NTuple{5, Float32}} = (soc_bess = 0.25f0, soc_ev = 0.3f0, soc_tess = 0.15f0, p_tess = 0.15f0, p_grid = 0.15f0),
    β::NamedTuple{(:soc_bess, :soc_ev, :soc_tess, :p_tess, :p_grid), NTuple{5, Float32}} = (soc_bess = 100.0f0, soc_ev = 10.0f0, soc_tess = 100.0f0, p_tess = 100.0f0, p_grid = 100.0f0)
)
    components = ["ξsoc_bess", "ξsoc_ev", "ξsoc_tess", "ξp_tess", "ξp_grid"]
    metrics = Dict{String, Float32}()
    
    for component in components
        projections = hook.energy[component]
        metrics[component] = mean(projections)*daylength # Daily projection mean. 
    end
    
    M_proj = (
        weights.soc_bess * exp(-β.soc_bess * metrics["ξsoc_bess"]),
        weights.soc_ev * exp(-β.soc_ev * metrics["ξsoc_ev"]),
        weights.soc_tess * exp(-β.soc_tess * metrics["ξsoc_tess"]),
        weights.p_tess * exp(-β.p_tess * metrics["ξp_tess"]),
        weights.p_grid * exp(-β.p_grid * metrics["ξp_grid"])
    )
    
    return metrics, sum(M_proj), min(M_proj...)
end

function operational_proj_from_hooks(hooks::Vector{<:MCES_Hook})
    components = ["ξsoc_bess", "ξsoc_ev", "ξsoc_tess", "ξp_tess", "ξp_grid"]
    
    metrics = Dict{String, Vector{Float32}}(
        component => Float32[] for component in components
    )
    
    for hook in hooks
        for component in components
            projections = hook.energy[component]
            push!(metrics[component], sum(projections))
        end
    end
    
    return metrics
end

function compute_auxtess(
    hook::MCES_Hook, 
    daylength::Integer = 96 # timesteps in a real day
    )
    seconds_per_timestep = 24*3600/daylength

    soc_tess = hook.energy["soc_tess"]
    ξsoc_tess = hook.energy["ξsoc_tess"] .* seconds_per_timestep 
    # count = 0
    # Projections for going below the SoCTESS minimum are not considered for AuXTESS
    for i in eachindex(ξsoc_tess)
        if soc_tess[i] < 0.5f0 && ξsoc_tess[i] > 0f0
            # count += 1
            ξsoc_tess[i] = 0f0 
        end
    end
    # println("There were $count lower-bound projections in the SoC TESS.")
    ξsoc_tess_per_day = [sum(day) for day in Iterators.partition(ξsoc_tess, daylength)]

    ξsoc_tess, ξsoc_tess_per_day, sum(ξsoc_tess)
end



@info "The Agent can now be rewarded"