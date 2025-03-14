# Here will be placed all the modifications of imported code, all the new methods. 


##################################################################################
# Modifying the run and _run functions

function Base.run(
    policy::AbstractPolicy,
    env::MCES_Env,
    W::Exogenous, 
    stop_condition=StopAfterEpisode(1),
    hook=VoidHook(), 
    reset_condition=ResetAtTerminal()
)
    _run(policy, env, W, stop_condition, hook, reset_condition)
end


function _run(policy::AbstractPolicy, env::MCES_Env, W::Exogenous, stop_condition, hook, reset_condition)

    hook(PreExperimentStage(), policy, env)
    policy(PreExperimentStage(), env)
    is_stop = false
    create_safety_model!(env, sf = 1.05f0) # sf = Safety Factor. 

    while !is_stop
        reset!(env)
        policy(PreEpisodeStage(), env)
        hook(PreEpisodeStage(), policy, env)

        while !reset_condition(policy, env) # one episode
            policy(PreActStage(), env)
            hook(PreActStage(), policy, env)
            
            env |> policy |> env
            exogenous!(env, W)
            state_buffer_update!(policy, env)
            reward_update!(env)
            # optimise!(policy) => Optimisation really happens in policy(PostEpisodeStage)

            policy(PostActStage(), env)
            hook(PostActStage(), policy, env)

            if stop_condition(policy, env)
                is_stop = true
                policy(PreActStage(), env)
                hook(PreActStage(), policy, env)
                # policy(env)  # let the policy see the last observation
                break
            end
        end # end of an episode

        if is_terminated(env)
            policy(PostEpisodeStage(), env)  # let the policy see the last observation
            hook(PostEpisodeStage(), policy, env)
        end
    end
    policy(PostExperimentStage(), env)
    hook(PostExperimentStage(), policy, env)

    hook
end

#####################################################################################################
# Run to get average reward
function run_for_reward(house::MCES_Env, policy; exog::Exogenous_BatchCollection = exog_train_ex, years::Integer = 1, total_reward = false, 
    id = "", patience = 150, delta_size = 20, min_δ::Union{Float32, Nothing} = nothing, 
    store_h::Union{Vector,Nothing} = nothing, store_m::Union{Vector,Nothing} = nothing, 
    mem_safe = false, projection_to_test::Bool = false, n_test_seeds::Union{Integer,Nothing} = 1
    )

    hook = mem_safe ? VoidHook() : MCES_Hook()
   
    ############ Training Phase
    trajectory = Trajectory(container=Episode(ElasticArraySARTTraces(state=Float32 => (length(state(house)),), action=Float32 =>(3,))))
    agent = Agent(policy, trajectory)
    stop_condition = StopAfterEpisode_or_Early(episode = fld(exog.last_timestep, house.episode_length),
    patience = patience, size = delta_size, min_δ = min_δ)
    
    for year in Base.OneTo(years)
        stop_condition.cur = 0
        exogenous_reset!(house)
        run(agent, house, exog, stop_condition, hook)
        if stop_condition.stop 
            println(id, " Early Stop at $year/$years.")
            break
        end
        println(id, " year: $year/$years.")
        stop_condition.patience = delta_size # Once the first year is over, the patience is unnecessary. 
    end
    
    if !mem_safe 
        store_h !== nothing && push!(store_h, hook)
        store_m !== nothing && push!(store_m, agent.policy)    
    end

    
    isnothing(n_test_seeds) && return 0f0

    ############ Evaluation Phase
    projection_to_test && use_projection!(house)

    run_dont_learn( 
    house, 
    agent.policy, 
    exog = deepcopy(exog_cv_91), 
    seeds = max(1, n_test_seeds), 
    mem_safe = true, # If mem_safe == true the output will be just the crossvalidation reward, if false it will be the crossvalidation hooks.
    total_reward = total_reward
    )
end

function run_for_reward_alone(house::MCES_Env, policy; exog::Exogenous_BatchCollection = exog_train_ex, years::Int = 1, total_reward = false, 
    id = "", patience = 150, delta_size = 20, min_δ::Union{Float32, Nothing} = nothing, 
    store_h::Union{Vector,Nothing} = nothing, store_m::Union{Vector,Nothing} = nothing, mem_safe = false)

    hook = mem_safe ? MCES_Moderate_Hook() : MCES_Hook()
   
    trajectory = Trajectory(container=Episode(ElasticArraySARTTraces(state=Float32 => (length(state(house)),), action=Float32 =>(3,))))
    agent = Agent(policy, trajectory)
    stop_condition = StopAfterEpisode_or_Early(episode = fld(exog.last_timestep, house.episode_length),
    patience = patience, size = delta_size, min_δ = min_δ)
    
    for year in Base.OneTo(years)
        stop_condition.cur = 0
        exogenous_reset!(house)
        run(agent, house, exog, stop_condition, hook)
        if stop_condition.stop 
            println(id, " Early Stop at $year/$years.")
            break
        end
        println(id, " year: $year/$years.")
        stop_condition.patience = 20
    end
    
    if !mem_safe && store_h !== nothing
        push!(store_h, hook)
    end
    store_m !== nothing && push!(store_m, agent.policy)    
    # GC.gc()
    
    compute_reward(hook, total_reward, 7000, 1f0)
end

#####################################################################################################
# VPG

function run_vpg(;
    threads::Bool = false, 
    mem_safe::Bool = false,
    seeds::Int = 1,
    years::Int = 1,
    exog = exog_train_ex,
    total_reward::Bool = false,
    p = MCES_Params(),
    ep_l::Int = 96,
    ns::Integer = 10,
    na::Integer = 3,
    disc_rate::Float32 = 0.99f0,
    init_std::Float32 = 0.5f0,
    adam_a::Float32 = 1f-4,
    adam_c::Float32 = 1f-4,
    actor_width::Integer = 64,
    critic_width::Integer = 64,
    actor_arch::Integer = 1,
    critic_arch::Integer = 1,
    f::Int = 4,
    store_h::Union{Vector,Nothing} = nothing,
    store_m::Union{Vector,Nothing} = nothing,
    patience::Integer = 150,
    delta_size::Integer = 20,
    min_δ::Union{Float32, Nothing} = nothing
    )

    added_rwd = Atomic{Float32}(0)
    if threads
        @threads for i in Base.OneTo(seeds)
            house = build_MCES(mem_safe = mem_safe, rng = Xoshiro(rand(1:1000)), params = p, episode_length = ep_l)
            policy = myvpg_tune(
                ns = ns, na = na, rng = Xoshiro(rand(1:1000)),
                disc_rate = disc_rate, init_std = init_std, adam_a = adam_a, adam_c = adam_c,
                actor_width = actor_width, critic_width = critic_width,
                actor_arch = actor_arch, critic_arch = critic_arch,
                ep_l = ep_l, f = f, mem_safe = mem_safe
            )
            rwd = run_for_reward(house, policy, exog = deepcopy(exog), years = years, total_reward = total_reward,
                id = "Th: $(threadid()) -> Testing seed $i/$seeds.", store_h = store_h, store_m = store_m,
                patience = patience, delta_size = delta_size, min_δ = min_δ, mem_safe = mem_safe)
            atomic_add!(added_rwd, rwd)
        end
    else # No multithreading
        for i in Base.OneTo(seeds)
            house = build_MCES(mem_safe = mem_safe, rng = Xoshiro(rand(1:1000)), params = p, episode_length = ep_l)
            policy = myvpg_tune(
                ns = ns, na = na, rng = Xoshiro(rand(1:1000)),
                disc_rate = disc_rate, init_std = init_std, adam_a = adam_a, adam_c = adam_c,
                actor_width = actor_width, critic_width = critic_width,
                actor_arch = actor_arch, critic_arch = critic_arch,
                ep_l = ep_l, f = f, mem_safe = mem_safe
            )
            rwd = run_for_reward(house, policy, exog = exog, years = years, total_reward = total_reward,
                id = "Th: $(threadid()) -> Testing seed $i/$seeds.", store_h = store_h, store_m = store_m,
                patience = patience, delta_size = delta_size, min_δ = min_δ, mem_safe = mem_safe)
            atomic_add!(added_rwd, rwd)
        end
    end
    return round(added_rwd[]/seeds, digits = 4)
end

#####################################################################################################
# A2CGAE

function run_a2cgae(;
    threads::Bool = false, 
    mem_safe = false,
    seeds::Int = 1,
    years::Int = 1,
    exog = exog_train_ex,
    total_reward = false,
    p = MCES_Params(),
    ep_l = 96,
    ns = 10,
    na = 3,
    disc_rate = 0.99f0,
    gae = 0.9f0,
    init_std = 0.5f0,
    w_a = 1f0,
    w_e = 0.5f0,
    adam_a = 1f-4,
    adam_c = 1f-4,
    actor_width::Int = 64,
    critic_width::Int = 64,
    actor_arch::Int = 1,
    critic_arch::Int = 1,
    f = 2,
    store_h = nothing,
    store_m = nothing,
    patience::Int = 150,
    delta_size = 20,
    min_δ = nothing
    )

    added_rwd = Atomic{Float32}(0)
    if threads
        @threads for i in Base.OneTo(seeds)
            house = build_MCES(mem_safe = mem_safe, rng = Xoshiro(rand(1:1000)), params = p, episode_length = ep_l)
            policy = a2cgae_tune(
                ns = ns, na = na, rng = Xoshiro(rand(1:1000)),
                disc_rate = disc_rate, gae = gae, init_std = init_std,
                w_a = w_a, w_e = w_e, adam_a = adam_a, adam_c = adam_c,
                actor_width = actor_width, critic_width = critic_width,
                actor_arch = actor_arch, critic_arch = critic_arch, ep_l = ep_l, f = f
            )
            rwd = run_for_reward(house, policy, exog = deepcopy(exog), years = years, total_reward = total_reward,
                id = "Th: $(threadid()) -> Testing seed $i/$seeds.", store_h = store_h, store_m = store_m,
                patience = patience, delta_size = delta_size, min_δ = min_δ, mem_safe = mem_safe)
            atomic_add!(added_rwd, rwd)
        end
    else # No multithreading
        for i in Base.OneTo(seeds)
            house = build_MCES(mem_safe = mem_safe, rng = Xoshiro(rand(1:1000)), params = p, episode_length = ep_l)
            policy = a2cgae_tune(
                ns = ns, na = na, rng = Xoshiro(rand(1:1000)),
                disc_rate = disc_rate, gae = gae, init_std = init_std,
                w_a = w_a, w_e = w_e, adam_a = adam_a, adam_c = adam_c,
                actor_width = actor_width, critic_width = critic_width,
                actor_arch = actor_arch, critic_arch = critic_arch, ep_l = ep_l, f = f
            )
            rwd = run_for_reward(house, policy, exog = exog, years = years, total_reward = total_reward,
                id = "Th: $(threadid()) -> Testing seed $i/$seeds.", store_h = store_h, store_m = store_m,
                patience = patience, delta_size = delta_size, min_δ = min_δ, mem_safe = mem_safe)
            atomic_add!(added_rwd, rwd)
        end
    end
    return round(added_rwd[]/seeds, digits = 4)
end

#####################################################################################################
# PPO

function run_ppo(;
    threads::Bool = false, 
    mem_safe::Bool = false,
    seeds::Integer = 1,
    years::Integer = 1,
    exog = exog_train_ex,
    reward_shape::Integer = 1,
    total_reward = false,
    p = MCES_Params(),
    ep_l = 96,
    # ns = 10,
    na = 3,
    disc_rate = 0.99f0,
    gae = 0.9f0,
    init_std = 0.5f0,
    w_a = 1f0,
    w_e = 0.5f0,
    adam_a = 1f-4,
    adam_c = 1f-4,
    actor_width::Integer = 64,
    critic_width::Integer = 64,
    actor_arch::Integer = 1,
    critic_arch::Integer = 1,
    actor_activ::Integer = 1,
    critic_activ::Integer = 1,
    f = 3,
    clip_coef::Float32 = 0.25f0,
    store_h = nothing,
    store_m = nothing,
    store_mces = nothing,
    patience::Int = 150,
    delta_size = 20,
    min_δ = nothing,
    ev_in_EMS::Bool = true,
    cum_cost_grid::Bool = false,
    online_stats = OnlineNorm(14),
    state_buffer_dict = Dict(),
    projection_to_train::Bool = false,
    projection_to_test::Bool = false,
    n_test_seeds::Integer = 1
    )

    ev = ev_in_EMS ? EV() : nothing

    ns = isempty(state_buffer_dict) ? 44 : count_features(state_buffer_dict)
    
    added_rwd = Atomic{Float32}(0)
    if threads
        @threads for i in 1:seeds
            house = build_MCES(
                mem_safe = mem_safe, rng = Xoshiro(rand(1:1000)), 
                params = p, episode_length = ep_l, ev = ev, 
                cum_cost_grid = cum_cost_grid, 
                state_buffer_dict = state_buffer_dict,
                reward_shape = reward_shape,
                simple_projection = !projection_to_train
                )
            policy = ppo_tune(
                ns = ns, na = na, rng = Xoshiro(rand(1:1000)),
                disc_rate = disc_rate, gae = gae, init_std = init_std,
                w_a = w_a, w_e = w_e, adam_a = adam_a, adam_c = adam_c,
                actor_width = actor_width, critic_width = critic_width,
                actor_arch = actor_arch, critic_arch = critic_arch, 
                actor_activ = actor_activ, critic_activ = critic_activ,
                ep_l = ep_l, f = f, clip_coef = clip_coef, mem_safe = mem_safe,
                online_stats = deepcopy(online_stats),
            )
            rwd = run_for_reward(
                house, policy, exog = deepcopy(exog), years = years, 
                total_reward = total_reward, id = "Th: $(threadid()) -> Testing seed $i/$seeds.", 
                store_h = store_h, store_m = store_m,
                patience = patience, delta_size = delta_size, min_δ = min_δ, 
                mem_safe = mem_safe, projection_to_test = projection_to_test,
                n_test_seeds = n_test_seeds)
            atomic_add!(added_rwd, rwd)
            
            if !isnothing(store_mces) && !mem_safe
                try 
                    store_mces[i] = house
                catch e
                    println("Could not store the MCES. 
                    Error -> $e"
                    )
                end
            end
        end
    else
        for i in 1:seeds
            house = build_MCES(
                mem_safe = mem_safe, rng = Xoshiro(rand(1:1000)), 
                params = p, episode_length = ep_l, ev = ev, 
                cum_cost_grid = cum_cost_grid,
                state_buffer_dict = state_buffer_dict,
                reward_shape = reward_shape,
                simple_projection = !projection_to_train
                )
            policy = ppo_tune(
                ns = ns, na = na, rng = Xoshiro(rand(1:1000)),
                disc_rate = disc_rate, gae = gae, init_std = init_std,
                w_a = w_a, w_e = w_e, adam_a = adam_a, adam_c = adam_c,
                actor_width = actor_width, critic_width = critic_width,
                actor_arch = actor_arch, critic_arch = critic_arch,
                actor_activ = actor_activ, critic_activ = critic_activ,
                ep_l = ep_l, f = f, clip_coef = clip_coef, mem_safe = mem_safe,
                online_stats = deepcopy(online_stats),
            )
            rwd = run_for_reward(
                house, policy, exog = deepcopy(exog), years = years, 
                total_reward = total_reward, id = "Th: $(threadid()) -> Testing seed $i/$seeds.", 
                store_h = store_h, store_m = store_m,
                patience = patience, delta_size = delta_size, min_δ = min_δ, 
                mem_safe = mem_safe, projection_to_test = projection_to_test,
                n_test_seeds = n_test_seeds)
            atomic_add!(added_rwd, rwd)
        end
    end
    return round(added_rwd[]/seeds, digits = 4)
end

function run_ppo_noCV(;
    threads::Bool = false, mem_safe = false,
    seeds::Int = 1,
    years::Int = 1,
    exog = exog_train_ex,
    total_reward = false,
    p = MCES_Params(),
    ep_l = 96,
    ns = 10,
    na = 3,
    disc_rate = 0.99f0,
    gae = 0.9f0,
    std_min = 1f-3,
    w_a = 1f0,
    w_e = 0.5f0,
    adam = 3e-4,
    actor_width::Int = 64,
    critic_width::Int = 64,
    actor_arch::Int = 1,
    critic_arch::Int = 1,
    f = 3,
    clip_coef::Float32 = 0.25f0,
    store_h = nothing,
    store_m = nothing,
    patience::Int = 150,
    delta_size = 20,
    min_δ = nothing
    )

    added_rwd = Atomic{Float32}(0)
    if threads
        @threads for i in 1:seeds
            house = build_MCES(mem_safe = mem_safe, rng = Xoshiro(rand(1:1000)), params = p, episode_length = ep_l)
            policy = ppo_tune(
                ns = ns, na = na, rng = Xoshiro(rand(1:1000)),
                disc_rate = disc_rate, gae = gae, std_min = std_min,
                w_a = w_a, w_e = w_e, adam = adam,
                actor_width = actor_width, critic_width = critic_width,
                actor_arch = actor_arch, critic_arch = critic_arch,
                ep_l = ep_l, f = f, clip_coef = clip_coef, mem_safe = mem_safe
            )
            rwd = run_for_reward_alone(house, policy, exog = exog, years = years, total_reward = total_reward,
                id = "Th: $(threadid()) -> Testing seed $i/$seeds.", store_h = store_h, store_m = store_m,
                patience = patience, delta_size = delta_size, min_δ = min_δ, mem_safe = mem_safe)
            atomic_add!(added_rwd, rwd)
        end
    else
        for i in 1:seeds
            house = build_MCES(mem_safe = mem_safe, rng = Xoshiro(rand(1:1000)), params = p, episode_length = ep_l)
            policy = ppo_tune(
                ns = ns, na = na, rng = Xoshiro(rand(1:1000)),
                disc_rate = disc_rate, gae = gae, std_min = std_min,
                w_a = w_a, w_e = w_e, adam = adam,
                actor_width = actor_width, critic_width = critic_width,
                actor_arch = actor_arch, critic_arch = critic_arch,
                ep_l = ep_l, f = f, clip_coef = clip_coef, mem_safe = mem_safe
            )
            rwd = run_for_reward_alone(house, policy, exog = exog, years = years, total_reward = total_reward,
                id = "Th: $(threadid()) -> Testing seed $i/$seeds.", store_h = store_h, store_m = store_m,
                patience = patience, delta_size = delta_size, min_δ = min_δ, mem_safe = mem_safe)
            atomic_add!(added_rwd, rwd)
        end
    end
    return round(added_rwd[]/seeds, digits = 4)
end


#################################################################################################################
# Running without Learning
function run_free(policy::AbstractPolicy, env::MCES_Env, W::Exogenous, stop_condition, hook, reset_condition = ResetAtTerminal())

    hook(PreExperimentStage(), policy, env)
    policy(PreExperimentStage(), env)
    is_stop = false
    create_safety_model!(env, sf = 1.07f0) # sf = Safety Factor. 

    while !is_stop
        reset!(env)
        policy(PreEpisodeStage(), env)
        hook(PreEpisodeStage(), policy, env)

        while !reset_condition(policy, env) # one episode
            policy(PreActStage(), env)
            hook(PreActStage(), policy, env)
            
            env |> policy |> env
            exogenous!(env, W)
            state_buffer_update!(policy, env, training = false) 
            reward_update!(env)  # Used for Logging Purposes. 

            policy(PostActStage(), env)
            hook(PostActStage(), policy, env)

            if stop_condition(policy, env)
                is_stop = true
                policy(PreActStage(), env)
                hook(PreActStage(), policy, env)
                # policy(env)  # let the policy see the last observation
                break
            end
        end # end of an episode

        if is_terminated(env)
            policy(PostEpisodeStage(), "Dont Learn")  # let the policy see the last observation
            hook(PostEpisodeStage(), policy, env)
        end
    end
    policy(PostExperimentStage(), env)
    hook(PostExperimentStage(), policy, env)

    hook
end


function run_dont_learn(house::MCES_Env, policy; 
    exog::Exogenous_BatchCollection = exog_train_ex, 
    seeds::Integer = 1, mem_safe = false, total_reward = false,
    rng::Union{AbstractRNG,Nothing} = nothing)

    store = Vector{Any}(undef, seeds)
   
    # Easy to make multithreaded (copy house and agent)
    for i in 1:seeds
        hook = MCES_Hook()
        total_recall!(house)
        
        if isa(policy, Union{A2CGAE,PPO,myVPG}) 
            policy.rng = isnothing(rng) ? Xoshiro(rand(1:1000)) : rng
        else
            println("No RNG to modify")
        end
        
        trajectory = Trajectory(container=Episode(ElasticArraySARTTraces(state=Float32 => (length(state(house)),), action=Float32 =>(3,))))
        agent = Agent(policy, trajectory)
        stop_condition = StopAfterEpisode(Int64(fld(exog.last_timestep, house.episode_length)), is_show_progress = false)
        
        run_free(agent, house, exog, stop_condition, hook)

        if mem_safe
            perf = compute_performance(
                hook, exog, house.daylength, house.ev.soc_dep;  
                mem_safe = true
            )  # Extra inputs can be added with the weights, now in default mode. 
            store[i] = perf
        else
            store[i] = hook
        end
    end
    mem_safe && return mean(store)

    return seeds == 1 ? store[] : store
end



function run_dont_learn_policies(house::MCES_Env, policies::Vector; 
    exog::Exogenous_BatchCollection = exog_train_ex, 
    years::Int64 = 1, seeds::Int64 = 1)

    all_hooks = []

    for policy in policies
        hooks = run_dont_learn_seeds(house, policy; exog=exog, seeds=seeds)
        if isa(hooks, Vector)
            push!(all_hooks, hooks...)
        else
            push!(all_hooks, hooks) 
        end
    end

    return all_hooks
end

"""
    run_with_redirect(house, policy; exog, filename::String = "debug.txt")

Execute a simulation with stdout redirected to a debug file.

# Arguments
- `house`: MCES model to simulate.
- `policy`: Agent to apply to the MCES environment.
- `exog`: Exogenous data for the simulation.
- `filename::String = "debug.txt"`: Name of the debug output file.

# Returns
- Result of the `run_dont_learn` function.

# Details
- Creates a debug file.
- Redirects standard output to this file during execution.
- Restores the original stdout after execution completes.
"""
function run_with_redirect(house, policy; exog, filename::String = "debug.txt")
    debug_file = joinpath(@__DIR__, filename)
    original_stdout = stdout

    local h
    open(debug_file, "w") do file
        redirect_stdout(file)
        h = run_dont_learn(house,policy; exog = exog)
        flush(file)
    end

    redirect_stdout(original_stdout)
    return h
end


#########################################################################################
# To run EMS Module Transition Function

function run_TransitionEnv!(house::MCES_Env, policy; 
    exog::Exogenous_BatchCollection)::MCES_Hook

    hook = MCES_Hook()
    trajectory = Trajectory(container=Episode(ElasticArraySARTTraces(state=Float32 => (length(state(house)),), action=Float32 =>(3,))))
    agent = Agent(policy, trajectory)
    stop_condition = StopAfterEpisode(1)
    exogenous_reset!(house)
    run_free(agent, house, exog, stop_condition, hook)
    
    hook
end



@info "Changing ways to run"
 