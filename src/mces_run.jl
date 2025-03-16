
##################################################################################
# Modifying the run and _run functions from ReinforcementLearning.jl

"""
    run(policy::AbstractPolicy, env::MCES_Env, W::Exogenous, stop_condition=StopAfterEpisode(1), hook=VoidHook(), reset_condition=ResetAtTerminal())

Run a reinforcement learning experiment with the given policy in the MCES environment. Modified from the ReinforcementLearning.jl package.

# Arguments
- `policy::AbstractPolicy`: The policy to use for decision making (i.e. the Agent).
- `env::MCES_Env`: The MCES environment to run the experiment in.
- `W::Exogenous`: The exogenous data necessary for transitioning to a new state in the simulation.
- `stop_condition=StopAfterEpisode(1)`: Condition that determines when to stop the experiment.
- `hook=VoidHook()`: Hook for gathering data during the experiment.
- `reset_condition=ResetAtTerminal()`: Condition that determines when to reset the environment.

# Returns
- The result of calling the internal `_run` function.
"""
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

"""
    _run(policy::AbstractPolicy, env::MCES_Env, W::Exogenous, stop_condition, hook, reset_condition)

Internal implementation of the reinforcement learning experiment runner. Modified from the ReinforcementLearning.jl package.

# Arguments
- `policy::AbstractPolicy`: The policy to use for decision making.
- `env::MCES_Env`: The MCES environment to run the experiment in.
- `W::Exogenous`: The exogenous data necessary for transitioning to a new state.
- `stop_condition`: Condition that determines when to stop the experiment.
- `hook`: Hook for gathering data during the experiment.
- `reset_condition`: Condition that determines when to reset the environment.

# Returns
- The hook object containing collected data during the experiment.

# Details
1. Initializes the experiment with policy and environment setup.
2. Creates a safety model with a safety factor of 1.07.
3. For each episode:
   - Resets the environment.
   - Runs policy and hook pre-episode callbacks.
   - While the reset condition is not met:
     - Executes the policy action sequence (learn the state of Env, then infer actions, then apply actions)
     - Updates environment with exogenous data and transitions to new timestep (see `exogenous!`).
     - Updates the state buffer and reward.
     - Runs post-action callbacks.
     - Checks for stop condition.
   - If environment terminated, runs post-episode callbacks.
4. Finalizes the experiment with post-experiment callbacks.
"""
function _run(policy::AbstractPolicy, env::MCES_Env, W::Exogenous, stop_condition, hook, reset_condition)

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

"""
   run_for_reward(house::MCES_Env, policy; 
                 exog::Exogenous_BatchCollection = exog_train_ex, 
                 years::Integer = 1, 
                 id = "", 
                 patience = 150, 
                 delta_size = 20, 
                 min_δ::Union{Float32, Nothing} = nothing, 
                 store_h::Union{Vector,Nothing} = nothing, 
                 store_m::Union{Vector,Nothing} = nothing, 
                 mem_safe = false, 
                 exog_to_test::Exogenous_BatchCollection = exog_cv_91,
                 projection_to_test::Bool = false, 
                 n_test_seeds::Union{Integer,Nothing} = 1)

Run a reinforcement learning training and evaluation process to obtain average reward.

# Arguments
- `house::MCES_Env`: The MCES environment to run the experiment in.
- `policy`: The policy to use for decision making.
- `exog::Exogenous_BatchCollection = exog_train_ex`: Exogenous data for training.
- `years::Integer = 1`: Number of years to train.
- `id = ""`: Identifier string for logging.
- `patience = 150`: Patience parameter for early stopping.
- `delta_size = 20`: Size parameter for early stopping evaluation.
- `min_δ::Union{Float32, Nothing} = nothing`: Minimum delta for early stopping.
- `store_h::Union{Vector,Nothing} = nothing`: Optional vector to store hooks.
- `store_m::Union{Vector,Nothing} = nothing`: Optional vector to store policies.
- `mem_safe = false`: Whether to operate in memory efficient mode.
- `exog_to_test::Exogenous_BatchCollection = exog_cv_91`: Exogenous data for evaluation/testing.
- `projection_to_test::Bool = false`: Whether to use safe projection during testing.
- `n_test_seeds::Union{Integer,Nothing} = 1`: Number of test seeds for evaluation.

# Returns
- Average reward from evaluation phase or 0f0 if n_test_seeds is nothing.

# Details
1. Training phase:
  - Sets up an agent with the provided policy.
  - Trains for the specified number of years with the `exog` data provided.
  - Uses early stopping logic based on patience, delta size, and min_δ parameters.
  - Stores hooks and policies if requested and not in memory efficient mode.
2. Evaluation phase (if n_test_seeds is not nothing):
  - Optionally applies projection for testing when projection_to_test is true.
  - Calls `run_dont_learn` function for evaluation.
  - Uses the provided `exog_to_test` data (defaults to exog_cv_91) for validation of performance.
"""
function run_for_reward(house::MCES_Env, policy; exog::Exogenous_BatchCollection = exog_train_ex, years::Integer = 1, 
    id = "", patience = 150, delta_size = 20, min_δ::Union{Float32, Nothing} = nothing, 
    store_h::Union{Vector,Nothing} = nothing, store_m::Union{Vector,Nothing} = nothing, 
    mem_safe = false, exog_to_test::Exogenous_BatchCollection = exog_cv_91,
    projection_to_test::Bool = false, n_test_seeds::Union{Integer,Nothing} = 1
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
    exog = deepcopy(exog_to_test), 
    seeds = max(1, n_test_seeds), 
    mem_safe = true, # If mem_safe == true the output will be just the crossvalidation reward, if false it will be the crossvalidation hooks.
    )
end


#################################################################################################################
# Running without Learning

"""
    run_free(policy::AbstractPolicy, env::MCES_Env, W::Exogenous, stop_condition, hook, reset_condition = ResetAtTerminal())

Run a reinforcement learning experiment without learning (evaluation only).

# Arguments
- `policy::AbstractPolicy`: The policy to use for decision making.
- `env::MCES_Env`: The MCES environment to run the experiment in.
- `W::Exogenous`: The exogenous data necessary for transitioning to a new state.
- `stop_condition`: Condition that determines when to stop the experiment.
- `hook`: Hook for gathering data during the experiment.
- `reset_condition = ResetAtTerminal()`: Condition that determines when to reset the environment.

# Returns
- The hook object containing collected data during the experiment.

# Details
This function is similar to `_run` but designed for evaluation without learning:
1. Initializes the experiment with policy and environment setup.
2. Creates a safety model with a safety factor of 1.07.
3. For each episode:
   - Similar execution to `_run` but with training=false for state buffer updates.
   - Specifically skips learning in the post-episode stage by passing a String ("Dont Learn").
4. Finalizes the experiment with post-experiment callbacks.
"""
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

"""
    run_dont_learn(house::MCES_Env, policy; 
                  exog::Exogenous_BatchCollection = exog_train_ex, 
                  seeds::Integer = 1, 
                  mem_safe = false,
                  rng::Union{AbstractRNG,Nothing} = nothing)

Run an evaluation of a policy without learning, potentially with multiple seeds.

# Arguments
- `house::MCES_Env`: The MCES environment to evaluate in.
- `policy`: The policy to evaluate.
- `exog::Exogenous_BatchCollection = exog_train_ex`: Exogenous data for evaluation.
- `seeds::Integer = 1`: Number of evaluation seeds/runs.
- `mem_safe = false`: Whether to operate in memory efficient mode.
- `rng::Union{AbstractRNG,Nothing} = nothing`: Optional random number generator.

# Returns
- If mem_safe is true: Mean performance across all seeds.
- If mem_safe is false: Vector of hooks, or single hook if seeds=1.

# Details
1. Prepares storage for evaluation results.
2. For each seed:
   - Creates a new hook.
   - Resets the environment.
   - Sets the RNG for supported policy types (A2CGAE, PPO, myVPG).
   - Sets up an agent and stop condition.
   - Calls `run_free` for evaluation.
   - Stores either performance metrics or hooks based on mem_safe setting.
3. Returns the appropriate result structure.
"""
function run_dont_learn(house::MCES_Env, policy; 
    exog::Exogenous_BatchCollection = exog_train_ex, 
    seeds::Integer = 1, mem_safe = false,
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


@info "Changing ways to run"
 