"""
    new_policy(; policy_type::String = "PPO", ns = 10, na = 3, rng = Random.default_rng(), 
        disc_rate = 0.99f0, init_std::Float32 = 0.5f0, adam_a = 1e-4, adam_c = 1e-4, 
        actor_width::Integer = 64, critic_width::Integer = 64, actor_arch::Integer = 1, 
        critic_arch::Integer = 1, actor_activ = 1, critic_activ = 1, ep_l = 96, f = 2, 
        n_epochs::Integer = 3, mem_safe::Bool = false, online_stats::Normaliser = OnlineNorm(14),
        gae = 0.9f0, w_a = 1f0, w_e = 0.01f0, clip_coef::Float32 = 0.20f0)

Creates a new RL agent (Actor + Critic DNNs) with configurable architecture and hyperparameters.

# Arguments
- `policy_type::String`: Type of policy algorithm to use, options are "PPO", "A2C"/"A2CGAE", or "VPG" (default: "PPO").
- `ns`: Number of state features (default: 10).
- `na`: Number of actions produced by Actor (default: 3).
- `rng`: Random number generator (default: system default).
- `disc_rate`: Discount rate for future rewards (default: 0.99).
- `init_std::Float32`: Initial standard deviation for policy Normal distribution (default: 0.5).
- `adam_a`: Learning rate for actor network (default: 1e-4).
- `adam_c`: Learning rate for critic network (default: 1e-4).
- `actor_width::Integer`: Width of hidden layers in actor network, depending on the chosen arhitecture this value may be scaled (default: 64).
- `critic_width::Integer`: Width of hidden layers in critic network, depending on the chosen arhitecture this value may be scaled (default: 64).
- `actor_arch::Integer`: Architecture type for actor network (default: 1, see `custom_chain` to select the desired architecure).
- `critic_arch::Integer`: Architecture type for critic network (default: 1, see `custom_chain` to select the desired architecure).
- `actor_activ`: Activation function identifier for actor (default: 1).
- `critic_activ`: Activation function identifier for critic (default: 1).
- `ep_l`: Episode length (default: 96).
- `f`: Update frequency (i.e. batch size divisor) (default: 2).
- `n_epochs::Integer`: Number of epochs per update (default: 3).
- `mem_safe::Bool`: Whether to use memory efficient operations (default: false).
- `online_stats::Normaliser`: Online normalisation for state features (default: OnlineNorm(14)).
- `gae`: Generalized Advantage Estimation lambda parameter (default: 0.9).
- `w_a`: Actor loss weight (default: 1.0).
- `w_e`: Entropy loss weight (default: 0.01).
- `clip_coef::Float32`: PPO clipping coefficient (default: 0.20).

# Returns
- A configured policy agent of the specified type.

# Details
1. Validates the policy type and creates appropriate network architectures.
2. Constructs actor and critic networks according to the specified architectures. Some arguments will be ignored if they are not used by the chosen policy type. 
3. Configures memory allocation for tracking training statistics.
4. Returns a fully initialized policy agent ready for training.

# Notes
- Architecture type 4 for actor uses a special network structure that connects the STD output to the environment state (usually independent). Doesn't use `Custom_NN` object.
- Architecture type 4 is not supported for critic networks.
- Memory efficient mode affects how memory is allocated for tracking statistics.
- All policies will use the Normal Distribution by default. 
"""
function new_policy(; policy_type::String = "PPO", # Options are VPG, A2C, PPO
    ns = 10, na = 3, rng = Random.default_rng(), 
    disc_rate = 0.99f0, init_std::Float32 = 0.5f0, adam_a = 1e-4, adam_c = 1e-4, 
    actor_width::Integer = 64, critic_width::Integer = 64, 
    actor_arch::Integer = 1, critic_arch::Integer = 1, 
    actor_activ = 1, critic_activ = 1,
    ep_l = 96, f = 2, n_epochs::Integer = 3,
    mem_safe::Bool = false, online_stats::Normaliser = OnlineNorm(14),
    gae = 0.9f0, w_a = 1f0, w_e = 0.01f0, clip_coef::Float32 = 0.20f0  # Not used in VPG
    )  

    @assert policy_type in ["PPO", "A2C", "A2CGAE", "VPG"] "policy must be PPO, A2C or VPG "

    actor = Approximator(
        model = Custom_NN( 
            π_mean = custom_chain(ns, na, rng, width = actor_width, type = actor_arch, gain = 0.01f0, activation = actor_activ),
            π_std = Chain(
                Dense(fill(inv_softplus(init_std)/ns, (na, ns)), fill(0f0, na), softplus), 
                (x -> x .+ 1f-2)
            )
        ), 
        optimiser = policy_type == "PPO" ? AdaBelief(adam_a, (0.9, 0.999), 1e-7) : AdaBelief(adam_a, (0.9, 0.999))
    )

    if actor_arch == 4 # π_std is incorporated into a parallel network to π_mean. Doesn't use Custom_NN object. 
        actor = Approximator(
            model = custom_chain(ns, na, rng, width = actor_width, type = actor_arch, gain = 0.01f0, activation = actor_activ), 
            optimiser = AdaBelief(adam_a, (0.9, 0.999))
        )
    end
    
    critic_arch == 4 && error("Choose a different architecture for the Critic.")

    create_vector = (size) -> mem_safe ? Float32[] : sizehint!(Float32[], size)
    b_size = round(Int64, ep_l/f)

    if policy_type == "PPO"
        return PPO(
        actor = actor,
        old_actor = deepcopy(actor),
        critic = Approximator(
            model = custom_chain(ns, 1, rng, width = critic_width, type = critic_arch, gain = 1f0, activation = critic_activ),
            optimiser = AdaBelief(adam_c, (0.9, 0.999), 1e-7)
        ),
        γ = disc_rate,
        λ = gae,
        clip_coef = clip_coef,
        dist = Normal,
        max_grad_norm = 0.5f0,
        actor_loss_weight = w_a,
        entropy_loss_weight = w_e, 
        batch_size = b_size,
        rng = rng,
        epochs = n_epochs,
        online_stats = online_stats,
        mem_safe = mem_safe,
        actor_loss = create_vector(5000),
        critic_loss = create_vector(5000),
        entropy_loss = create_vector(5000),
        loss = create_vector(5000),
        raw_loss = create_vector(5000),
        actor_norm = create_vector(5000),
        critic_norm = create_vector(5000),
        avg_adv = create_vector(5000),
        δ = create_vector(5000),
        clip_fracs = create_vector(5000),
        approx_kl = create_vector(5000),
        mean = create_vector(350000),
        std = create_vector(350000)
        )
    end

    if policy_type in ["A2C", "A2CGAE"]
        return A2CGAE(
        actor = actor,
        critic = Approximator(
            model = custom_chain(ns, 1, rng, width = critic_width, type = critic_arch, gain = 1f0, activation = critic_activ),
            optimiser = AdaBelief(adam_c)
        ),
        γ = disc_rate,
        λ = gae,
        dist = Normal,
        max_grad_norm = 0.5f0,
        actor_loss_weight = w_a,
        entropy_loss_weight = w_e, 
        batch_size = b_size,
        rng = rng,
        epochs = n_epochs,
        online_stats = online_stats,
        mem_safe = mem_safe,
        actor_loss = create_vector(5000),
        critic_loss = create_vector(5000),
        entropy_loss = create_vector(5000),
        loss = create_vector(5000),
        raw_loss = create_vector(5000),
        actor_norm = create_vector(5000),
        critic_norm = create_vector(5000),
        avg_adv = create_vector(5000),
        δ = create_vector(5000),
        mean = create_vector(350000),
        std = create_vector(350000)
        )
    end

    if policy_type == "VPG"
        return myVPG(
            actor = actor,
            critic = Approximator(
                model = custom_chain(
                    ns, 1, rng, width = critic_width, type = critic_arch, 
                    gain = 1f0, activation = critic_activ
                    ),
                optimiser = AdaBelief(adam_c)
            ),
            γ = disc_rate,
            dist = Normal,
            batch_size = b_size,
            epochs = n_epochs,
            rng = rng,
            mem_safe = mem_safe,
            actor_loss = create_vector(5000),
            critic_loss = create_vector(5000),
            avg_adv = create_vector(5000),
            raw_loss = create_vector(5000),
            actor_norm = create_vector(5000),
            critic_norm = create_vector(5000),
            mean = create_vector(350000),
            std = create_vector(350000),
            δ = create_vector(5000)
        )    
    end
    return nothing
end

"""
    train_new_policy(;
        policy_type::String = "PPO", 
        threads::Bool = false, mem_safe::Bool = false,
        seeds::Integer = 1, years::Integer = 1,
        exog = exog_train_ex, reward_shape::Integer = 1,
        total_reward = false, p = MCES_Params(), ep_l = 96,
        na = 3, disc_rate = 0.99f0, gae = 0.9f0,
        init_std = 0.5f0, w_a = 1f0, w_e = 0.0f0,
        adam_a = 1f-4, adam_c = 1f-4,
        actor_width::Integer = 64, critic_width::Integer = 64,
        actor_arch::Integer = 1, critic_arch::Integer = 1,
        actor_activ::Integer = 1, critic_activ::Integer = 1,
        f = 3, n_epochs::Integer = 3, clip_coef::Float32 = 0.25f0,
        store_h = nothing, store_m = nothing, store_mces = nothing,
        patience::Int = 150, delta_size = 20,
        min_δ = nothing, ev_in_EMS::Bool = true,
        cum_cost_grid::Bool = false, online_stats = OnlineNorm(14),
        state_buffer_dict = Dict(),
        projection_to_train::Bool = false, projection_to_test::Bool = false,
        n_test_seeds::Union{Integer,Nothing} = 1
    )

Trains a new reinforcement learning Agent for controlling an MCES environment. 

# Arguments
- `policy_type::String`: Type of policy algorithm to use, options are "PPO", "A2C", "A2CGAE", or "VPG" (default: "PPO").
- `threads::Bool`: Whether to use multi-threading for training across seeds (default: false).
- `mem_safe::Bool`: Whether to use memory-efficient operations (default: false).
- `seeds::Integer`: Number of training seeds to run and average results over (default: 1).
- `years::Integer`: Number of simulated years to train for. The yearly data is simply repeated (essentially an epoch)(default: 1).
- `exog`: Exogenous variable data for training (default: exog_train_ex).
- `reward_shape::Integer`: Type of reward shaping to use (default: 1).
- `total_reward`: Whether to use total reward instead of shaped reward (default: false).
- `p`: MCES parameters (default: MCES_Params()).
- `ep_l`: Episode length (default: 96).
- `na`: Number of actions decided by the Agent (default: 3).
- `disc_rate`: Discount rate for future rewards (default: 0.99).
- `gae`: Generalized Advantage Estimation lambda parameter (default: 0.9).
- `init_std`: Initial standard deviation for policy distribution (default: 0.5).
- `w_a`: Actor loss weight (default: 1.0).
- `w_e`: Entropy loss weight (default: 0.0).
- `adam_a`: Learning rate for actor network (default: 1e-4).
- `adam_c`: Learning rate for critic network (default: 1e-4).
- `actor_width::Integer`: Width of hidden layers in actor network, depending on the chosen arhitecture this value may be scaled (default: 64).
- `critic_width::Integer`: Width of hidden layers in critic network, depending on the chosen arhitecture this value may be scaled (default: 64).
- `actor_arch::Integer`: Architecture type for actor network (default: 1, see `custom_chain` to select the desired architecure).
- `critic_arch::Integer`: Architecture type for critic network (default: 1, see `custom_chain` to select the desired architecure).
- `actor_activ::Integer`: Activation function identifier for actor (default: 1).
- `critic_activ::Integer`: Activation function identifier for critic (default: 1).
- `f`: Update frequency (batch size divisor) (default: 3).
- `n_epochs::Integer`: Number of epochs per update (default: 3).
- `clip_coef::Float32`: PPO clipping coefficient (default: 0.25).
- `store_h`: Optional container for storing output hooks (default: nothing).
- `store_m`: Optional container for storing output agents (default: nothing).
- `store_mces`: Optional container for storing output MCES objects (default: nothing).
- `patience::Int`: Number of initial episodes to ignore for early stopping (default: 150).
- `delta_size`: Window size for loss averaging in early stopping (default: 20).
- `min_δ`: Minimum improvement threshold for early stopping (default: nothing, this will deactivate early stopping).
- `ev_in_EMS::Bool`: Whether to include electric vehicle in energy management system (default: true).
- `cum_cost_grid::Bool`: Whether to use cumulative cost grid for rewards (default: false).
- `online_stats`: Online normalization for state features (default: OnlineNorm(14)).
- `state_buffer_dict`: Dictionary defining state buffer configuration (default: Dict()).
- `projection_to_train::Bool`: Whether to use safety projection during training (default: false). Selecting true will incur considerable computational cost, as every decision made will be projected.
- `projection_to_test::Bool`: Whether to use safety projection during testing (default: false). Select true for final validation tests, as this represents the real world conditions of operation (when safety is a top priority).
- `n_test_seeds::Union{Integer,Nothing}`: Number of test seeds to evaluate with (default: 1).

# Returns
- The average reward across all seeds, rounded to 4 decimal places. Produced hooks, agents and environments will be stored with provided with Arrays in `store_h`, `store_m`, and `store_mces` respectively.

# Details
1. Validates the policy type is one of the supported options.
2. Determines the state dimension based on the state buffer configuration.
3. Creates and trains policy agents for each seed, either sequentially or in parallel (if threads=true).
4. For each seed:
   - Builds an MCES environment with specified parameters
   - Creates a new policy agent of the specified type (see `new_policy`)
   - Runs the policy for the specified number of years (see `run_for_reward`)
   - Accumulates the reward. Optionally stores the Hook and Agent. 
   - Optionally stores the MCES object, if requested
5. Returns the average reward across all seeds.

# Notes
- The function supports optional safety projections which can be applied during training and/or testing.
- When `projection_to_train=true`, the safe projection mechanism is used to dynamically adjust action limits
  at each timestep based on system component constraints, which will reduce training efficiency.
- When `projection_to_test=true`, the safe projection is applied during testing/validation, which
  simulates real-world conditions and ensures operational safety. Incurs a cost but not as severe as `projection_to_train=true`.
- Early stopping is implemented to prevent overfitting and save computational resources.
- Multi-threading can be used to train multiple seeds in parallel.
"""
function train_new_policy(;
    policy_type::String = "PPO", 
    threads::Bool = false, 
    mem_safe::Bool = false,
    seeds::Integer = 1,
    years::Integer = 1,
    exog = exog_train_ex,
    reward_shape::Integer = 1,
    total_reward = false,
    p = MCES_Params(),
    ep_l = 96,
    na = 3, # number of actions
    disc_rate = 0.99f0,
    gae = 0.9f0,
    init_std = 0.5f0,
    w_a = 1f0,
    w_e = 0.0f0,
    adam_a = 1f-4,
    adam_c = 1f-4,
    actor_width::Integer = 64,
    critic_width::Integer = 64,
    actor_arch::Integer = 1,
    critic_arch::Integer = 1,
    actor_activ::Integer = 1,
    critic_activ::Integer = 1,
    f = 3,
    n_epochs::Integer = 3,
    clip_coef::Float32 = 0.25f0,
    store_h = nothing, # For storing output Hooks in global variables
    store_m = nothing, # For storing output Agents (DNN) in global variables
    store_mces = nothing, # For storing output MCES Objects in global variables
    patience::Int = 150,
    delta_size = 20,
    min_δ = nothing,
    ev_in_EMS::Bool = true,
    cum_cost_grid::Bool = false,
    online_stats = OnlineNorm(14),
    state_buffer_dict = Dict(),
    projection_to_train::Bool = false,
    projection_to_test::Bool = false,
    n_test_seeds::Union{Integer,Nothing} = 1
    )

    @assert policy_type in ["PPO", "A2C", "A2CGAE", "VPG"] "policy must be PPO, A2C or VPG "
    ev = ev_in_EMS ? EV() : nothing
    ns = isempty(state_buffer_dict) ? 44 : count_features(state_buffer_dict)  # dimensions of the feature vector. 
    
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
            policy = new_policy( policy_type = policy_type,
                ns = ns, na = na, rng = Xoshiro(rand(1:1000)),
                disc_rate = disc_rate, gae = gae, init_std = init_std,
                w_a = w_a, w_e = w_e, adam_a = adam_a, adam_c = adam_c,
                actor_width = actor_width, critic_width = critic_width,
                actor_arch = actor_arch, critic_arch = critic_arch, 
                actor_activ = actor_activ, critic_activ = critic_activ,
                ep_l = ep_l, f = f, clip_coef = clip_coef, n_epochs = n_epochs,
                mem_safe = mem_safe, online_stats = deepcopy(online_stats),
            )
            rwd = run_for_reward(
                house, policy, exog = deepcopy(exog), years = years, 
                total_reward = total_reward, id = "Th: $(threadid()) -> Testing seed $i/$seeds.", 
                store_h = store_h, store_m = store_m,
                patience = patience, delta_size = delta_size, min_δ = min_δ, 
                mem_safe = mem_safe, projection_to_test = projection_to_test,
                n_test_seeds = n_test_seeds
                )
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
            policy = new_policy( policy_type = policy_type,
                ns = ns, na = na, rng = Xoshiro(rand(1:1000)),
                disc_rate = disc_rate, gae = gae, init_std = init_std,
                w_a = w_a, w_e = w_e, adam_a = adam_a, adam_c = adam_c,
                actor_width = actor_width, critic_width = critic_width,
                actor_arch = actor_arch, critic_arch = critic_arch, 
                actor_activ = actor_activ, critic_activ = critic_activ,
                ep_l = ep_l, f = f, clip_coef = clip_coef, n_epochs = n_epochs,
                mem_safe = mem_safe, online_stats = deepcopy(online_stats)
            )
            rwd = run_for_reward(
                house, policy, exog = deepcopy(exog), years = years, 
                total_reward = total_reward, id = "Th: $(threadid()) -> Testing seed $i/$seeds.", 
                store_h = store_h, store_m = store_m,
                patience = patience, delta_size = delta_size, min_δ = min_δ, 
                mem_safe = mem_safe, projection_to_test = projection_to_test,
                n_test_seeds = n_test_seeds
                )
            atomic_add!(added_rwd, rwd)
        end
    end
    return round(added_rwd[]/seeds, digits = 4)
end


"""
    threaded_hyperopt(
        policy_type::String = "PPO", 
        samples::Integer = 10, 
        parameters::Dict = Dict(); 
        save = true, 
        seeds_per_sample::Integer = 3
    )

Performs hyperparameter optimization for reinforcement learning policies trained in the MCES environment.

# Arguments
- `policy_type::String`: Type of policy algorithm to use, options are "PPO", "A2C", or "VPG" (default: "PPO").
- `samples::Integer`: Number of hyperparameter combinations to evaluate (default: 10).
- `parameters::Dict`: Dictionary of hyperparameter ranges to explore (default: empty Dict).
- `save::Bool`: Whether to save the hyperparameter optimization results (default: true).
- `seeds_per_sample::Integer`: Number of training seeds to run and average for each hyperparameter combination (default: 3).

# Returns
- A `Hyperopt` object containing the optimization results.

# Details
1. Uses atomic counters to track progress across multiple threads.
2. Automatically handles parameter range restrictions based on the selected policy type:
   - For "A2C" and "VPG", sets clip coefficient to 0.
   - For "VPG", sets GAE parameter and entropy loss weight to 0.
3. Employs BOHB (Bayesian Optimization with Hyperband) for efficient hyperparameter search.
4. Evaluates each hyperparameter combination by training a policy and measuring its performance.
5. Optionally saves the optimization results to disk.

# Notes
- The function supports exploration of up to 18 different hyperparameters simultaneously.
- Default parameter ranges are provided if not specified in the input dictionary.
- Policy evaluation includes safety projection during testing but not during training.
- The objective function is the negative of the reward (multiplied by 100) to convert the maximization problem to a minimization problem.
- Results are saved in the parent directory.

# Example parameter ranges
```julia
Example_Hyperopt_Dict = Dict{String, Vector{Any}}(
   "discount_factor" => [0.6f0, 0.8f0, 0.9f0, 0.95f0, 0.99f0, 0.999f0],
   "gae_parameter" => [0.2f0, 0.4f0, 0.6f0, 0.8f0, 0.95f0],
   "initial_std" => [0.5f0, 1f0, 2f0],
   "w_entropy_loss" => [-1f-2, 0f0, 1f-2],
   "adam_actor" => [3f-5, 1f-4, 3f-4, 1f-3],
   "adam_critic" => [3f-5, 1f-4, 3f-4, 1f-3, 3f-3],
   "upd_freq" => [1, 2, 3],
   "actor_width" => [32, 64, 128, 256, 512],
   "critic_width" => [32, 64, 128, 256, 512],
   "actor_arch" => [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
   "critic_arch" => [1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12],
   "actor_activation" => [1, 2],
   "critic_activation" => [1, 2],
   "clip_coef" => [0.1f0, 0.2f0, 0.3f0, 0.5f0],
   "years" => [3, 6],
   "features" => [1, 2, 3, 4, 5, 6, 7, 8, 9],
   "params" => [1, 2, 3],
   "reward_shape" => [2, 3, 4, 5, 6, 7, 8, 9]
)
```
"""
function threaded_hyperopt(
    policy_type::String = "PPO", samples::Integer = 10, 
    parameters::Dict = Dict(); save = true, 
    seeds_per_sample::Integer = 3
    )
    
    samples_count = Atomic{Int32}(0)

    function get_param(key::String, default_value=nothing)
        if !haskey(parameters, key)
            @warn "Key '$key' not found in parameters dictionary. Using default values."
            return default_value
        end
        return parameters[key]
    end

    discount_factor_range = get_param("discount_factor", [0.6f0, 0.8f0, 0.9f0, 0.95f0, 0.99f0, 0.999f0])
    gae_parameter_range = get_param("gae_parameter", [0.2f0, 0.4f0, 0.6f0, 0.8f0, 0.95f0])
    initial_std_range = get_param("initial_std", [0.5f0, 1f0, 2f0])
    w_entropy_loss_range = get_param("w_entropy_loss", [-1f-2, 0f0, 1f-2])
    adam_actor_range = get_param("adam_actor", [3f-5, 1f-4, 3f-4, 1f-3])
    adam_critic_range = get_param("adam_critic", [3f-5, 1f-4, 3f-4, 1f-3, 3f-3])
    upd_freq_range = get_param("upd_freq", [1, 2, 3])
    actor_width_range = get_param("actor_width", [32, 64, 128, 256, 512])
    critic_width_range = get_param("critic_width", [32, 64, 128, 256, 512])
    actor_arch_range = get_param("actor_arch", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    critic_arch_range = get_param("critic_arch", [1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12])
    actor_activation_range = get_param("actor_activation", [1, 2])
    critic_activation_range = get_param("critic_activation", [1, 2])
    years_range = get_param("years", [3, 6])
    features_range = get_param("features", [1, 2, 3, 4, 5, 6, 7, 8, 9])
    params_range = get_param("params", [1, 2, 3])
    reward_shape_range = get_param("reward_shape", [2, 3, 4, 5, 6, 7, 8, 9])
    clip_coef_range = get_param("clip_coef", [0.1f0, 0.2f0, 0.3f0, 0.5f0]) 

    if policy_type in ["A2C", "VPG"] 
        clip_coef_range = [0f0]
    end

    if policy_type == "VPG"
        gae_parameter_range = [0f0]
        w_entropy_loss_range = [0f0]
    end

    ho = @thyperopt for i in samples,
        sampler = BOHB(dims=[Hyperopt.Continuous() for _ in 1:18]),
        disc_rate = discount_factor_range,
        gae = gae_parameter_range,
        init_std = initial_std_range,
        w_e_loss = w_entropy_loss_range,
        adam_a = adam_actor_range,
        adam_c = adam_critic_range,
        upd_freq = upd_freq_range,
        actor_width = actor_width_range,
        critic_width = critic_width_range,
        actor_arch = actor_arch_range,
        critic_arch = critic_arch_range,
        actor_activ = actor_activation_range,
        critic_activ = critic_activation_range,
        clip_coef = clip_coef_range,
        years = years_range,
        features = features_range,
        params = params_range,
        rew_shape = reward_shape_range

        atomic_add!(samples_count, Int32(1))
        println("\n Testing $(samples_count[])/$samples Samples")
        - 1f2 * train_new_policy( # A negative sign is used because the Hyperopt module tries to minimise the objective. 
            policy_type = policy_type,
            threads = false, 
            mem_safe = true,
            seeds = seeds_per_sample, 
            years = years, 
            total_reward = false, 
            p = what_parameters(params), 
            disc_rate = disc_rate, gae = gae, 
            init_std = init_std,
            w_a = 1f0, 
            w_e = w_e_loss, 
            adam_a = adam_a, 
            adam_c = !isnothing(adam_c) ? adam_c : adam_a, 
            ep_l = 96, 
            actor_width = actor_width,
            critic_width = critic_width,
            actor_arch = actor_arch,
            critic_arch = critic_arch,
            actor_activ = actor_activ,
            critic_activ = critic_activ,
            clip_coef = clip_coef,
            reward_shape = rew_shape,
            f = upd_freq,
            state_buffer_dict = what_features(features),
            ev_in_EMS = true,
            cum_cost_grid = false,
            min_δ = nothing,
            projection_to_train = false, # It is highly recommended to keep this Boolean false, otherwise computational cost increases considerably. 
            projection_to_test = true,
            n_test_seeds = 1 # Once the agent is trained, this value defines how many test runs are averaged to get the estimate for the agent's performance.
        )
    end
    
    save && save_hyperopt(ho, path = joinpath(@__DIR__,".."), 
    file = "hypertune_log.txt", policy_type = policy_type, n_seeds = seeds_per_sample,
    extra = "")
    
    ho
end


@info "Policies can be created, trained and optimised"