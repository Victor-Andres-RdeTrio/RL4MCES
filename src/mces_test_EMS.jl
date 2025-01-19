include("mces_core.jl")

hooks = []; models = []; envs = Vector{Any}(nothing, 6);
train_new_policy(
    policy_type = "PPO",
    threads = true,
    mem_safe = false,
    seeds = 1, 
    years = 1,
    total_reward = false,
    p = what_parameters(1),
    disc_rate = 0.999f0, gae = 0.4f0,
    init_std = 1f0,
    w_a = 1.0f0, w_e = -0.01f0,
    adam_a = 3f-4, adam_c = 3f-4,
    f = 1, ep_l = 96,
    actor_width = 256, critic_width = 128,
    actor_arch = 4, critic_arch = 1,
    clip_coef = 0.5f0,
    state_buffer_dict = what_features(1),
    reward_shape = 7,
    ev_in_EMS = true,
    projection_to_train = false,
    projection_to_test = false,
    store_h = hooks, store_m = models, store_mces = envs,
    n_test_seeds = 1,
    online_stats = OnlineNorm(14),
    min_δ = nothing, # For Early Stop.
    cum_cost_grid = false
)


#########################################################
# Hyperopt Runs
    first_batch = Dict{String, Vector{Any}}(
        "discount_factor" => [0.6f0, 0.8f0, 0.9f0, 0.95f0, 0.99f0, 0.999f0],
        "gae_parameter" => [0.2f0, 0.4f0, 0.6f0, 0.8f0, 0.9f0, 0.95f0, 0.99f0],
        "initial_std" => [0.1f0, 0.5f0, 1f0, 2f0],
        "w_entropy_loss" => [-1f-2, 0f0, 1f-2],
        "adam_actor" => [3f-5, 1f-4, 3f-4, 1f-3, 3f-3],
        "adam_critic" => [nothing], # same as adam_actor
        "upd_freq" => [1, 2, 3],
        "clip_coef" => [0.1f0, 0.2f0, 0.3f0, 0.5f0],
        "actor_width" => [16, 32, 64, 128, 256, 512],
        "critic_width" => [128],
        "actor_arch" => [1],
        "critic_arch" => [1],
        "actor_activation" => [1],
        "critic_activation" => [1],
        "years" => [3, 6],
        "features" => [2],
        "params" => [1],
        "reward_shape" => [2]
    )

    second_batch = Dict{String, Vector{Any}}(
        "discount_factor" => [0.6f0, 0.8f0, 0.9f0, 0.95f0, 0.99f0, 0.999f0],
        "gae_parameter" => [0.2f0, 0.4f0, 0.6f0, 0.8f0, 0.9f0, 0.95f0, 0.99f0],
        "initial_std" => [0.1f0, 0.5f0, 1f0, 2f0],
        "w_entropy_loss" => [-1f-2, 0f0, 1f-2],
        "adam_actor" => [3f-5, 1f-4, 3f-4, 1f-3],
        "adam_critic" => [3f-5, 1f-4, 3f-4, 1f-3, 3f-3],
        "upd_freq" => [1, 2, 3],
        "clip_coef" => [0.1f0, 0.2f0, 0.3f0, 0.5f0],
        "actor_width" => [16, 32, 64, 128, 256, 512],
        "critic_width" => [16, 32, 64, 128, 256, 512],
        "actor_arch" => [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        "critic_arch" => [1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12],
        "actor_activation" => [1, 2],
        "critic_activation" => [1, 2],
        "years" => [3, 6],
        "features" => [1, 2, 3, 4, 5, 6, 7, 8, 9],
        "params" => [1, 2, 3],
        "reward_shape" => [2, 3, 4, 5, 6, 7, 8, 9]
    )

    true_rew_batch = Dict{String, Vector{Any}}(
        "discount_factor" => [0.6f0, 0.8f0, 0.9f0, 0.95f0, 0.99f0, 0.999f0],
        "gae_parameter" => [0.2f0, 0.4f0, 0.6f0, 0.8f0, 0.9f0, 0.95f0, 0.99f0],
        "initial_std" => [0.1f0, 0.5f0, 1f0, 2f0],
        "w_entropy_loss" => [-1f-2, 0f0, 1f-2],
        "adam_actor" => [3f-5, 1f-4, 3f-4, 1f-3],
        "adam_critic" => [3f-5, 1f-4, 3f-4, 1f-3, 3f-3],
        "upd_freq" => [1, 2, 3],
        "actor_width" => [16, 32, 64, 128, 256, 512],
        "critic_width" => [16, 32, 64, 128, 256, 512],
        "actor_arch" => [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        "critic_arch" => [1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12],
        "actor_activation" => [1, 2],
        "critic_activation" => [1, 2],
        "clip_coef" => [0.1f0, 0.2f0, 0.3f0, 0.5f0],
        "years" => [3, 6],
        "features" => [1, 2, 3, 4, 5, 6, 7, 8, 9],
        "params" => [1, 2, 3],
        "reward_shape" => [2, 4]
    )

    max_outliers_batch = Dict{String, Vector{Any}}(
        "discount_factor" => [0.6f0, 0.8f0, 0.9f0, 0.95f0, 0.99f0],
        "gae_parameter" => [0.2f0, 0.4f0, 0.6f0, 0.8f0, 0.9f0, 0.95f0, 0.99f0],
        "initial_std" => [1f0, 2f0],
        "w_entropy_loss" => [-1f-2, 0f0, 1f-2],
        "adam_actor" => [3f-5, 1f-4, 3f-4, 1f-3],
        "adam_critic" => [3f-5, 1f-4, 3f-4, 1f-3, 3f-3],
        "upd_freq" => [1, 2, 3],
        "actor_width" => [32, 64, 128, 256],
        "critic_width" => [16, 32, 64, 128, 256, 512],
        "actor_arch" => [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        "critic_arch" => [1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12],
        "actor_activation" => [1, 2],
        "critic_activation" => [1, 2],
        "clip_coef" => [0.1f0, 0.2f0, 0.3f0, 0.5f0],
        "years" => [3, 6],
        "features" => [1, 2, 3, 4, 5, 9],
        "params" => [1, 2, 3],
        "reward_shape" => [2, 6, 7, 8, 9]
    ) # remember to change the compute rewards. 
#

ho_vph = threaded_hyperopt("VPG", 500, first_batch; 
    save = true, seeds_per_sample = 3
)


#########################################################
# Easy Validation runs

    a2c_1st_batch_best = Dict( # Look at number of seeds and years
        :policy_type => "A2C",
        :threads => true,
        :mem_safe => false,
        :seeds => 100,
        :years => 6,
        :disc_rate => 0.99f0,
        :gae => 0.9f0,
        :init_std => 0.5f0,
        :w_e => -0.01f0,
        :adam_a => 3.0f-5,
        :adam_c => 3.0f-5,
        :f => 2,
        :actor_width => 512,
        :critic_width => 128,
        :actor_arch => 1,
        :critic_arch => 1,
        :actor_activ => 1, 
        :critic_activ => 1,
        :clip_coef => 0.0f0,
        :state_buffer_dict => what_features(2),
        :p => what_parameters(1),
        :reward_shape => 2,
        :projection_to_train => false,
        :projection_to_test => false,
        :n_test_seeds => nothing,
    )

    a2c_2nd_batch_best = Dict( # Look at number of seeds and years
        :policy_type => "A2C",
        :threads => true,
        :mem_safe => false,
        :seeds => 100,
        :years => 6,
        :disc_rate => 0.99f0,
        :gae => 0.9f0,
        :init_std => 2f0,
        :w_e => 0.01f0,
        :adam_a => 0.0003f0,
        :adam_c => 0.0001f0,
        :f => 2,
        :actor_width => 16,
        :critic_width => 512,
        :actor_arch => 11,
        :critic_arch => 3,
        :actor_activ => 2, 
        :critic_activ => 1,
        :clip_coef => 0.0f0,
        :state_buffer_dict => what_features(5),
        :p => what_parameters(1),
        :reward_shape => 7,
        :projection_to_train => false,
        :projection_to_test => false,
        :n_test_seeds => nothing,
    )

    training_params = Dict(
        :policy_type => "PPO",
        :threads => true,
        :mem_safe => false,
        :seeds => 3,
        :years => 1,
        :disc_rate => 0.6f0,
        :gae => 0.9f0,
        :clip_coef => 0.0f0,
        :state_buffer_dict => what_features(2),
        :p => what_parameters(1),
        :reward_shape => 2,
        :projection_to_train => false,
        :projection_to_test => false,
        :n_test_seeds => nothing,
    )

    train_and_get_CV_performance("3010_test" ; 
                                hard = false, 
                                train_params = training_params
    );

#

#########################################################


# Get CV Peformances
    get_CV_performance(
        "1st\\1030_vpg_CV_hard",
        vpg1_pol;
        feature_dict = what_features(2),
        hard = true,
        folder_path = joinpath(@__DIR__, "..", "saved_policies\\202411_Final_Test\\All_models")
    )
#

#########################################################
# Final Test Run
## Policies
vpg_basic = load_policy(joinpath(@__DIR__, "..", "saved_policies\\202411_Final_Test\\First_Batch\\model_vpg1_best.jld2"));

# Get Test Peformances
    get_test_performance(
        vpg_basic; 
        feature_dict = what_features(2), 
        folder_path = joinpath(@__DIR__, "..", "saved_hooks\\202411_Final_Test\\First_Batch"),
        test_name = "vpg_basic_4",
        rng = Xoshiro(65)
    )
    
#

#####################################################
# Objective Metrics 

    get_objective_confidence_int(
        ppo_ext; 
        exog = exog_test_90,
        feature_dict = what_features(1), 
        folder_path = joinpath(@__DIR__, "..", "saved_policies\\202411_Final_Test\\All_models\\extra\\1121_CI_PPO_ext_Test"),
        safety = true,
        samples = 100,
        id = "Test_Data_PPO_ext"
    )
#

