function train_and_get_CV_performance(
    folder_name::String ; hard::Bool = false, 
    folder_path::String = joinpath(@__DIR__, "..", "Figs\\debug\\"),
    train_params::Dict = Dict())

    seeds = get(train_params, :seeds, 1)
    feature_dict = get(train_params, :state_buffer_dict, Dict())

    folder = joinpath(folder_path, folder_name)

    performance_file = joinpath(folder, "CV_performance.txt")
    open(performance_file, "w") do file
        write(file, "=============================\n")
        write(file, "Training Information \n")
        write(file, "Policy Type -> $(get(train_params, :policy_type, "Default"))\n")
        write(file, "Reward Type -> $(get(train_params, :reward_shape, "Default"))\n")
        write(file, "Training Seeds -> $(seeds)\n")
        write(file, "CV -> $(hard ? "Hard" : "Easy")\n")
        write(file, "=============================\n\n")
    end

    print_dict_to_file(train_params, performance_file, title = "All the parameters used as arguments:")

    model_storage = []
    println("The training of $seeds seeds starts NOW!!")
    train_new_policy(; train_params..., store_m = model_storage, n_test_seeds = nothing)    
    println("Finished Training\n\n\n\n")

    get_CV_performance(
        folder_name, model_storage;
        feature_dict = feature_dict,
        hard = hard,
        folder_path = folder_path
    )
    
    return nothing
end

function get_CV_performance(
    folder_name::String , model_array::Array; 
    feature_dict::Dict = Dict(), 
    hard::Bool = false,
    folder_path::String = joinpath(@__DIR__, "..", "Figs\\debug\\")
    )

    folder = joinpath(folder_path, folder_name)
    performance_file = joinpath(folder, "CV_performance.txt")
    open(performance_file, "a") do file
        write(file, "======================\n")
        write(file, "CV Performance Results\n")
        write(file, "CV -> $(hard ? "Hard" : "Easy")\n")
        write(file, "======================\n\n")
    end    

    models = length(model_array)
    result_array = Vector{Union{Float32,Nothing}}(nothing, models)
    samples_count = Atomic{Int32}(0)
    
    @threads for i in eachindex(model_array)
        
        house = build_MCES(
            params = what_parameters(1), 
            simple_projection = false, 
            mem_safe = false,
            state_buffer_dict = feature_dict,
            reward_shape = 1,
            episode_length = 96,
        )

        ex = hard ? deepcopy(exog_cv_hard_91) : deepcopy(exog_cv_91)

        h = run_dont_learn(
            house, 
            model_array[i];    
            exog = ex,
            seeds = 1
        )

        sleep(10*sqrt(i)) # avoid overwritting the performance file 

        save_policy(
            joinpath(folder, "model$i$(hard ? "h" : "e").jld2"), 
            model_array[i]
        )

        open(performance_file, "a") do file
            write(file, "Seed $i Results:\n")
            write(file, "-----------------\n")
            compute_comprehensive_performance(h, ex, house.daylength, house.ev.soc_dep; io=file)
            write(file, "\n\n")
        end

        result_array[i] = compute_performance(h, ex, house.daylength, house.ev.soc_dep)
        atomic_add!(samples_count, Int32(1))

        samples_count[] % 5 == 0 && println("\n Tested $(samples_count[])/$models Models")
    end

    result_array = filter(x -> !isnothing(x), result_array)

    open(performance_file, "a") do file
        write(file, "========================\n")
        write(file, "Original nº seeds -> $(models)\n")
        write(file, "Recorded nº seeds -> $(length(result_array))\n\n")
        write(file, "CV Mean -> $(round(mean(result_array), digits=5))\n") 
        write(file, "CV Std  -> $(round(std(result_array), digits=5))\n")
        write(file, "CV Max  -> $(round(maximum(result_array), digits=5))\n")
        write(file, "========================\n\n")
    end  
    write_best_seeds(result_array, performance_file, append = true)

    open(performance_file, "a") do file
        write(file, "Full Result Array\n")
        write(file, string(result_array))
        write(file, "\n========================\n\n")
    end  

    println("Finished Getting CV Performances\n\n\n\n\n")
    return nothing
end

function get_test_performance(
    model; 
    feature_dict::Dict = Dict(), 
    folder_path::String = joinpath(@__DIR__, "..", "Figs\\debug\\"),
    test_name::String = "$(rand(1:100))",
    safety::Bool = true,
    rng = Xoshiro(1107)
    )

    folder = folder_path
    performance_file = joinpath(folder, "Test_Performance_$(test_name).txt")

    # Build the test scenario with specified features
    house = build_MCES(
        params = what_parameters(1), 
        simple_projection = !safety, 
        mem_safe = false,
        state_buffer_dict = feature_dict,
        reward_shape = 1,
        episode_length = 96,
    )

    ex = deepcopy(exog_test_90)  

    # Run test without learning
    h = run_dont_learn(
        house, 
        model;    
        exog = ex,
        seeds = 1,
        rng = rng
    )

    save_MCES_Hook(
        joinpath(folder, "hook_$(test_name).jld2"), 
        h
    ) 

    # Record comprehensive performance
    open(performance_file, "a") do file
        write(file, "Test Results:\n")
        write(file, "-----------------\n")
        compute_comprehensive_performance(h, ex, house.daylength, house.ev.soc_dep; io=file)
        write(file, "\n\n")
    end

    println("Finished Testing Model Performance\n")
    return nothing
end

function get_objective_confidence_int(
    policy; 
    exog::Exogenous_BatchCollection,
    feature_dict::Dict = Dict(), 
    folder_path::String = joinpath(@__DIR__, "..", "Figs\\debug\\"),
    safety::Bool = true,
    samples::Integer = 10,
    id::String = ""
    )
     
    folder = folder_path
    performance_file = joinpath(folder, "Objective_Performance_$(id).txt")
    Cgrid = zeros(Float32, samples) ; pDep = zeros(Float32, samples) ; auxtess = zeros(Float32, samples)

    for i in Base.OneTo(samples)
        house = build_MCES(
            params = what_parameters(1), 
            simple_projection = !safety, 
            mem_safe = false,
            state_buffer_dict = feature_dict,
            reward_shape = 1,
            episode_length = 96,
        )

        ex = deepcopy(exog)  

        h = run_dont_learn(
            house, 
            policy;    
            exog = ex,
            seeds = 1,
            mem_safe = false,
        )

        save_MCES_Hook(
            joinpath(folder, "hook_$(id)_$i.jld2"), 
            h
        ) 

        Cgrid[i], pDep[i], auxtess[i] = compute_objective_metrics(h, ex, Int32(96), 0.85f0)

    end

    # Record comprehensive performance
    open(performance_file, "a") do file
        write(file, "Objective Metric Results:\n")
        write(file, "-----------------\n")
        println(file, "Cgrid Stats:")
        print_bootstrap_ci_mean(Cgrid; io = file)
        write(file, "Cgrid -> $(string(Cgrid)) \n\n")
        write(file, "-----------------\n")
        println(file, "pDep Stats:")
        print_bootstrap_ci_mean(pDep; io = file)
        write(file, "pDep -> $(string(pDep)) \n\n")
        write(file, "-----------------\n")
        println(file, "AuxTESS Stats:")
        print_bootstrap_ci_mean(auxtess; io = file)
        write(file, "AuxTESS -> $(string(auxtess)) \n\n")
        write(file, "\n\n")
    end

    println("Finished Testing Objective Performance\n")
    return nothing
end

function get_objective_confidence_int(
    model_path::String; 
    exog::Exogenous_BatchCollection,
    feature_dict::Dict = Dict(), 
    folder_path::String = joinpath(@__DIR__, "..", "Figs\\debug\\"),
    safety::Bool = true,
    samples::Integer = 10,
    id::String = ""
    )

    get_objective_confidence_int(
        load_policy(model_path); 
        exog = exog,
        feature_dict = feature_dict, 
        folder_path = folder_path,
        safety = safety,
        samples = samples,
        id = id
    )
end


function test_safety_impact(model;hard::Bool = false, feature_dict::Dict = what_features(2))
    house = build_MCES(
            params = what_parameters(1), 
            simple_projection = true, 
            mem_safe = false,
            episode_length = 96,
            # ev = nothing, 
            cum_cost_grid = false,
            state_buffer_dict = feature_dict,
            reward_shape = 1,
            model_type = "NL",
    );

    ex = hard ? deepcopy(exog_cv_hard_91) : deepcopy(exog_cv_91)
    rng = rand(1:1000)

    @time h1 = run_dont_learn(
        house,  
        model;  
        exog = ex,
        rng = Xoshiro(rng)
    );

    compute_comprehensive_performance(h1, ex)

    house.simple_projection = false

    println("Running with Safety Projection...")
    @time h2 = run_dont_learn(
        house,  
        model;  
        exog = ex,
        rng = Xoshiro(rng)
    );

    compute_comprehensive_performance(h2, ex)
    return h1, h2
end


@info "Policies can be easily tested"