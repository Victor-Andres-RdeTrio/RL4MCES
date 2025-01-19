import EMSmodule: spv!, st!, pei!, gridThermal!, costFunction!, tess!, @unpack_Generic
# import EMSmodule: add_battPerf, ev!, 

"""
    build_data(; nEV::Int64 = 1, season::String = "winter", profType::String = "yearly",
               loadType::String = "GV", year::Int64 = 2022, fs::Int64 = 4, cellID::String = "SYNSANYO")

Build the data for the optimization problem. The output is a dictionary that contains the models for all the different devices in the Multi-carrier Energy System.

# Arguments
- `nEV::Int64`: Number of EVs in the system. Default is 1.
- `season::String`: "summer" or "winter". Default is "winter".
- `profType::String`: "weekly" or "daily". Default is "yearly".
- `loadType::String`: "GV", "mffbas", or "base_models". Default is "GV".
- `year::Int64`: Year for the profile. Default is 2022.
- `fs::Int64`: Samples per hour. Default is 4.
- `cellID::String`: Cell ID for the battery packs. Default is "SYNSANYO".

# Returns
- `Dict`: Dictionary containing models for different devices in the system as well as exogenous information needed for the transition function.
"""
function build_data(; nEV::Int64 = 1, 
                    season::String = "winter", 
                    profType::String = "yearly", 
                    loadType::String = "GV", 
                    year::Int64 = 2022,
                    fs::Int64 = 4, 
                    cellID::String = "SYNSANYO")
    
    @assert profType ∈ ["yearly"] "Invalid profile type"

    # Exogenous Information
    test_data = joinpath(@__DIR__, "..","data\\test.jld2")
    @load test_data pv_test prices_test loadE_test loadTh_test γ_test tDep_test tArr_test pdrive_test
    pdrive_data = joinpath(@__DIR__, "..","data\\pdrive_MPC.jld2")
    @load pdrive_data pdrive_MPC # Currently Not Used

    spvModel = SPVData(MPPTData = pv_test)
    priceData = prices_test
    loadE = loadE_test
    loadTh = loadTh_test
    γ = γ_test
    tDep = tDep_test
    tArr = tArr_test
    Pdrive_data = Float64.(pdrive_test[1:96:end]) # Get one value per day. 
    # Pdrive = Array_and_Pointer(array = Pdrive_data, pointer = Int32(1))

    # BESS model parameters
    cellID = "SYNSANYO"
    Q0 = 5.2; η = 0.99
    PowerLim = [-17, 17] ; P0 = 0
    SoCLim = [0.205, 0.945] ; SoC0 = 0.5
    ηC = 1
    SoHQ = 1
    SoHR0 = 28e-3
    Ns = 100 ; Np = 10
    ocv_params = OCVlinearPerfParams(ocvLine = [3.525, 0.625])
    vLim = [2.8, 4.2]
    aging_params = empAgingParams()
    initVal = 494.246
    termCond = 6.0
    gen_params = Generic(PowerLim, P0, SoCLim, SoC0, termCond, Q0, SoHQ, SoHR0, Np, Ns, η, ocv_params, vLim, ηC, initVal)
    perf_params = bucketPerfParams()
    battModel = BESSData(gen_params, perf_params, aging_params, cellID)

    # EV model parameters (many shared with BESS)
    PowerLim_EV = [-12.5, 12.5]
    # PowerLim_EV = [-0.1, 0.1]
    Ns_EV = 100 ; Np_EV = 25
    P0_EV = [0, 0]
    Q0_EV = Q0 .* ones(nEV)
    SoC0_EV = [0.5, 0.5]
    SoCLim_EV = [0.205, 0.945]
    # SoCLim_EV = [0.0, 1.0]
    termCond_EV = 6.0
    gen_params = [Generic(PowerLim_EV, P0_EV[n], SoCLim_EV, SoC0_EV[n], termCond_EV, Q0_EV[n], SoHQ, SoHR0, Np_EV, Ns_EV, η, ocv_params, vLim, ηC, initVal) for n ∈ 1:nEV]
    perf_params = [bucketPerfParams() for n ∈ 1:nEV]
    batteryPack = [BESSData(gen_params[n], perf_params[n], aging_params, cellID) for n ∈ 1:nEV]

    av = [(γ, tDep, tArr) for _ ∈ 1:nEV]
    refSoC = [0.85, 0.85]
    drive_info = [driveData(Pdrive_data, refSoC[n], av[n][1], av[n][2], av[n][3]) for n in 1:nEV]
    evModel = [EVData(batteryPack[n], drive_info[n]) for n ∈ 1:nEV]

    stModel = ElectroThermData(
        RatedPower = 2.7, 
        capex = 1500, 
        η = 0.6
    )

    tessModel = TESSData(
        Q = 200, 
        PowerLim = [-5, 5],
        SoCLim = [0.105, 0.945], 
        SoC0 = 0.4, # Careful 
        η = 0.95        
    ) 

    peiModel = peiData(RatedPower = 17)
    
    gridModel = gridData(
        PowerLim = [-16.9, 16.9], 
        η = 0.9, 
        λ = priceData, 
        loadE = loadE, 
        loadTh = loadTh
    )
    
    hpModel = ElectroThermData(
        RatedPower = 4, 
        capex = 500, 
        η = 4.5
    )

    
    env = build_MCES(
        tess = convert(TESS, tessModel),
        bess = convert(Battery, battModel),
        ev = convert(EV, evModel[1]),
        hp = convert(HP, hpModel),
        stc = convert(STC, stModel),
        pei = convert(PEI, peiModel, 0.9),
        Δt = 900, # seconds
        mem_safe = false,
        simple_projection = true,
        episode_length = 96,
        reward_shape = 1
    )

    Dict("SPV" => spvModel,
        "BESS" => battModel,
        "EV" => evModel,
        "ST" => stModel,
        "HP" => hpModel,
        "TESS" => tessModel,
        "grid" => gridModel,
        "PEI" => peiModel,
        # "Pdrive" => Pdrive,
        "Env" => env,
        "Noise" => false
    )
end

function solvePolicies(optimizer, # model optimizer
    sets::modelSettings, # number of EVs, discrete time supports, etc.
    data::Dict, # information for the model (mainly parameters), these are the devices (EV, BESS, PV, etc.), the costs (interests, capex, etc) and the revenues
    preRes::Dict = Dict(), # previous results to warm start the model
    )
    # Sets
    tend=sets.dTime[end]
    t0=sets.dTime[1]; # initial time, can´t divide by 0
    model = InfiniteModel(optimizer) # create model
    noise = data["Noise"]

    # If KNITRO is not working, you can use Juniper with Ipopt.
    # Example:
    # ipopt = optimizer_with_attributes(Ipopt.Optimizer, "print_level"=>0)
    # optimizer = optimizer_with_attributes(Juniper.Optimizer, "nl_solver"=>ipopt)
    # optimizer would be fed to solvePolicies as the first argument.
 

    # Optimizer attributes
    # KNITRO attributes
    set_optimizer_attributes(model,
                            # "tuner"=>1,
                            "scale"=>1,
                            "presolve"=>1,
                            # "tuner_file"=>tunerpath,
                            # "mip_maxnodes" => 3500,
                            "convex"=>1,
                            "opttol"=>1e-3,
                            "feastol"=>1e-3,
                            "mip_opt_gap_rel"=>1e-3,
                            "mip_multistart"=>1,
                            # options
                            "hessopt"=>1,
                            "hessian_no_f"=>1,
                            "mip_method" =>	1,
                            "mip_nodealg" => 1,
                            "mip_selectrule" =>	2,
                            "mip_branchrule" =>	2,
                            "mip_heuristic_strategy" =>	2,
                            "mip_gomory" =>	0,
                            "mip_heuristic_diving" => 0,
                            "mip_heuristic_feaspump" =>1,
                            # "mip_heuristic_localsearch" =>	1,
                            "mip_heuristic_mpec" => 1
                            )
    # set_time_limit_sec(model, 6*60*60);
    set_time_limit_sec(model, 10*60);

    # define cont t
    @infinite_parameter(model, t ∈ [t0, tend], supports = collect(sets.dTime),
                        derivative_method = FiniteDifference(Backward())
    )
    
    # Add devices
    
    # Electrical
    spv!(model, data, noise); # ok
    bess!(model, sets, data); # ok
    # steal the model here, use it in getPdrive. 
    ev!(model, sets, data);
    
    # Thermal
    st!(model, data, noise);
    heatpump!(model, data);
    tess!(model, data);
    gridThermal!(model, sets, data, noise); # thermal balance
    
    # grids and balances
    gridConn!(model, data); # grid connection
    pei!(model, sets, data, noise); # electrical balance
    
    # Add objective function
    costFunction!(model, sets, data, noise)
    # Warm starting.
    # check if the previous solution is not empty
    !isempty(preRes) ? set_warm_start(model, preRes) : nothing;

    # Solve model
    optimize!(model)
    results = getResults(model)
    # println(latex_formulation(model)) # To get a the formulation in markdown. 
    # ddict, sdpdict = getDuals(model)
    MOI.empty!(InfiniteOpt.backend(model))
    return results;
end;

function handleInfeasible!(results, ctrlRes, ts, s; typeOpt::String="day-ahead")
    # this function is used to handle infeasible solutions
    # it is called in the solvePolicies() function
   
    typeOpt == "day-ahead" ? shift = Int(ceil(length(results[ts]["t"])/2))-1 : nothing;

    println("\r Checking the status of the last set of results")
    if results[ts][:"status"] == LOCALLY_INFEASIBLE
    # discard the last set of results and implement the second step of the previous solution at ts-1
        println("\r Infeasible solution at step $ts, implementing the second step of the previous solution at step $(ts-1)")
        results[ts]=copy(ctrlRes[ts-1]);
        if typeOpt == "day-ahead"
            for k in keys(results[ts])
                # Check if the key requires special handling and skip the "status" key
                if k == :"status"
                    continue
                elseif k == :"γ_cont"
                    if s.nEV !== 1
                        results[ts][k] = [results[ts][k][n][(shift+1):end] for n ∈ 1:s.nEV];
                    else
                        results[ts][k] = results[ts][k][(shift+1):end];
                    end
                else
                    results[ts][k] = results[ts][k][(shift+1):end];
                end
            end
        else
            error("typeOpt should be day-ahead")
        end
        # update the controller setpoints so that for next iteration this stored somewhere
        ctrlRes[ts]=copy(results[ts]);
    else
        println("\r Locally feasible solution at step $ts")
    end
    return results, ctrlRes;
end

function rollingHorizon(s;
    typeOpt::String="day-ahead", RLEnv = true) # MPC or day-ahead
    ## Rolling Horizon Simulation.
    # This function simulates the EMS for a given number of steps. 
    # The EMS is initialized at time t0 and then it is solved for a time window of Tw hours.
    # Then, the EMS is solved again for the next Tw hours, but this time the initial conditions are the ones obtained from the previous solution.
    # This process is repeated until the number of steps is reached.
    # The output of this function are:
    # - results::Vector{Dict}, where each dictionary contains the results of the EMS for each time window.
    # - rhDict::Dict, which is a dictionary with the concatenated results of the EMS for each time window.
    @assert typeOpt ∈ ["MPC", "day-ahead"] "typeOpt must be either MPC or DA"
    if typeOpt == "day-ahead"
       @assert (s.Tw + s.Δt) % 24 == 0 "Tw must be a multiple of fs times 24"
    end

    steps = s.steps; Tw = s.Tw; Δt = s.Δt;
    Dt = s.dTime[1]:(s.dTime[2]-s.dTime[1]):s.dTime[end]; # 900.0:900.0:172800.0 (Not exactly an Array)
    # allocate memory
    results=Vector{Dict}(undef, steps);
    ddict=Vector{Dict}(undef, steps); sdpdict=Vector{Dict}(undef, steps);
    controllerRes=Vector{Dict}(undef, steps);
    # Initialize an array to store the times
    times = Vector{Float64}(undef, steps)

    EMSData=build_data(;nEV = s.nEV,
                        # season = s.season,
                        profType = s.profType,
                        loadType = s.loadType,
                        year = s.year,
                        cellID = s.cellID,
                        );

    #=
    # Modify the TESS init condition
    EMSData[:"TESS"].SoC0=0.4;
    # Modify the BESS perf. model
    EMSData[:"BESS"].GenInfo.SoC0=0.6;
    # EMSData[:"BESS"].GenInfo.SoCLim[1]=0.15; # for CPBDeg
    EMSData[:"BESS"].PerfParameters = bucketPerfParams();
    # Modify the EV model
    [EMSData[:"EV"][n].carBatteryPack.PerfParameters = bucketPerfParams() for n ∈ 1:s.nEV];
    =#

    # Initial solution
    times[1] = @elapsed begin
        # results[1], ddict[1], sdpdict[1] = solvePolicies(KNITRO.Optimizer, s, EMSData, Dict());
        results[1] = solvePolicies(KNITRO.Optimizer, s, EMSData, Dict());
        controllerRes[1] = copy(results[1]);
        # Check the status of the last set of results
        handleInfeasible!(results, controllerRes, 1, s; typeOpt=typeOpt);
        if RLEnv
            simTransitionEnv!(results[1], EMSData, s; typeOpt = typeOpt)
        else
            simTransitionFun!(results[1], EMSData, s; typeOpt = typeOpt)
        end
        # move time window
        typeOpt == "MPC" ? Dt = Dt .+ Δt*3600.0 : Dt = Dt.+ 24*3600.0;
        # update_forecasts(EMSData, Dt)
        s.dTime=collect(Dt);
    end

    run_number = rand(1:100)
    folder = joinpath(@__DIR__, "results_backup")
    
    for ts in 2:steps
        try
            times[ts] = @elapsed begin
                # build+solve model
                # results[ts]=solvePolicies(KNITRO.Optimizer, s, EMSData, controllerRes[ts-1]); # Looks like a Warm Start
                # results[ts], ddict[ts], sdpdict[ts] =solvePolicies(KNITRO.Optimizer, s, EMSData, Dict());
                results[ts] =solvePolicies(KNITRO.Optimizer, s, EMSData, Dict());
                controllerRes[ts] = copy(results[ts]);
                # Check the status of the last set of results
                handleInfeasible!(results, controllerRes, ts, s; typeOpt=typeOpt);
                # EMSData=update_measurements(results[ts], s, EMSData; typeOpt=typeOpt);
                if RLEnv
                    _ ,_ , hook = simTransitionEnv!(results[ts], EMSData, s; typeOpt = typeOpt)
                    #### Logging
                    save_MCES_Hook(joinpath(folder, "Run_$(Dates.format(now(), "yyyymmdd"))_ID$(run_number)_s$(ts)_of_$(steps)_hook.jld2"), hook)
                else
                    simTransitionFun!(results[ts], EMSData, s; typeOpt = typeOpt)
                end
                # move time window
                typeOpt == "MPC" ? Dt = Dt .+ Δt*3600.0 : Dt = Dt.+ 24*3600.0;
                # update_forecasts(EMSData, Dt)
                s.dTime=collect(Dt);
            end
            # Base.GC.gc(); # collect garbage
            println("\r Step $ts out of $steps done!")
        catch e
            println("Error at step $ts")
            showerror(stdout, e, catch_backtrace())
            return results, EMSData,s, controllerRes;
        end
        #### Logging
        file_path = joinpath(folder, "Run_$(Dates.format(now(), "yyyymmdd"))_ID$(run_number)_s$(ts)_of_$(steps).json")
        open(file_path, "w") do f
            JSON3.pretty(f, results)
        end
        # rhDict = concatResultsRH(results; typeOpt="day-ahead");
        jld2_path = joinpath(folder, "Run_$(Dates.format(now(), "yyyymmdd"))_ID$(run_number)_s$(ts)_of_$(steps).jld2")
        jldsave(jld2_path, results = results)
    end
    [results[ts]["compTime"]=times[ts] for ts in 1:steps]

    return results, EMSData,s, controllerRes, ddict, sdpdict;
end

# Eliminating the Noise from functions
function spv!(model::InfiniteModel, data::Dict, noise::Bool = true) # solar pv panel
    # MPPT measurement
    t = model[:t];
    Dt = supports(t);
    t0 = supports(t)[1]; Δt=supports(t)[2]-supports(t)[1];
    # make new time window
    it0 = round(Int,(t0/Δt));
    itend = it0+length(supports(t))-1;
    
    MPPTmeas = data["SPV"].MPPTData[it0:itend];
    # simulated forecast
    if noise 
        MPPTnoisy = zeros(size(MPPTmeas));
        MPPTnoisy[MPPTmeas .> 0] = [rand(Uniform(0.5*MPPTmeas[tt],1.1*MPPTmeas[tt])) for tt ∈ findall(MPPTmeas .> 0)]
    else
        MPPTnoisy = MPPTmeas
    end
    # Add forecast
    @parameter_function(model, PpvMPPT == (t) -> MPPTnoisy[zero_order_time_index(Dt, t)])

    return model;
end;

function st!(model::InfiniteModel, data::Dict, noise::Bool = true) # solar thermal 
    # The thermal pv/heatpipes are only the converted measurement of the irradiance.
    t = model[:t]; 
    Dt = supports(t);
    t0 = supports(t)[1]; Δt = supports(t)[2]-supports(t)[1];
    # make new time window
    it0 = round(Int,(t0/Δt));
    itend = it0+length(supports(t))-1;
    # Extract params
    # PstRated = data["ST"].RatedPower; # rated power of the panels
    ηST = data["ST"].η; # conversion factor from Electric PV to thermal

    # Extract data
    MPPTmeas = data["SPV"].MPPTData[it0:itend];
    # simulated forecast
    if noise 
        MPPTnoisy = zeros(size(MPPTmeas));
        MPPTnoisy[MPPTmeas .> 0] = [rand(Uniform(0.5*MPPTmeas[tt],1.1*MPPTmeas[tt])) for tt ∈ findall(MPPTmeas .> 0)]
    else
        MPPTnoisy = MPPTmeas
    end
    
    @parameter_function(model, Pst == (t) -> ηST*MPPTnoisy[zero_order_time_index(Dt, t)])

    return model;
end;

function gridThermal!(model::InfiniteModel, sets::modelSettings, data::Dict, noise::Bool = true) # thermal grid
    # Load data
    Dt = sets.dTime;
    t = model[:t]; 
    t0=supports(t)[1]; Δt=supports(t)[2]-supports(t)[1];
    # make new time window
    it0 = round(Int,(t0/Δt));
    itend = it0+length(Dt)-1;
    # make InfiniteOpt compatible Plt(t) work for arbitrary times
    loadTh = data["grid"].loadTh[it0:itend];
    # white noise to simulate a forecast
    εₗᵗʰ= randn(Int(length(loadTh)*Δt/3600)) .* 0.05 # 50W
    εₗᵗʰ = repeat(εₗᵗʰ, inner = Int(3600/Δt))
    loadTh = loadTh .+ (εₗᵗʰ .* noise); # add noise
    @parameter_function(model, Plt == (t) -> loadTh[zero_order_time_index(Dt, t)])
    
    # When there´s no FCR then there´s no decision variable Ppve hence Pst can just be a @parameter_function.
    # Extract params
    ηHP=data["HP"].η; # conversion factor from Electric to thermal heat pump
    # Thermal Power balance
    model[:thBalance]=@constraint(model, model[:Pst]+model[:Phpe].*ηHP+model[:Ptess] .==  Plt);
    return model;
end;

function pei!(model::InfiniteModel, sets::modelSettings, data::Dict, noise::Bool = true) # power electronic interface
    # Extract data
    Dt = sets.dTime; nEV=sets.nEV;
    t = model[:t];
    t0=supports(t)[1]; Δt=supports(t)[2]-supports(t)[1];
    # make new time window
    it0 = round(Int,(t0/Δt));
    itend = it0+length(Dt)-1;
    # Load data
    loadElec = data["grid"].loadE[it0:itend];
    # white noise to simulate a forecast
    εₗᵉ= randn(Int(length(loadElec)*Δt/3600)) .* 0.2 # 200W
    εₗᵉ = repeat(εₗᵉ, inner = Int(3600/Δt))
    loadElec = loadElec .+ (εₗᵉ * noise); # add noise
    # loadElec can't be negative
    loadElec[loadElec .< 0] .= 0
    @parameter_function(model, Ple == (t) -> loadElec[zero_order_time_index(Dt, t)])
    
    # Power balance DC busbar
    # If we have EVs
    if any(name.(all_variables(model)) .== "Pev[1]")
        Pev_sum = nEV != 1 ? sum(model[:γf][n].*model[:Pev][n] for n in 1:nEV) : sum(model[:γf].*model[:Pev][n] for n in 1:nEV)
    else
        Pev_sum = 0
    end

    # If we have HP
    Phpe = any(name.(all_variables(model)) .== "Phpe") ? model[:Phpe] : 0

    model[:powerBalance]=@constraint(model, model[:PpvMPPT] + model[:Pbess] + Pev_sum + model[:Pg] .== Ple + Phpe)

    return model;
end;

function costFunction!(model, sets::modelSettings, data::Dict, noise::Bool = true) # Objective function
    W = sets.costWeights;
    Dt = sets.dTime;
    t = model[:t]; # in seconds
    t0=supports(t)[1]; Δt=supports(t)[2]-supports(t)[1];
    # make new time window
    it0 = round(Int,t0/Δt);
    itend = it0+length(Dt)-1;

    priceBuy=hcat(data["grid"].λ[it0:itend, 1].*1e-3/3600) # convert from €/MWh to €/kWs
    priceSell=hcat(data["grid"].λ[it0:itend, 2].*1e-3/3600) # convert from €/MWh to €/kWs
    
    # simulated forecast
    ελ = randn(Int(length(priceBuy)*Δt/3600)) .* 20*1e-3/3600 # 20 €/MWh noise
    ελ = repeat(ελ, inner = Int(3600/Δt))
    priceBuy = priceBuy .+ (ελ .* noise); # add noise
    priceSell = priceSell .+ (ελ .* noise); # add noise
    @parameter_function(model, λbuy == (t) -> priceBuy[zero_order_time_index(Dt, t)])
    @parameter_function(model, λsell == (t) -> priceSell[zero_order_time_index(Dt, t)])

    # Grid costs
    Wgrid = W[1]; # regularization factor for grid cost. max(λ)*max(P)
    cgrid = Wgrid .* (model[:PgPos]*λbuy + model[:PgNeg]*λsell); # PgNeg and PgPos in kW. 
    
    # Aging costs CHECK
    clossbess = 1.2; # cost of lost capacity EUR/Ah
    Wloss=W[3]; # regularization factor for lost capacity
    if Wloss != 0.
        # cQloss = Wloss != 0 ? ( Wloss*(model[:ilossbess]+sum(model[:ilossev][n] for n ∈ 1:sets.nEV))/3600) : 0.0;
        lossBESS = data["BESS"].GenInfo.Ns * data["BESS"].GenInfo.Np * model[:ilossbess];
        if any(name.(all_variables(model)) .== "ilossev[1]")
            lossEV = [data["EV"][n].carBatteryPack.GenInfo.Ns * 
                    data["EV"][n].carBatteryPack.GenInfo.Np *
                    model[:ilossev][n] for n ∈ 1:sets.nEV]
        else
            lossEV = zeros(sets.nEV);
        end
        cQloss = ∫(( Wloss*(lossBESS+sum(lossEV[n] for n ∈ 1:sets.nEV))/3600),t);
    else
        cQloss = 0.0;
    end

    # Penalty/soft constraint for TESS overcharging
    Wlims = W[4]; # penalty for TESS overcharging
    SoCtess = model[:SoCtess]; SoCtessMax = data["TESS"].SoCLim[2];
    # SoCbess = model[:SoCbess]; SoCbessMin = data["BESS"].GenInfo.SoCLim[1];
    @variable(model, auxTess ≥ 0., Infinite(t));
    @constraint(model, auxTess ≥ SoCtess - SoCtessMax);
    # @variable(model, auxBess ≥ 0, Infinite(t));
    # @constraint(model, auxBess ≥ SoCbessMin - SoCbess);

    # Define penalty for not charging
    WSoCDep = W[2]
    pDep = (any(name.(all_variables(model)) .== "Pev[1]") ?
            WSoCDep*sum(model[:ϵSoC][n]^2 for n ∈ eachindex(model[:ϵSoC])) : 0);

    # Define objective function
    if any(name.(all_variables(model)) .== "ilossbess")
        @objective(model, Min, ∫(cgrid,t)/sum(Dt) + pDep + Wlims*∫(auxTess,t)/sum(Dt) +
            # clossbess*∫(iloss,t)/sum(Dt))
            clossbess*cQloss/sum(Dt))
        # @objective(model, Max, -∫(cgrid,t)/sum(Dt) - pDep - Wlims*∫(auxTess + auxBess,t)/sum(Dt) -
        #     clossbess*cQloss/sum(Dt))
    else
        @objective(model, Min, ∫(cgrid,t)/sum(Dt) + pDep + Wlims*∫(auxTess,t)/sum(Dt))
        # Cgrid in [Euros]
    end 
    return model;
end;

function concat_broken_rh(vec::Vector{Dict}) 
    for i in length(vec):-1:1
        if !isassigned(vec, i)
            continue
        else
            return concatResultsRH(vec[1:i]; typeOpt="day-ahead");
        end
    end
end

# function tess!(model::InfiniteModel, data::Dict) # thermal energy storage buffer
#     # Thermal Energy Storage System
#     t = model[:t];
#     t0 = supports(t)[1];
#     # Bucket model
#     # Extract data
#     Qtess = data["TESS"].Q; # Capacity [kWh]
#     PtessMin = data["TESS"].PowerLim[1]; # Min power [kW]
#     PtessMax = data["TESS"].PowerLim[2]; # Max power [kW]
#     # Ptess0 = data["TESS"].P0; # Initial Power [p.u.]
#     SoCtessMin = data["TESS"].SoCLim[1]; # Min State of Charge [p.u.]
#     SoCtessMax = data["TESS"].SoCLim[2]; # Max State of Charge [p.u.]
#     SoCtess0 = data["TESS"].SoC0; # Initial State of Charge [p.u.]
#     ηtess = data["TESS"].η; # thermal efficiency
       
#     # Add variables
#     @variables(model, begin
#         # SoCtessMin ≤ SoCtess ≤ SoCtessMax, Infinite(t) # State of Charge
#         SoCtess >= SoCtessMin , Infinite(t) # State of Charge
#         Ptess, Infinite(t) # Thermal power
#         bPtess, Infinite(t), Bin # Binary variable for TESS power
#         0 ≤ PtessPos, Infinite(t) # Ptess^+ out power
#         PtessNeg ≤ 0, Infinite(t) # Ptess^- in power
#     end);
#     # Dummy variables for bidirectional power flow, ensuring only export or import
#     @constraints(model, begin
#         bPtess*PtessMin ≤ PtessNeg
#         PtessNeg + PtessPos .== Ptess
#         # PtessNeg * (1/ηtess) + PtessPos * ηtess .== Ptess
#         PtessPos ≤ (1-bPtess)*PtessMax
#     end);
    
#     # Initial conditions
#     @constraint(model, SoCtess(t0) .== SoCtess0)
#     # @constraint(model, Ptess(t0).== Ptess0)

#     # Bucket model
#     @constraint(model, ∂.(SoCtess, t) .== -ηtess*Ptess/Qtess/3600);
#     # @constraints(model, begin
#     #     SoCtess(t0+(Tw+Δt)/2) .- SoCtess(t0) .≤ 0.05
#     #     SoCtess(t0+(Tw+Δt)/2) + 0.05 .≤ SoCtessMax
#     # end);
#     return model;
# end;

















@info "Added EMS Mods"