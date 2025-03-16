"""
    simTransitionEnv!(results::Dict, data::Dict, s::modelSettings; typeOpt::String="day-ahead")

Simulates the environmental transition function Sₐ,ₜ₊₁ = Sₐ,ₜᴹ(Sₐ,ₜ, Pₐ,ₜ*, Wₐ,ₜ₊₁) for the energy management system, 
updating the state of various storage assets based on optimal control decisions and exogenous information.
This function was modified from the code provided by the EMSModule.  

# Arguments
- `results::Dict`: Dictionary containing simulation results, including optimal control decisions.
- `data::Dict`: Dictionary containing simulation data, including environmental data and asset information.
- `s::modelSettings`: Model settings containing simulation parameters.
- `typeOpt::String="day-ahead"`: Type of optimization (currently only supports "day-ahead").

# Returns
- `Tuple{Dict,Dict,Any}`: Updated results dictionary, updated data dictionary, and simulation hook (resulting from performing the simulation step within the RL Environment).

# Details
1. Extracts the optimal control decisions (Pbess, Pev, Phpe) from the results dictionary.
2. Collects exogenous information (loads, PV generation, EV driving patterns) from the data dictionary.
3. Creates an actor and collects the exogenous information for the transition.
4. Runs the transition in the RL environment (defined in src) to simulate system dynamics.
5. Updates the results dictionary with new state information (SoC, voltage, current) for all storage assets.
6. Updates the state information in the data dictionary for the next simulation step.
7. Simulates degradation for applicable storage assets.

# Throws
- `ArgumentError`: If `typeOpt` is not "day-ahead".
- `KeyError`: If required keys are missing from `results` or `data`.

"""
function simTransitionEnv!(results::Dict, data::Dict, s::modelSettings; typeOpt::String="day-ahead")
    # This function simulates the transition function Sₐ,ₜ₊₁ = Sₐ,ₜᴹ(Sₐ,ₜ , Pₐ,ₜ* , Wₐ,ₜ₊₁).
    # Sₐ,ₜ: state in timestep t
    # Pₐ,ₜ*: optimal setpoint in timestep t
    # Wₐ,ₜ₊₁: exogenous information in timestep t+1
    if typeOpt != "day-ahead"
        throw(ArgumentError("typeOpt must be 'day-ahead'. Got: $typeOpt"))
    end

    
    # results["t"] -> (Day 3) 173700.0, 174600.0, 175500.0, 176400.0 
    t = results["t"];
    t0 = t[1]
    Δt = t[2] - t[1] # -> 900 seconds
    day_in_sec = 24*3600
    shift = Int(day_in_sec/Δt) - 1 # shift -> 96 - 1
    it0 = round(Int, (t0/Δt)) # For Day 3 - > it0 = 193
    itend = it0 + shift; # For Day 3 -> itend = 288
    day = ceil(Int, it0/(24*3600/Δt)); # For Day 3 -> ceil(193/96) = 3
    
    # First, we get the optimal decisions from our policy function.
    Pbess = results["Pbess"][1:shift+1]
    nEV = 1 # Only one EV implemented in Environment currently. 
    Pev = [results["Pev[$n]"][1:shift+1] for n ∈ 1:nEV]
    haskey(results, "Phpe") ? Phpe = results["Phpe"][1:shift+1] : Phpe = zeros(shift+1)

    actor = Replay32(Pev[1], Pbess, Phpe)
    
    # Now the Exogenous Data needed for the transition
    Ple = copy(data["grid"].loadE[it0:itend])
    Plt = copy(data["grid"].loadTh[it0:itend])
    PpvMPPT = copy(data["SPV"].MPPTData[it0:itend])
    drive = data["EV"][1].driveInfo.Pdrive[day]
    γ_cont = data["EV"][1].driveInfo.γ[it0:itend]

    Exg_Info = Exogenous_BatchCollection(
        load_e = Ple,
        load_th = Plt,
        pv = Float32.(PpvMPPT),
        λ_buy = zeros(Float32, shift + 1), # They are zeros because price data will only affect rewards, and MPC is indifferent to those. 
        λ_sell = zeros(Float32,shift + 1),
        p_drive = fill(drive, shift + 1),
        γ_ev = γ_cont,
        last_timestep = shift + 1
    )
    
    env = data["Env"]
    env.episode_length = shift + 1

    hook = run_TransitionEnv!(env, actor, exog = Exg_Info) # env is updated in place
    en = hook.energy
    results["Ptess"] = Float64.(en["p_tess"][1:shift+1])
    results["Pg"] = Float64.(en["grid"][1:shift+1])

    # Update BESS
    results["SoCbess"] = Float64.(en["soc_bess"][1:shift+1])
    data["BESS"].GenInfo.SoC0 = Float64(en["soc_bess"][shift+1])
    results["vtbess"] = Float64.(en["v_bess"][1:shift+1])
    results["ibess"] = Float64.(en["i_bess"][1:shift+1])

    # Update EV Battery
    results["SoCev[1]"] = Float64.(en["soc_ev"][1:shift+1])
    data["EV"][1].carBatteryPack.GenInfo.SoC0 = Float64(en["soc_ev"][shift+1])
    results["vtev[1]"] = Float64.(en["v_ev"][1:shift+1])
    results["iev[1]"] = Float64.(en["i_ev"][1:shift+1])

    # Update TESS
    results["SoCtess"] = Float64.(en["soc_tess"][1:shift+1])
    data["TESS"].SoC0 = Float64(en["soc_tess"][shift+1])
       
  
    for sa ∈ ["BESS", "EV", "TESS"]
        if isa(data[sa], Vector)
            for n ∈ 1:length(data[sa])
                # pick the right key
                if typeof(data[sa][n]) == BESSData
                    key="bess[$n]"
                    stgAsset = data[sa][n];
                elseif typeof(data[sa][n]) == EVData
                    key="evTot[$n]"
                    stgAsset = data[sa][n].carBatteryPack;
                elseif typeof(data[sa][n]) == TESSData
                    key="tess[$n]"
                    stgAsset = data[sa][n];
                end
                # Sₛₐ,ₜ₊₁ = Sₛₐ,ₜᴹ(Sₛₐ,ₜ , Pₛₐ,ₜ* , Wₛₐ,ₜ₊₁)
                simulate_degrading!(stgAsset, results, key; typeOpt=typeOpt)
            end
        else
            # pick the right key
            if typeof(data[sa]) == BESSData
                key="bess"
                stgAsset = data[sa];
            elseif typeof(data[sa]) == EVData
                key="evTot"
                stgAsset = data[sa].carBatteryPack;
            elseif typeof(data[sa]) == TESSData
                key="tess"
                stgAsset = data[sa];
            end
            # Sₛₐ,ₜ₊₁ = Sₛₐ,ₜᴹ(Sₛₐ,ₜ , Pₛₐ,ₜ* , Wₛₐ,ₜ₊₁)
            simulate_degrading!(stgAsset, results, key; typeOpt=typeOpt)
            # adjust power setpoints if the storage has been depleted

        end
    end
    return results, data, hook
end

function simulate_degrading!(stgAsset::Union{BESSData,TESSData}, results::Dict, key::String; typeOpt::String="day-ahead")
    isa(stgAsset, TESSData) && return nothing   # No degradation for TESS
         
    perfModel = perfModel_matching(stgAsset)
    if occursin("evTot", key) 
        key = replace(key, "evTot" => "ev")
    end
        
    simulate_storage_asset_deg!(stgAsset, perfModel, results, key; typeOpt=typeOpt)
    nothing
end


"""
    run_TransitionEnv!(house::MCES_Env, policy; exog::Exogenous_BatchCollection)::MCES_Hook

Run a single episode transition of the EMS module.

# Arguments
- `house::MCES_Env`: The MCES environment to run the transition in.
- `policy`: The policy to use for decision making.
- `exog::Exogenous_BatchCollection`: Exogenous data for the transition.

# Returns
- An `MCES_Hook` containing data collected during the transition.

# Details
1. Creates a new hook for data collection.
2. Sets up an agent with the provided policy.
3. Configures a stop condition for one episode.
4. Resets the exogenous data.
5. Calls `run_free` to execute the transition without learning.
6. Returns the hook with collected data.
"""
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

@info "Implemented Environment Transition"