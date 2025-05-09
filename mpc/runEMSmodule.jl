include("EMS_pkg.jl")
includet("EMS_mods.jl")
include(joinpath(@__DIR__, "..", "src\\mces_core.jl"))
includet("convertEMSobjs.jl")
includet("EMS_transition.jl")

Random.seed!(10022025);


"""
    runEMS(steps; save::Bool = true, RLEnv::Bool = true)

Runs the Energy Management System (EMS) simulation with the MPC Expert Agent for the specified number of steps. Can use the RL Environtment for direct comparison.

# Arguments
- `steps`: Number of rolling horizon steps to simulate.
- `save::Bool=true`: Whether to save the simulation results to disk.
- `RLEnv::Bool=true`: If true, uses RL environment for simulation. If false, uses EMSModule own environment for the simulation.  

# Returns
- `rhDict`: Dictionary containing the concatenated results from the rolling horizon simulation.
- `EMSData`: Dictionary containing the data used in the EMS simulation.
- `rhCtrDict`: Dictionary containing the concatenated controller results from the rolling horizon simulation.

# Details
1. Initializes model settings with specific parameters (nEV=1, t0=1/4, etc.).
2. Executes the rolling horizon optimization with "day-ahead" type and the selected Environment.
3. Concatenates the results from individual time steps.
4. Measures and reports execution time.
5. When `save=true`, stores results in JSON and JLD2 formats in the "results" folder with timestamps.
"""
function runEMS(steps; save::Bool = true, RLEnv::Bool = true)
    s=modelSettings(nEV=1, t0=1/4,Tw=48-1/4, Δt=1/4, steps=steps, costWeights=[1, 1000, 0, 1000], season="summer",
                profType="yearly", loadType="GV", year=2022, cellID="SYNSANYO")

    start = time()
    results, EMSData, s, controllerRes, ddict, sdpdict = rollingHorizon(s; typeOpt="day-ahead", RLEnv = RLEnv)
    rhDict=concatResultsRH(results; typeOpt="day-ahead")
    rhCtrDict = concatResultsRH(controllerRes; typeOpt="day-ahead");
    finish = time()
    duration = round(Int,(finish-start)/60)
    println("$steps steps -> $duration min")

    !save && return rhDict, EMSData, rhCtrDict
    
    folder = joinpath(@__DIR__, "results")
    file_path = joinpath(folder, "MPC_$(Dates.format(now(), "yyyymmdd_HHmm"))_s$(s.steps)_results.json")
    open(file_path, "w") do f
        JSON3.pretty(f, results)
    end

    file_path_jld2 = joinpath(folder, "MPC_$(Dates.format(now(), "yyyymmdd_HHmm"))_s$(s.steps)_rhDict.jld2")
    jldsave(file_path_jld2, rhDict = rhDict)

    println("Saved Output in JSON and JLD2 formats. 
    File name -> $(file_path)")
    
    rhDict, EMSData, rhCtrDict
end



#############################################################################################################################
# Outputs

rhDict, EMSData, rhCtrDict = runEMS(90, save = true);
