### Needed functions for import
using JLD2
using Serialization
using Distributions
using CSV
using DataFrames
using Interpolations
using Random
using Plots

function test_and_train(arr)
    train = Float64[]; test = Float64[];
    @assert length(arr) == 35040 "Input length for test_and_train is not 35040"
    arr = reshape(arr, 96,365)
    for i in 1:size(arr)[2]
        if rem(i, 4) == 0
            push!(test, arr[:,i]...) 
        else
            push!(train, arr[:,i]...)
        end
    end
    # println("Input has $(length(arr)) elements")
    println("Train has $(length(train)) elements")
    println("Test has $(length(test)) elements")
    test, train
end

function expand_train(arr, rng = Xoshiro(42))
    ex = Float64[] 
    for i in Iterators.partition(arr,96*3)
        if length(i) == 288
            res = reshape(i,96,3)
            m = mean(res, dims = 2)
            noise = randn(rng, 96, 1).* (m / 6)
            noisy_m = m .+ noise
            push!(ex, i...)
            push!(ex, noisy_m...)
        else
            push!(ex, i...)
            println("Just pushed at the end")
        end
        if length(ex) > 35040
            error("Train Expand Went above 35040 elements")
        end
    end
    clamp.(ex, minimum(arr), maximum(arr))
end
   
function test_train_expand(arr; first = false, rng = Xoshiro(42))
    while length(arr) > 35040 
        if first
            popfirst!(arr) # get it to 35040 elements
        else
            pop!(arr)
        end
    end
    test, train = test_and_train(arr)
    test, train, expand_train(train, rng)
end

function cv_train(arr, rng = Xoshiro(42); divergence = 1/3)
    cv = Float32[] 
    for i in Iterators.partition(arr, 96*3)
        res = reshape(i,96,:)
        size(res)[2] != 3 && break
        m = mean(res, dims = 2)
        noise = randn(rng, 96, 1).* (divergence * m)
        noisy_m = m .+ noise
        push!(cv, noisy_m...)
    end
    println("CV has $(length(cv)) elements.")
    clamp.(cv, minimum(arr), maximum(arr))
end

function my_rejCriteriaEnergy(μDrive, σDrive, tDep, tArr, Tconn, Ns, Np, vmax, Q0, chgPmax; rng::AbstractRNG = Xoshiro(42))
    Pdrive = rand(rng, truncated(Normal(μDrive, σDrive); lower = 0.01), length(tDep));
    Ereq = Pdrive .* (tArr .- tDep) # energy required for each session [kWh]
    Emax = chgPmax .* Tconn # max energy that can be charged [kWh]
    Eev = Ns .* Np .* vmax .* Q0 .* 1e-3 * 0.6 # 60% energy of the EV battery [kWh]
    # Check if the energy required is less than the max energy that can be charged
    # if not adjust the demanded Pdrive
    for i ∈ eachindex(Ereq)
        if Ereq[i] > Emax[i] || Ereq[i] > Eev
            # pick the lowest bound between charger and EV battery
            println("Energy required for session $i is too high")
            println("Adjusting power...")
            if Emax[i] < Eev
                # adjust the power to the max that can be charged
                println("From $(Ereq[i]) kWh to $(Emax[i]) kWh")
                Pdrive[i] = Emax[i] / (tArr[i] - tDep[i])

            else
                println("From $(Ereq[i]) kWh to $(Eev) kWh")
                Pdrive[i] = Eev / (tArr[i] - tDep[i])
            end
        end
    end
    # return Pdrive, Ereq, Emax
    return Pdrive
end

function myavailabilityEV(
    ns, # number of samples
    fs = 4, # samples per hour
    rng::AbstractRNG = Xoshiro(42),
    μDrive::Float64 = 3.5, # mean of Pdrive [kW]
    σDrive::Float64 = 1.5, # standard dev. of Pdrive [kW]
    Ns::Int64 = 100, # number of series cells
    Np::Int64 = 25, # number of parallel Branches
    vmax::Float64 = 4.2, # max voltage [V] (possibly problematic, maybe use vrated or voltage at socdep)
    Q0::Float64 = 5.2, # initial capacity [Ah]
    chgPmax::Float64 = 12.5, # EV charger limits [kW]
    )
        
    ndays=Int(floor(ns/fs/24)); # number of days

    # tArr and tDep seem to have a value per day with the hour of arrival or departure. 
    γ = ones(ns)

    # using the data from Elaadusing Serialization
    # Deserialize the mixture model
    
    open(joinpath(@__DIR__, "..", "mpc\\input\\gmmElaadFit.dat"), "r") do f
        global gmm = deserialize(f) # Gaussian Mixture Model
    end
    # println("gmm", typeof(gmm), gmm)
    # load the lookup table of the connection times
    μtCon_tarr_df = CSV.read(joinpath(@__DIR__, "..", "mpc\\input\\mean-session-length-per.csv"),
                    DataFrame, silencewarnings = true);
    # sort following the arrival times
    sort!(μtCon_tarr_df, "Arrival Time")
    # add a column with the arrival time in hs
    μtCon_tarr_df.tArr = collect(0:0.5:23.5)
    gmmt = truncated(gmm, 0, 23.5) # truncate the GMM to the limits
    tArr = rand(rng, gmmt, ndays+1)

    # Interpolate mean session length for arrival times
    tCon_interp = linear_interpolation(μtCon_tarr_df.tArr, μtCon_tarr_df.home)
    tDep = tArr .+ tCon_interp.(tArr)

    # Adjust departure times if outside the limits
    tDep = [td > 23.5 ? td - 23.5 : td for td ∈ tDep]

    # Adjust first arrival and last departure times
    tDep = tDep[1:end-1]
    tArr = tArr[2:end]

    # Ensure departure time is before arrival time and not 0
    tDep = [t == 0 ? 0.5 : t for t in tDep]
    tArr = [tDep[i] > t ? tDep[i] + 1.0 : t for (i, t) in enumerate(tArr)]
    
    # Create the availability signal
    for day in 1:ndays
        # Determine the time indices corresponding to arrival and departure for this day
        t = collect(0:1/fs:(24-1/fs))
        # get the index of the departure and arrival times
        depIdx = findmin(abs.(tDep[day] .- t))[2]
        arrIdx = findmin(abs.(tArr[day] .- t))[2]
        # modify the index to be in the range of the time series, to avoid modifying supports
        tDep[day] = t[depIdx]
        tArr[day] = t[arrIdx]
        # Correct for the day
        depIdx = depIdx + (day-1)*24*fs
        arrIdx = arrIdx + (day-1)*24*fs
        # Mark the time series as parked during the parked interval for this day
        if depIdx <= arrIdx
            γ[depIdx:arrIdx] .= 0
        end
    end
    # Since we removed the the first element of tArr, 
    # the session length is the same as the first tDep append
    # the first element of tDep in the first position of Tconn
    # Tconn = [tDep[1]; Tconn]; Tconn = Tconn[1:end-1];
    Tconn = [24. + tDep[i] - tArr[i-1] for i ∈ 2:ndays]
    Tconn = [tDep[1]; Tconn]

    # Create the driving signal
    Pdrive_per_day = my_rejCriteriaEnergy(μDrive, σDrive, tDep, tArr, Tconn, Ns, Np, vmax, Q0, chgPmax; rng = rng)
    # Adapting the Pdrive to the number of samples
    Pdrive = repeat(Pdrive_per_day, inner = (fs*24))

    return γ, tDep, tArr, Pdrive
end

function replace_chunk_with_max(arr::Vector, chunk_size = 96)
    result = []
    
    for c in Iterators.partition(arr, chunk_size) 
        max_value = maximum(c)
        push!(result, fill(max_value, chunk_size)...)
    end
    @assert length(arr) == length(result) "Output doesn't match input length"
    return result
end

function replace_chunk_with_mean(arr::Vector{T}, chunk_size = 96) where {T}
    result = Vector{T}(undef, 0)
    
    for c in Iterators.partition(arr, chunk_size) 
        mean_value = mean(c)
        push!(result, fill(mean_value, chunk_size)...)
    end
    @assert length(arr) == length(result) "Output doesn't match input length"
    return result
end

################################################################################
# Train -> 274 days (26304 steps)
# Train_ex -> 365 days (35040 steps)
# Test -> 91 days (8736 steps)

# Get data
mpc_input = joinpath(@__DIR__,"..\\mpc\\input")

# PV data:
pvdata = JLD2.load(joinpath(mpc_input,"pv_data.jld2"), "pv_data")
pv_test, pv_train, pv_train_ex = test_train_expand(pvdata, first = true, rng = Xoshiro(14));
pv_cv = cv_train(pv_train, Xoshiro(11));
pv_cv_hard = cv_train(pv_train, Xoshiro(111), divergence = 1/2) .* 0.7;

# Prices:
priceData = JLD2.load(joinpath(mpc_input,"prices_2022.jld2"), "prices_2022")
buydata = priceData[:,1]
buy_test, buy_train, buy_train_ex = test_train_expand(buydata, first = true, rng = Xoshiro(12))
buy_cv = replace_chunk_with_mean(cv_train(buy_train, Xoshiro(7)), 4)
buy_cv_hard = replace_chunk_with_mean(cv_train(buy_train, Xoshiro(117), divergence = 2/3), 4) .* 1.25

selldata = priceData[:,2]
sell_test, sell_train, sell_train_ex = test_train_expand(selldata, first = true, rng = Xoshiro(16))
sell_cv = replace_chunk_with_mean(cv_train(sell_train, Xoshiro(6)), 4)
sell_cv_hard = replace_chunk_with_mean(cv_train(sell_train, Xoshiro(116), divergence = 2/3), 4) .* 0.85

prices_train = hcat(buy_train, sell_train)
prices_train_ex = hcat(buy_train_ex, sell_train_ex)
prices_test = hcat(buy_test, sell_test)
prices_cv = hcat(buy_cv, sell_cv)
prices_cv_hard = hcat(buy_cv_hard, sell_cv_hard)

# Load e and th:
loadE = JLD2.load(joinpath(mpc_input,"load_e.jld2"), "load_e")
loadE_mod = loadE * 5           # 5 kW as maximum value. 
loadE_test, loadE_train, loadE_train_ex = test_train_expand(loadE_mod, first = true, rng = Xoshiro(18))
loadE_cv = replace_chunk_with_mean(cv_train(loadE_train, Xoshiro(2001)), 4)
loadE_cv_hard = replace_chunk_with_mean(cv_train(loadE_train, Xoshiro(2101), divergence = 2/3), 4) .* 1.25

loadTh = JLD2.load(joinpath(mpc_input,"load_th.jld2"), "load_th")
loadTh_mod = loadTh * 2             # Maximum around 3 to 4 kWt
loadTh_test, loadTh_train, loadTh_train_ex = test_train_expand(loadTh_mod, first = true, rng = Xoshiro(19));
loadTh_cv = replace_chunk_with_mean(cv_train(loadTh_train, Xoshiro(99)), 4);
loadTh_cv_hard = replace_chunk_with_mean(cv_train(loadTh_train, Xoshiro(1199), divergence = 2/3), 4) .* 1.25 ;

# EV information:
γ_train, tDep_train, tArr_train, pdrive_train = myavailabilityEV(26304, 4, Xoshiro(42), 3.5, 1.5);
γ_train_ex, tDep_train_ex, tArr_train_ex, pdrive_train_ex = myavailabilityEV(35040, 4, Xoshiro(42), 3.5, 1.5);
γ_test, tDep_test, tArr_test, pdrive_test = myavailabilityEV(8736, 4, Xoshiro(1), 2.5, 1.25);
γ_cv, tDep_cv, tArr_cv, pdrive_cv = myavailabilityEV(8736, 4, Xoshiro(23), 3.5, 2.0);
clamp!(pdrive_cv, minimum(pdrive_train_ex), maximum(pdrive_train_ex));
γ_cv_hard, tDep_cv_hard, tArr_cv_hard, pdrive_cv_hard = myavailabilityEV(8736, 4, Xoshiro(123), 4.0, 1.5);


# Save the Data
train_data = joinpath(@__DIR__,"..\\data\\train.jld2")
jldsave(train_data, 
    pv_train = pv_train,
    buy_train = buy_train, 
    sell_train = sell_train, 
    prices_train = prices_train,
    loadE_train = loadE_train,
    loadTh_train = loadTh_train,
    γ_train = γ_train,
    tDep_train = tDep_train,
    tArr_train = tArr_train,
    pdrive_train = pdrive_train
)

train_ex_data = joinpath(@__DIR__,"..\\data\\train_ex.jld2")
jldsave(train_ex_data, 
    pv_train_ex = pv_train_ex,
    buy_train_ex = buy_train_ex,
    sell_train_ex = sell_train_ex,
    prices_train_ex = prices_train_ex,
    loadE_train_ex = loadE_train_ex,
    loadTh_train_ex = loadTh_train_ex,
    γ_train_ex = γ_train_ex,
    tDep_train_ex = tDep_train_ex,
    tArr_train_ex = tArr_train_ex,
    pdrive_train_ex = pdrive_train_ex
)

# Save the test data
test_data = joinpath(@__DIR__,"..\\data\\test.jld2")
jldsave(test_data, 
    pv_test = pv_test,
    buy_test = buy_test,
    sell_test = sell_test,
    prices_test = prices_test,
    loadE_test = loadE_test,
    loadTh_test = loadTh_test,
    γ_test = γ_test,
    tDep_test = tDep_test,
    tArr_test = tArr_test,
    pdrive_test = pdrive_test
)

Pdrive = rhDict["PevTot[1]"] - rhDict["Pev[1]"].*rhDict["γf"]
pdrive_mod = replace_chunk_with_max(Pdrive, 96)
γ_mod = rhDict["γf"]
jldsave(joinpath(@__DIR__,"..\\data\\pdrive_MPC.jld2"), pdrive_MPC = pdrive_mod, γ_MPC = γ_mod)

cv_data = joinpath(@__DIR__,"..\\data\\cv.jld2")
jldsave(cv_data, 
    pv_cv = pv_cv,
    buy_cv = buy_cv, 
    sell_cv = sell_cv,
    prices_cv = prices_cv,
    loadE_cv = loadE_cv,
    loadTh_cv = loadTh_cv,
    γ_cv = γ_cv,
    tDep_cv = tDep_cv,
    tArr_cv = tArr_cv,
    pdrive_cv = pdrive_cv
)

cv_data_hard = joinpath(@__DIR__,"..\\data\\cv_hard.jld2")
jldsave(cv_data_hard, 
    pv_cv_hard = pv_cv_hard,
    buy_cv_hard = buy_cv_hard, 
    sell_cv_hard = sell_cv_hard,
    prices_cv_hard = prices_cv_hard,
    loadE_cv_hard = loadE_cv_hard,
    loadTh_cv_hard = loadTh_cv_hard,
    γ_cv_hard = γ_cv_hard,
    tDep_cv_hard = tDep_cv_hard,
    tArr_cv_hard = tArr_cv_hard,
    pdrive_cv_hard = pdrive_cv_hard
)

@info "External Data has been imported."