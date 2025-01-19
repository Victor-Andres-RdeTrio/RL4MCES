# Load train data
data_path = joinpath(@__DIR__, "..", "data")
train_data = joinpath(data_path, "train.jld2")
@load train_data pv_train buy_train sell_train prices_train loadE_train loadTh_train γ_train tDep_train tArr_train pdrive_train

# Load expanded train data
train_ex_data = joinpath(data_path, "train_ex.jld2")
@load train_ex_data pv_train_ex buy_train_ex sell_train_ex prices_train_ex loadE_train_ex loadTh_train_ex γ_train_ex tDep_train_ex tArr_train_ex pdrive_train_ex

# Load test data
test_data = joinpath(data_path, "test.jld2")
@load test_data pv_test buy_test sell_test prices_test loadE_test loadTh_test γ_test tDep_test tArr_test pdrive_test

cv_data = joinpath(data_path, "cv.jld2")
@load cv_data pv_cv buy_cv sell_cv prices_cv loadE_cv loadTh_cv γ_cv tDep_cv tArr_cv pdrive_cv

cv_data_hard = joinpath(data_path, "cv_hard.jld2")
@load cv_data_hard pv_cv_hard buy_cv_hard sell_cv_hard prices_cv_hard loadE_cv_hard loadTh_cv_hard γ_cv_hard tDep_cv_hard tArr_cv_hard pdrive_cv_hard

pdrive_data = joinpath(data_path, "pdrive_MPC.jld2") # The Pdrive sequence implmented by the MPC is slightly different from the artificial pdrive_test. 
@load pdrive_data pdrive_MPC γ_MPC

exog_train = Exogenous_BatchCollection(
    load_e = Float32.(loadE_train),
    load_th = Float32.(loadTh_train),
    pv = Float32.(pv_train),
    λ_buy = Float32.(buy_train/1000), # €/kWh
    λ_sell = Float32.(sell_train/1000), # €/kWh
    p_drive = Float32.(pdrive_train),
    γ_ev = Float32.(γ_train),
    last_timestep = 26304
)

exog_train_ex = Exogenous_BatchCollection(
    load_e = Float32.(loadE_train_ex),
    load_th = Float32.(loadTh_train_ex),
    pv = Float32.(pv_train_ex),
    λ_buy = Float32.(buy_train_ex/1000),
    λ_sell = Float32.(sell_train_ex/1000),
    p_drive = Float32.(pdrive_train_ex),
    γ_ev = Float32.(γ_train_ex),
    last_timestep = 35040
)

exog_train_ex_γ_true = Exogenous_BatchCollection(
    load_e = Float32.(loadE_train_ex),
    load_th = Float32.(loadTh_train_ex),
    pv = Float32.(pv_train_ex),
    λ_buy = Float32.(buy_train_ex/1000),
    λ_sell = Float32.(sell_train_ex/1000),
    p_drive = Float32.(pdrive_train_ex),
    γ_ev = Float32.(fill(true, 35040)),
    last_timestep = 35040
)

exog_test_91 = Exogenous_BatchCollection(
    load_e = Float32.(loadE_test),
    load_th = Float32.(loadTh_test),
    pv = Float32.(pv_test),
    λ_buy = Float32.(buy_test/1000),
    λ_sell = Float32.(sell_test/1000),
    p_drive = Float32.(pdrive_test),
    γ_ev = Float32.(γ_test),
    last_timestep = 8736
)

exog_test_90 = Exogenous_BatchCollection(
    load_e = Float32.(loadE_test),
    load_th = Float32.(loadTh_test),
    pv = Float32.(pv_test),
    λ_buy = Float32.(buy_test/1000),
    λ_sell = Float32.(sell_test/1000),
    p_drive = Float32.(pdrive_MPC),  
    γ_ev = Float32.(γ_MPC), 
    last_timestep = 8640
)

exog_cv_90 = Exogenous_BatchCollection(
    load_e = Float32.(loadE_cv),
    load_th = Float32.(loadTh_cv),
    pv = Float32.(pv_cv),
    λ_buy = Float32.(buy_cv/1000),
    λ_sell = Float32.(sell_cv/1000),
    p_drive = Float32.(pdrive_cv),  
    γ_ev = Float32.(γ_cv),
    last_timestep = 8640
)

exog_cv_91 = Exogenous_BatchCollection(
    load_e = Float32.(loadE_cv),
    load_th = Float32.(loadTh_cv),
    pv = Float32.(pv_cv),
    λ_buy = Float32.(buy_cv/1000),
    λ_sell = Float32.(sell_cv/1000),
    p_drive = Float32.(pdrive_cv),  
    γ_ev = Float32.(γ_cv),
    last_timestep = 8736
)

exog_cv_hard_91 = Exogenous_BatchCollection(
    load_e = Float32.(loadE_cv_hard),
    load_th = Float32.(loadTh_cv_hard),
    pv = Float32.(pv_cv_hard),
    λ_buy = Float32.(buy_cv_hard/1000),
    λ_sell = Float32.(sell_cv_hard/1000),
    p_drive = Float32.(pdrive_cv_hard),  
    γ_ev = Float32.(γ_cv_hard),
    last_timestep = 8736
)

exog_cv_91_γ_true = Exogenous_BatchCollection(
    load_e = Float32.(loadE_cv),
    load_th = Float32.(loadTh_cv),
    pv = Float32.(pv_cv),
    λ_buy = Float32.(buy_cv/1000),
    λ_sell = Float32.(sell_cv/1000),
    p_drive = Float32.(pdrive_cv),  
    γ_ev = Float32.(fill(true, 8736)),
    last_timestep = 8736
)


@info "External Data has been imported."

