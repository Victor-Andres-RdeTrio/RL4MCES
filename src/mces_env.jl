# Environment for the Multi Carrier Energy System


""" 
This is the environment for the Multi Carrier Energy System. 
It recreates the high fidelity digital twin of the house studied in the project. 

""" 

####################################################################################################
# Storage Assets
abstract type StorageAsset end


"""
    Battery <: StorageAsset

A mutable struct representing a Battery storage asset. Most paramters relate to the cells that make up the battery.

**Fields:**

* `soc` (0-1): **State of Charge**. The remaining battery capacity as a percentage.
* `soc_min` (0-1): **Minimum Safe SOC**. The lowest level the battery can be discharged to without risk of damage.
* `q` (Ah): **Capacity**. The total amount of electrical charge a battery cell can store.
* `p` (kW): **Power Output**. The current rate at which the whole battery is delivering or absorbing power.
* `i` (A): **Current**. The current flowing through each battery cell.
* `i_max` (A): **Maximum Current**. The highest safe current for the battery cell.
* `v` (V): **Voltage**. The electrical potential across the battery cell terminals.
* `v_max` (V): **Maximum Voltage**. The highest voltage the battery cell can reach.
* `ocv` (V): **Open Circuit Voltage**. The theoretical voltage when no current flows.
* `a` (V): **OCV Bias Parameter**. A constant used to calculate the Open Circuit Voltage.
* `b` (V/pu): **OCV Slope Parameter**. A constant that determines how Open Circuit Voltage changes with State of Charge.
* `η_c` (0-1): **Coulombic Efficiency (constant)**. The ratio of charge recovered during discharge compared to charge put in during charging (always between 0 and 1).
* `Np`: **Parallel Branches (constant)**. The number of independent battery cells connected in parallel.
* `Ns`: **Series Cells per Branch (constant)**. The number of battery cells connected in series within each parallel branch.
"""
@kwdef mutable struct Battery <: StorageAsset
    # State of Charge and Capacity
    soc::Float32 = 0.5f0     # State of Charge (0 < soc < 1)
    const soc_min::Float32 = 0.2f0 # Minimum SoC
    const soc_max::Float32 = 0.95f0 # Maximum SoC
    q::Float32 = 5.2f0       # [Ah/cell]
    p::Float32 = 0f0       # Power being delivered by the whole battery [kW]
    const p_max::Float32 = 17f0 # Max power that can be delivered or accepted by the whole battery [kW] 

    # Current and Voltage (with limits) of each cell.
    i::Float32 = 0f0       # Amperes
    const i_max::Float32 = 7.8f0  # Maximum value for the current (A)
    v::Float32 = 3.6f0    # Cell current voltage (V)
    const v_rated::Float32 = 3.6f0   # Rated value for the Voltage (V)
    const v_max::Float32 = 4.2f0   # Maximum value for the Voltage (V)

    # Open Circuit Voltage Parameters
    ocv::Float32 = 0f0  # Open Circuit Voltage (V)
    const a::Float32 = 3.525f0  # Parameter for OCV calculation (bias)  [V]
    const b::Float32 = 0.625f0  # Parameter for OCV calculation (slope) [V/pu] (pu = per unit of SOC, 50 pu = 0.5)

    # Efficiency and Cell Configuration
    const η_c::Float32 = 0.99f0   # Coulombic efficiency (0 < η_c < 1)
    const Np::Int32 = Int32(10)     # Parallel branches
    const Ns::Int32 = Int32(100)     # Series Cells per branch
end

  
"""
    EV <: StorageAsset

A mutable struct representing an electric vehicle (EV) with associated properties.

# Fields:
* `bat::Battery`: Battery properties.
* `γ::Bool`: Indicates whether the EV is in the house.
* `soc_dep::Real`: Desired State of Charge at departure.
* `departs::Bool`: Indicates whether the EV departs at the current timestep.
* `in_EMS::Bool` : Indicates whether the EV is part of the Energy Management System . 

"""  
@kwdef mutable struct EV{A} <: StorageAsset
    # Battery Properties
    bat :: A = Battery(Np=Int32(25), Ns=Int32(100), p_max = 12.5f0)
  
    # Location and Charge
    γ:: Bool = true           # Is the EV in the House?
    p_drive::Float32 = 0f0    # Power being demanded for driving the EV [kW]
    const soc_dep::Float32 = 0.85f0   # Desired SoC at departure
    departs:: Bool = false    # It will be true for a timestep when the ev departs.
    const in_EMS::Bool = true       # Is the EV part of the Energy Management System? 
end

"""
    TESS

A mutable struct representing a Thermal Energy Storage System (TESS) with associated properties.

# Fields:
* `soc::Real`: State of Charge (0 < soc < 1).
* `soc_min::Real`: Minimum State of Charge.
* `q::Real`: Capacity [KWh].
* `η::Real`: Thermal transfer efficiency.

"""
@kwdef mutable struct TESS 
    # State of Charge 
    soc::Float32 = 0.4f0                # (0 < soc < 1)
    const soc_min::Float32 = 0.1f0      # Minimum SoC
    const soc_max::Float32 = 0.95f0     # Maximum SoC
    p::Float32   = 0f0                # Power being delivered (or received) by the TESS
    const p_max::Float32 = 5f0        # Max power that can be delivered or accepted by the TESS [kW] 

    # Design properties
    const q ::Float32 = 200f0 # Capacity [kWh]
    const η ::Float32 = 0.95f0  # Thermal transfer efficiency
end

# function TESS(;soc::Real = 0.4, p::Real = 0, soc_min::Real=0.1, soc_max::Real=0.95, p_max::Real=5, q::Real=200, η::Real=0.95)
#     TESS(Float32(soc), Float32(soc_min), Float32(soc_max), Float32(p), Float32(p_max), Float32(q), Float32(η))
# end
####################################################################################################
# Heat Pump

"""
    HP

A mutable struct representing a Heat Pump with associated properties.

# Fields:
* `e::Real`: Heat pump electrical power [kW].
* `th::Real`: Heat pump thermal power [kW].
* `e_max::Real`: Maximum Electrical Power that could be demanded [kW].
* `η::Real`: Heat pump energy conversion efficiency.

"""
@kwdef mutable struct HP
    e::Float32 = 0.1f0          # Heat pump electrical power [kW]
    th::Float32 = 0.45f0        # Heat pump thermal power [kW]
    const e_max::Float32 = 4f0  # Maximum Electrical Power that could be demanded [kW]
    const η::Float32 = 4.5f0    # Heat pump energy conversion efficiency 

end

"""
    STC

A struct representing a Solar Thermal Collector (STC) with associated properties.

# Fields:
* `η::Real`: Conversion from Photovoltaic to Solar Thermal.
* `th_max::Real`: Maximum thermal power output [kW].

"""
@kwdef struct STC
    η::Float32 = 0.6f0       # Conversion from Photovoltaic to Solar Thermal
    th_max::Float32 = 2.7f0  # Maximum thermal power output [kW]
end

@kwdef struct PEI
    η::Float32 = 0.9f0
    p_max::Float32 = 17f0
end


  
####################################################################################################
# Weights and constant Parameters
@kwdef struct MCES_Params
    w_init_proj::Float32 = 0.5f0  # Weight for initial projection cost
    w_op_proj::Float32 = 0.5f0  # Weight for operational projection cost
    w_grid::Float32 = 1.0f0  # Weight for grid power penalty
    w_soc::Float32 = 50f0  # Weight for EV SoC Penalty at departure
    w_ξsoc_ev::Float32 = 1f0  # Weight for extra state of charge of EV (filtered)
    w_ξsoc_bess::Float32 = 1f0  # Weight for extra state of charge of BESS (filtered)
    w_ξsoc_tess::Float32 = 1f0  # Weight for extra state of charge of TESS (filtered)
    w_ξp_tess::Float32 = 1f0  # Weight for extra power of TESS (filtered)
    w_ξp_grid::Float32 = 1f0  # Weight for extra power of grid (filtered)

    # Not implemented
    w_loss::Float32 = 0f0  # Weight for Capacity losses
    c_loss::Float32 = 0f0 # Unit cost of loss Capacity [€/Ah], 1.2 €/Ah good estimate. 
end


####################################################################################################
# Environment
abstract type MCES_Env <: AbstractEnv end

@kwdef mutable struct MCES{P,T,B,E,H,S,PE} <: MCES_Env
    # System parameters
    params::P = MCES_Params()

    # Customize the Environment (Booleans)
    mem_safe::Bool = false             # Is memory efficiency a concern?
    cum_cost_grid::Bool = false        # Is the grid cost cummulative?

    # If false, a prediction of the operational projection will be applied together with the initial projection, 
    # so that most operational projection is avoided.
    simple_projection::Bool = true    

    # Power flows (Watts)
    load_e::Float32 = 0f0   # Electrical load [kW]
    load_th::Float32 = 0f0  # Thermal load [kW]
    pv::Float32 = 0f0       # Photovoltaic power [kW]
    st::Float32 = 0f0       # Solar thermal power [kW]
    grid::Float32 = 0f0     # Power to/from the Grid. [kW]
  
    # Prices (currency per kWh)
    λ_buy::Float32 = 0f0   # Price to buy electricity [€/kWh]
    λ_sell::Float32 = 0f0  # Price to sell electricity [€/kWh]
  
    # Components
    tess::T = TESS()         # Thermal Energy Storage System
    bess::B = Battery()      # Battery Energy Storage System
    ev::E = EV()             # Electric Vehicle
    hp::H = HP()             # Heat pump
    stc::S = STC()           # Solar Thermal Collector
    pei::PE = PEI()          # Power Electronic Interface 

    # Safety Layer
    safety_model::JuMP.AbstractModel = Model()  # Optimization model used to project actions into safe space. 
    model_type::String = "NL"  # Current options: NL 

    # Environment properties
    rng::AbstractRNG = Xoshiro(42)  # Environment random number generator
    t::Int32 = Int32(1)                      # General Timestep (doesn't reset with end of episode) 
    t_ep::Int32 = Int32(1)                   # Timestep within the episode. 
    const Δt::Int32 = Int32(900)             # Seconds per timestep (default = 15 min)
    const daylength::Int32 = Int32(96)       # Timesteps per day
    episode_length::Int32 = Int32(96)        # Length of the episode in timesteps. 

    # Last Unfiltered actions
    p_ev_raw::Float32   = 0f0   # EV Power [kW] (not filtered)
    p_bess_raw::Float32 = 0f0   # BESS Power [kW] (not filtered)
    p_hp_e_raw::Float32 = 0f0   # HPe Power [kW] (not filtered)
 
    # Rewards/Penalties
    reward_shape::Int32 = 1            # Defines the shape that the reward function will have. 
    cost_grid::Float32 = 0f0           # Penalty for total cost. 
    cost_degradation::Float32 = 0f0    # Penalty for capacity lost to degradation
    cost_ev::Float32 = 0f0             # Penalty for not charging EV to desired SoC. 
    init_projection::Float32 = 0f0     # Penalty for the initial projection of actions (before passed to env)
    grid_buffer::AbstractArray = CircularArrayBuffer(zeros(Float32, 3, 96*7))  # Buffer to store a week of prices and grid exchanges. 
    
    # Logging: Constraint violations (Initial Projection)
    ξp_ev::Float32 = 0f0        # EV power [kW] removed by filter. 
    ξp_bess::Float32 = 0f0      # BESS power [kW] removed by filter. 
    ξp_hp_e::Float32 = 0f0      # HPe power [kW] removed by filter. 
    
    
    op_projection::Float32 = 0f0     # Penalty for Operational Projection -> constraint violations within environment. 
    # Logging: Constraint violations (During Operation)
    ξsoc_ev::Float32 = 0f0
    ξsoc_bess::Float32 = 0f0
    ξsoc_tess::Float32 = 0f0
    ξp_tess::Float32 = 0f0
    ξp_grid::Float32 = 0f0

    # State representation (features for the Neural Network)
    n_actions::Int32 = Int32(3)     # Amount of decisions the agent makes per timestep
    n_state_vars::Int32 = Int32(14)     # Amount of distinct state variables that the agent will have access to
    state_buffer::AbstractArray = CircularArrayBuffer(zeros(Float32, 14, 96))  # Buffer to store a day of state information
    
    # To choose what state information is useful from the buffer,
    # A Dict with Symbol => (row_index, [lags]) is used.
    # The lags are in days and day fractions. Each row should be assigned to one state variable. 
    state_buffer_dict::Dict = Dict{Symbol, Tuple{<:Integer, Vector{<:Real}}}()   

    # The state_buffer_dict gets converted into a Matrix. Used to extract relevant values from buffer. 
    state_buffer_ind::AbstractArray = falses(14, 96)
end

# Outer constructor for memory efficiency -> converts all fields to Float32 or Int32.
function build_MCES(; 
    params::MCES_Params = MCES_Params(),
    mem_safe::Bool=false,
    cum_cost_grid::Bool=false,
    simple_projection::Bool=true,
    n_actions::Integer = 3,
    n_state_vars::Integer = 14,
    load_e::Real=0,
    load_th::Real=0,
    pv::Real=0,
    st::Real=0,
    grid::Real=0,
    λ_buy::Real=0,
    λ_sell::Real=0,
    tess::TESS=TESS(),
    bess::Battery=Battery(),
    ev::Union{EV,Nothing}=EV(),
    hp::HP=HP(),
    stc::STC=STC(),
    pei::PEI=PEI(),
    rng::AbstractRNG=Xoshiro(42),
    t::Integer=1,
    t_ep::Integer=1,
    Δt::Integer=900,
    episode_length::Integer=96,
    state_buffer_dict::Dict = Dict{Symbol, Tuple{<:Integer, Vector{<:Real}}}(),
    reward_shape::Integer = 1,
    model_type::String = "NL"
    )

    if isnothing(ev)
        ev_battery = Battery(
            soc = 0f0,
            soc_min = 0f0,
            soc_max = 0f0,
            p_max = 0f0,
            i_max = 0f0,
            v = 0f0,
            v_rated = 0f0,
            v_max = 0f0
        )
        
        ev = EV{Battery}(
            bat = ev_battery,
            in_EMS = false
        )
    end

    if isempty(state_buffer_dict)
        state_buffer_dict = Dict(
            :load_e => (1, [0.0, 0.5, 1.0]),
            :load_th => (2, [0.0, 0.5, 1.0]),
            :pv => (3, [0.0, 0.5, 0.65, 1.0]),
            :λ_buy => (4, [0.1, 0.45, 0.65, 1.0]),
            :λ_sell => (5, [0.1, 0.45, 0.65, 1.0]), 
            :γ_ev => (6, [0.0, 1.0]),
            :grid => (7, [0.0, 0.5, 1.0]),
            :p_bess => (8, [0.0, 1.0]),
            :p_ev => (9, [0.0, 0.5, 1.0]),
            :p_hp_e => (10, [0.0, 0.5, 1.0]),
            :soc_bess => (11, [0.1, 0.4, 0.8, 1.0]),
            :soc_ev => (12, [0.25, 0.75, 1.0]),
            :soc_tess => (13, [0.25, 0.5, 0.75]),
            :t_ep_ratio => (14, [0.0, 0.3, 0.9])
        )
        @info "Using default features from the State Buffer."
    end

    return MCES(
        params=deepcopy(params),
        mem_safe=mem_safe,
        cum_cost_grid=cum_cost_grid,
        simple_projection=simple_projection,
        load_e=Float32(load_e),
        load_th=Float32(load_th),
        pv=Float32(pv),
        st=Float32(st),
        grid=Float32(grid),
        λ_buy=Float32(λ_buy),
        λ_sell=Float32(λ_sell),
        tess=deepcopy(tess),
        bess=deepcopy(bess),
        ev=deepcopy(ev),
        hp=deepcopy(hp),
        stc=deepcopy(stc),
        pei=deepcopy(pei),
        rng=rng,
        t=Int32(t),
        t_ep=Int32(t_ep),
        Δt=Int32(Δt),
        episode_length = Int32(episode_length),
        daylength = Int32(div(24*3600, Δt)),
        n_actions = Int32(n_actions),
        n_state_vars = Int32(n_state_vars),
        state_buffer = CircularArrayBuffer(zeros(Float32, n_state_vars, div(24*3600, Δt))),
        state_buffer_dict = deepcopy(state_buffer_dict),
        state_buffer_ind = matrix_select_states(;state_buffer_dict...),
        reward_shape = Int32(reward_shape),
        grid_buffer = CircularArrayBuffer(zeros(Float32, 3, div(24*3600, Δt)*7)),
        model_type = model_type
    )
end

####################################################################################################
# Implementations of ReinforcementLearningBase functions
# function RLBase.state(env::MCES_Env) 
#     t_ep_ratio = mod1(env.t_ep, 96)/96 # Moment of the day, between 0 and 1. Independent of ep length
#     # t_year_ratio = env.t/(365*24*3600/env.Δt) # Time of the year from 0 to 1

#     # Normalized according to expanded training set. 
#     Float32[
#         z_score(env.load_e, 1.711f0, 1.013f0),
#         z_score(env.load_th, 1.310f0, 0.811f0),
#         z_score(env.pv, 0.528f0, 0.873f0),
#         z_score(env.λ_buy, 0.242f0, 0.128f0), 
#         z_score(env.λ_sell, 0.231f0, 0.121f0),
#         env.tess.soc,
#         env.bess.soc,
#         env.ev.bat.soc, 
#         env.ev.γ, 
#         t_ep_ratio
#     ]    
# end

function RLBase.state(env::MCES_Env)
    collect(env.state_buffer)[env.state_buffer_ind] 
end

# State space
function RLBase.state_space(env::MCES_Env)
    ArrayProductDomain(fill((-1000 .. 1000), size(state(env)))) 
end

# The actions are: [P_EV, P_BESS, P_HPe]
RLBase.action_space(env::MCES_Env) = (-100.0 .. 100.0)×(-100.0 .. 100.0)×(0.0 .. 100.0)
RLBase.action_space(env::MCES_Env, player::DefaultPlayer) = action_space(env)

#Is the episode terminated
RLBase.is_terminated(env::MCES_Env) = env.t_ep > env.episode_length
#Reseed the RNG
Random.seed!(env::MCES_Env, seed::Integer) = Random.seed!(env.rng, seed)



# Basic Reset, performed everytime an experiment starts and at the end of an episode.
# Doesn't affect the global timestep of the env.  
function RLBase.reset!(env::MCES_Env)
    env.cost_grid = 0f0 # Reset so that if cummulative then only a day is considered. This can be modified. 
    env.t_ep = Int32(1)
    nothing
end

####################################################################################################
# Acting upon the environment and updating it

"""
    (env::MCES_Env)(decision)

Updates the environment object (env) with valid power levels for EV, BESS, and Heat Pump based on the proposed decision.

# Args:

* `env`: The MCES environment object.
* `decision`: A list containing proposed power levels for EV, BESS, and Heat Pump (electric).

# Returns:

* `nothing`. The function updates the environment object (`env`) in-place.

# Details
1. Extracts EV battery, BESS, and Heat Pump objects from the environment.
2. Calls `valid_actions` to project proposed power levels into valid ones for the current state.
3. Updates power levels in EV, BESS, and Heat Pump objects within the environment.
4. Calculates and assigns a penalty for the projection.

"""
function (env::MCES_Env)(decision)
    #Extracting components from the Env
    ev = env.ev           # EV 
    bess = env.bess       # BESS 
    hp = env.hp           # Heat Pump

    decision[1] = decision[1] * ev.in_EMS  # If EV is not in EMS, power_EV should be 0. 

    # Potentially unsafe decisions
    env.p_ev_raw   = decision[1]   
    env.p_bess_raw = decision[2]   
    env.p_hp_e_raw = decision[3]

    update_model!(env)

    # Project the decision into feasible space
    act = Vector{Float32}(undef, 3)
    valid_actions!(act, decision, env; simple = env.simple_projection) # All in kW

    # Logging constraint violations
    env.ξp_ev = abs(decision[1] - act[1])
    env.ξp_bess = abs(decision[2] - act[2])
    env.ξp_hp_e = abs(decision[3] - act[3])

    # Update the decision fields in the env: 
    ev.bat.p = act[1]       # EV [kW] (it will be always 0 if EV is not in EMS)
    bess.p = act[2]         # BESS [kW]
    hp.e  = act[3]          # HP [kW]
    hp.th = act[3] * hp.η   # HP [kWt]

    nothing
end


####################################################################################################
# Reset options

function energy_reset!(env::MCES_Env)
    # Reset Battery
    env.bess.soc = 0.5f0
    env.bess.i = 0.0f0
    env.bess.v = env.bess.v_rated

    # Reset Electric Vehicle
    if env.ev.in_EMS
        env.ev.bat.soc = 0.5f0
        env.ev.bat.i = 0.0f0
        env.ev.bat.v = env.ev.bat.v_rated
    end
    # Reset Thermal Energy Storage System
    env.tess.soc = 0.4f0

    # Reset Heat Pump
    env.hp.e = 0.1f0
    env.hp.th = 0.45f0

    nothing
end

@inline function exogenous_reset!(env::MCES_Env)
    # Reset exogenous information
    env.load_e = 0f0
    env.load_th = 0f0
    env.pv = 0f0
    env.st = 0f0
    env.grid = 0f0
    
    # Reset Prices
    env.λ_buy = 0f0
    env.λ_sell = 0f0

    env.t = 1
    nothing
end

function total_recall!(env::MCES_Env, rng::AbstractRNG = Xoshiro(42))
    
    energy_reset!(env)
    exogenous_reset!(env)
    
    # Reset Environment Properties
    env.rng = rng 
    env.t_ep = 1
    
    # Reset Rewards/Penalties
    env.init_projection = 0f0
    env.op_projection = 0f0
    env.cost_grid = 0f0
    env.cost_degradation = 0f0
    env.cost_ev = 0f0
    env.state_buffer .= 0f0 
    env.grid_buffer .= 0f0 

    nothing
end


"""

| Trait Type        |                  Value |
|:----------------- | ----------------------:|
| NumAgentStyle     |          SingleAgent() |
| DynamicStyle      |           Sequential() |
| InformationStyle  | ImperfectInformation() |
| ChanceStyle       |           Stochastic() |
| RewardStyle       |       TerminalReward() |
| UtilityStyle      |           GeneralSum() |
| ActionStyle       |     MinimalActionSet() |
| StateStyle        |     Observation{Any}() |
| DefaultStateStyle |     Observation{Any}() |
"""


@info "Basic Environment functionality now operational"