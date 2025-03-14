# Environment for the Multi Carrier Energy System

####################################################################################################
# Storage Assets
abstract type StorageAsset end

"""
    Battery <: StorageAsset

A mutable struct representing a battery storage asset in an energy management system.

# Fields
- `soc::Float32`: State of Charge (0-1), indicating the remaining battery capacity as a percentage.
- `soc_min::Float32`: Constant minimum safe SOC (0.2), the lowest level the battery can discharge to without damage.
- `soc_max::Float32`: Constant maximum safe SOC (0.95), the highest level the battery should be charged to.
- `q::Float32`: Capacity in Amp-hours (Ah) per cell, representing the total electrical charge a battery cell can store.
- `p::Float32`: Power output in kilowatts (kW), the rate at which the whole battery is delivering or absorbing power.
- `p_max::Float32`: Constant maximum power in kilowatts (17 kW) that can be delivered or accepted by the battery.
- `i::Float32`: Current in Amperes (A) flowing through each battery cell.
- `i_max::Float32`: Constant maximum safe current (7.8 A) for each battery cell.
- `v::Float32`: Voltage in Volts (V) across the battery cell terminals.
- `v_rated::Float32`: Constant rated voltage (3.6 V) for each battery cell.
- `v_max::Float32`: Constant maximum voltage (4.2 V) each battery cell can reach.
- `ocv::Float32`: Open Circuit Voltage in Volts (V), the theoretical voltage when no current flows.
- `a::Float32`: Constant OCV bias parameter (3.525 V), used in Open Circuit Voltage calculation.
- `b::Float32`: Constant OCV slope parameter (0.625 V/pu), determines how OCV changes with State of Charge.
- `η_c::Float32`: Constant Coulombic efficiency (0.99).
- `Np::Int32`: Constant number of parallel branches (10), independent battery cells connected in parallel.
- `Ns::Int32`: Constant number of series cells per branch (100), battery cells connected in series in each parallel branch.
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
    EV{A} <: StorageAsset

A mutable struct representing an Electric Vehicle (EV) with battery storage capabilities in an energy management system.

# Type Parameters
- `A`: The type of battery used in the EV, typically `Battery`.

# Fields
- `bat::A`: Battery properties, defaulting to a Battery with 25 parallel branches, 100 series cells per branch, and 12.5 kW max power.
- `γ::Bool`: Indicates whether the EV is present at the house/charging location (true) or away (false).
- `p_drive::Float32`: Power being demanded for driving the EV in kilowatts (kW).
- `soc_dep::Float32`: Constant desired State of Charge (0.85) at departure time.
- `departs::Bool`: Indicates whether the EV departs at the current timestep.
- `in_EMS::Bool`: Constant flag indicating whether the EV is part of the Energy Management System.
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

A mutable struct representing a Thermal Energy Storage System (TESS) for storing thermal energy.

# Fields
- `soc::Float32`: State of Charge (0-1), indicating the current level of stored thermal energy.
- `soc_min::Float32`: Constant minimum State of Charge (0.1) for safe operation.
- `soc_max::Float32`: Constant maximum State of Charge (0.95) for optimal operation.
- `p::Float32`: Power in kilowatts (kW) being delivered or received by the TESS.
- `p_max::Float32`: Constant maximum power (5 kW) that can be delivered or accepted by the TESS.
- `q::Float32`: Constant thermal energy capacity (200 kWh) of the storage system.
- `η::Float32`: Constant thermal transfer efficiency (0.95) of the storage system.
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

"""
    HP

A mutable struct representing a Heat Pump system for thermal energy conversion.

# Fields
- `e::Float32`: Heat pump electrical power consumption in kilowatts (kW).
- `th::Float32`: Heat pump thermal power output in kilowatts (kW).
- `e_max::Float32`: Constant maximum electrical power (4 kW) that could be demanded by the heat pump.
- `η::Float32`: Constant heat pump energy conversion efficiency/coefficient of performance (4.5).
"""
@kwdef mutable struct HP
    e::Float32 = 0.1f0          # Heat pump electrical power [kW]
    th::Float32 = 0.45f0        # Heat pump thermal power [kW]
    const e_max::Float32 = 4f0  # Maximum Electrical Power that could be demanded [kW]
    const η::Float32 = 4.5f0    # Heat pump energy conversion efficiency 

end


"""
    STC

A struct representing a Solar Thermal Collector (STC) for converting solar energy to thermal energy.

# Fields
- `η::Float32`: Constant conversion efficiency (0.6) from photovoltaic to solar thermal energy.
- `th_max::Float32`: Constant maximum thermal power output (2.7 kW) of the collector.
"""
@kwdef struct STC
    η::Float32 = 0.6f0       # Conversion from Photovoltaic to Solar Thermal
    th_max::Float32 = 2.7f0  # Maximum thermal power output [kW]
end

"""
    PEI

A struct representing a Power Electronics Interface for energy conversion and management.

# Fields
- `η::Float32`: Constant conversion efficiency (0.9) of the power electronics.
- `p_max::Float32`: Constant maximum power capacity (17 kW) of the interface.
"""
@kwdef struct PEI
    η::Float32 = 0.9f0
    p_max::Float32 = 17f0
end


  
####################################################################################################
# Weights and constant Parameters
"""
    MCES_Params

A struct holding reward weight parameters for the Multi-Carrier Energy System (MCES) optimization.

# Fields
- `w_init_proj::Float32`: Weight (0.5) for initial projection cost in the optimization objective.
- `w_op_proj::Float32`: Weight (0.5) for operational projection cost in the optimization objective.
- `w_grid::Float32`: Weight (1.0) for grid power penalty in the optimization objective.
- `w_soc::Float32`: Weight (50.0) for EV State of Charge penalty at departure.
- `w_ξsoc_ev::Float32`: Weight (1.0) for extra state of charge of EV (filtered).
- `w_ξsoc_bess::Float32`: Weight (1.0) for extra state of charge of BESS (filtered).
- `w_ξsoc_tess::Float32`: Weight (1.0) for extra state of charge of TESS (filtered).
- `w_ξp_tess::Float32`: Weight (1.0) for extra power of TESS (filtered).
- `w_ξp_grid::Float32`: Weight (1.0) for extra power of grid (filtered).
- `w_loss::Float32`: Weight (0.0) for capacity losses (not implemented).
- `c_loss::Float32`: Unit cost (0.0) of loss capacity in €/Ah (not implemented, 1.2 €/Ah noted as good estimate).
"""
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

"""
    MCES{P,T,B,E,H,S,PE} <: MCES_Env

A mutable struct representing a Multi-Carrier Energy System (MCES) environment that integrates various energy assets and manages their interactions.

# Type Parameters
- `P`: Type of parameters (`MCES_Params`)
- `T`: Type of thermal energy storage system (`TESS`)
- `B`: Type of battery energy storage system (`Battery`)
- `E`: Type of electric vehicle (`EV`)
- `H`: Type of heat pump (`HP`)
- `S`: Type of solar thermal collector (`STC`)
- `PE`: Type of power electronic interface (`PEI`)

# Fields
## System Configuration
- `params::P`: Parameters for weighting reward components during training.
- `mem_safe::Bool`: Flag indicating whether memory efficiency is a concern.
- `cum_cost_grid::Bool`: Flag indicating whether grid cost is cumulative.
- `simple_projection::Bool`: If true, uses simplified projection approach; if false, uses predictive projection (i.e. "safe projection").

## Power Flows
- `load_e::Float32`: Electrical load in kilowatts (kW).
- `load_th::Float32`: Thermal load in kilowatts (kW).
- `pv::Float32`: Photovoltaic power in kilowatts (kW).
- `st::Float32`: Solar thermal power in kilowatts (kW).
- `grid::Float32`: Power to/from the grid in kilowatts (kW).

## Pricing
- `λ_buy::Float32`: Price to buy electricity in currency per kWh (€/kWh).
- `λ_sell::Float32`: Price to sell electricity in currency per kWh (€/kWh).

## Energy Components
- `tess::T`: Thermal Energy Storage System.
- `bess::B`: Battery Energy Storage System.
- `ev::E`: Electric Vehicle.
- `hp::H`: Heat Pump.
- `stc::S`: Solar Thermal Collector.
- `pei::PE`: Power Electronic Interface.

## Safety and Optimization
- `safety_model::JuMP.AbstractModel`: Optimization model used to project actions into safe space.
- `model_type::String`: Type of optimization model (default: "NL").

## Simulation Parameters
- `rng::AbstractRNG`: Random number generator for the environment.
- `t::Int32`: General timestep counter that doesn't reset with end of episode.
- `t_ep::Int32`: Timestep counter within the current episode.
- `Δt::Int32`: Constant seconds per timestep (default: 900 seconds = 15 minutes).
- `daylength::Int32`: Constant timesteps per day (default: 96 timesteps).
- `episode_length::Int32`: Length of an episode in timesteps.

## Action Management
- `p_ev_raw::Float32`: Unfiltered EV power in kilowatts (kW).
- `p_bess_raw::Float32`: Unfiltered BESS power in kilowatts (kW).
- `p_hp_e_raw::Float32`: Unfiltered heat pump electrical power in kilowatts (kW).

## Reward System
- `reward_shape::Int32`: Defines the shape of the reward function.
- `cost_grid::Float32`: Penalty for total grid cost.
- `cost_degradation::Float32`: Penalty for capacity lost to degradation.
- `cost_ev::Float32`: Penalty for not charging EV to desired State of Charge.
- `init_projection::Float32`: Penalty for initial projection of actions.
- `grid_buffer::AbstractArray`: Buffer to store a week of prices and grid exchanges.

## Constraint Violations (Initial Projection)
- `ξp_ev::Float32`: EV power removed by filter in kilowatts (kW).
- `ξp_bess::Float32`: BESS power removed by filter in kilowatts (kW).
- `ξp_hp_e::Float32`: Heat pump electrical power removed by filter in kilowatts (kW).

## Operational Projection
- `op_projection::Float32`: Penalty for operational projection (constraint violations within environment).

## Constraint Violations (During Operation)
- `ξsoc_ev::Float32`: EV State of Charge violation.
- `ξsoc_bess::Float32`: BESS State of Charge violation.
- `ξsoc_tess::Float32`: TESS State of Charge violation.
- `ξp_tess::Float32`: TESS power violation in kilowatts (kW).
- `ξp_grid::Float32`: Grid power violation in kilowatts (kW).

## State Representation
- `n_actions::Int32`: Number of decisions the agent makes per timestep.
- `n_state_vars::Int32`: Number of distinct state variables accessible to the agent.
- `state_buffer::AbstractArray`: Buffer to store a day of state information.
- `state_buffer_dict::Dict`: Dictionary mapping state variables to buffer indices and lags.
- `state_buffer_ind::AbstractArray`: Matrix used to extract relevant values from the buffer.
"""
@kwdef mutable struct MCES{P,T,B,E,H,S,PE} <: MCES_Env
    # System parameters
    params::P = MCES_Params()

    # Customize the Environment (Booleans)
    mem_safe::Bool = false             # Is memory efficiency a concern?
    cum_cost_grid::Bool = false        # Is the grid cost cummulative?

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

"""
    build_MCES(; params::MCES_Params=MCES_Params(), mem_safe::Bool=false, cum_cost_grid::Bool=false, simple_projection::Bool=true, n_actions::Integer=3, n_state_vars::Integer=14, load_e::Real=0, load_th::Real=0, pv::Real=0, st::Real=0, grid::Real=0, λ_buy::Real=0, λ_sell::Real=0, tess::TESS=TESS(), bess::Battery=Battery(), ev::Union{EV,Nothing}=EV(), hp::HP=HP(), stc::STC=STC(), pei::PEI=PEI(), rng::AbstractRNG=Xoshiro(42), t::Integer=1, t_ep::Integer=1, Δt::Integer=900, episode_length::Integer=96, state_buffer_dict::Dict=Dict{Symbol, Tuple{<:Integer, Vector{<:Real}}}(), reward_shape::Integer=1, model_type::String="NL")

Constructor function for creating a memory-efficient MCES environment by converting all numeric values to appropriate types.

# Arguments
- `params::MCES_Params`: Parameters for the MCES optimization (default: `MCES_Params()`).
- `mem_safe::Bool`: Flag for memory efficiency concerns (default: `false`).
- `cum_cost_grid::Bool`: Flag for cumulative grid cost calculation (default: `false`).
- `simple_projection::Bool`: Flag for using simplified projection (default: `true`).
- `n_actions::Integer`: Number of decisions per timestep (default: `3`).
- `n_state_vars::Integer`: Number of state variables (default: `14`).
- `load_e::Real`: Initial electrical load in kW (default: `0`).
- `load_th::Real`: Initial thermal load in kW (default: `0`).
- `pv::Real`: Initial photovoltaic power in kW (default: `0`).
- `st::Real`: Initial solar thermal power in kW (default: `0`).
- `grid::Real`: Initial grid power in kW (default: `0`).
- `λ_buy::Real`: Initial price to buy electricity in €/kWh (default: `0`).
- `λ_sell::Real`: Initial price to sell electricity in €/kWh (default: `0`).
- `tess::TESS`: Thermal Energy Storage System (default: `TESS()`).
- `bess::Battery`: Battery Energy Storage System (default: `Battery()`).
- `ev::Union{EV,Nothing}`: Electric Vehicle (default: `EV()`). If `nothing`, creates a disabled EV.
- `hp::HP`: Heat Pump (default: `HP()`).
- `stc::STC`: Solar Thermal Collector (default: `STC()`).
- `pei::PEI`: Power Electronic Interface (default: `PEI()`).
- `rng::AbstractRNG`: Random number generator (default: `Xoshiro(42)`).
- `t::Integer`: Initial general timestep (default: `1`).
- `t_ep::Integer`: Initial episode timestep (default: `1`).
- `Δt::Integer`: Seconds per timestep (default: `900`).
- `episode_length::Integer`: Length of episode in timesteps (default: `96`).
- `state_buffer_dict::Dict`: Dictionary mapping state variables to buffer indices and lags (default: empty).
- `reward_shape::Integer`: Shape of reward function (default: `1`).
- `model_type::String`: Type of optimization model (default: `"NL"`).

# Returns
- An initialized `MCES` environment with appropriate type conversions.

# Details
1. If `ev` is `nothing`, creates a disabled EV with zeroed parameters.
2. If `state_buffer_dict` is empty, creates a default mapping of state variables to buffer indices and lags.
3. Converts all numeric values to appropriate types (`Float32` for floating-point values, `Int32` for integers).
4. Calculates `daylength` based on `Δt` and creates appropriate sized buffers.
5. Initializes the state buffer index matrix using the provided or default state buffer dictionary.

"""
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

"""
    RLBase.state(env::MCES_Env)

Retrieves the current state representation (feature vector) of the Multi-Carrier Energy System environment, which will be the input for the RL agent.

# Arguments
- `env::MCES_Env`: The MCES environment.

# Returns
- An array of the relevant state variables from the state buffer, selected according to the predefined state buffer indices.

# Details
The function collects data from the circular state buffer and filters it according to the boolean mask defined in `env.state_buffer_ind`.
"""
function RLBase.state(env::MCES_Env)
    collect(env.state_buffer)[env.state_buffer_ind] 
end

# State space
function RLBase.state_space(env::MCES_Env)
    ArrayProductDomain(fill((-1000 .. 1000), size(state(env)))) 
end

# The actions are: [P_EV, P_BESS, P_HPe]
RLBase.action_space(env::MCES_Env) = (-100.0 .. 100.0)×(-100.0 .. 100.0)×(0.0 .. 100.0) # Function from ReinforcementLearning.jl, not used in my implementation, but needed to avoid some errors.
RLBase.action_space(env::MCES_Env, player::DefaultPlayer) = action_space(env) # Function from ReinforcementLearning.jl, not used in my implementation, but needed to avoid some errors.

#Is the episode terminated
"""
    RLBase.is_terminated(env::MCES_Env)

Determines whether the current episode has terminated.

# Arguments
- `env::MCES_Env`: The MCES environment.

# Returns
- `true` if the current episode timestep exceeds the defined episode length, `false` otherwise.
"""
RLBase.is_terminated(env::MCES_Env) = env.t_ep > env.episode_length

# Reseed the RNG
"""
    Random.seed!(env::MCES_Env, seed::Integer)

Sets the random number generator seed for the MCES environment.

# Arguments
- `env::MCES_Env`: The MCES environment.
- `seed::Integer`: The seed value for the random number generator.

# Returns
- The result of calling `Random.seed!` on the environment's random number generator.
"""
Random.seed!(env::MCES_Env, seed::Integer) = Random.seed!(env.rng, seed)



"""
    RLBase.reset!(env::MCES_Env)

Performs a basic reset of the MCES environment at the start of a new episode.

# Arguments
- `env::MCES_Env`: The MCES environment to reset.

# Returns
- `nothing`

# Details
1. Resets the grid cost to zero (important when using cumulative cost calculation).
2. Resets the episode timestep counter to 1.
3. Does not affect the global timestep of the environment.
"""
function RLBase.reset!(env::MCES_Env)
    env.cost_grid = 0f0 # Reset so that if cummulative then only a day is considered. This can be modified. 
    env.t_ep = Int32(1)
    nothing
end

####################################################################################################
# Acting upon the environment and updating it

"""
    (env::MCES_Env)(decision)

Updates the environment object with valid power levels for EV, BESS, and Heat Pump based on the proposed decision vector (produced by the Agent).

# Arguments
- `env::MCES_Env`: The MCES environment object.
- `decision`: A vector containing proposed power levels for EV, BESS, and Heat Pump (in kW).

# Returns
- `nothing`: The function updates the environment object in-place.

# Details
1. Adjusts the EV power demand based on its availability in the energy management system (`in_EMS`).
2. Stores the raw (potentially unsafe) decision values in the environment for logging purposes.
3. Updates the internal safe projection model state via `update_model!`.
4. Projects the decision vector into a feasible action space using `valid_actions!`.
5. Calculates and stores penalties for initial constraint violations.
6. Updates the power levels in the EV battery, BESS, and Heat Pump components:
   - EV battery power (electrical) in kW
   - BESS power (electrical) in kW
   - Heat Pump electrical consumption in kW
   - Heat Pump thermal output in kWt (calculated using the heat pump efficiency)

The `simple_projection` flag in the environment determines which projection method is used.
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

"""
    energy_reset!(env::MCES_Env)

Resets the energy storage components in the MCES environment to their default states.

# Arguments
- `env::MCES_Env`: The MCES environment.

# Returns
- `nothing`

# Details
1. Resets Battery Energy Storage System (BESS) to default values:
   - State of Charge: 0.5
   - Current: 0.0
   - Voltage: rated voltage
2. If the Electric Vehicle (EV) is part of the Energy Management System:
   - Resets its battery to similar default values
3. Resets Thermal Energy Storage System (TESS) to default State of Charge (0.4)
4. Resets Heat Pump to default electrical (0.1 kW) and thermal (0.45 kW) power levels
"""
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

"""
    exogenous_reset!(env::MCES_Env)

Resets all exogenous (external) variables in the MCES environment to zero.

# Arguments
- `env::MCES_Env`: The MCES environment.

# Returns
- `nothing`
"""
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

"""
    total_recall!(env::MCES_Env, rng::AbstractRNG = Xoshiro(42))

Performs a complete reset of the MCES environment to its initial state.

# Arguments
- `env::MCES_Env`: The MCES environment.
- `rng::AbstractRNG`: Random number generator to use (default: Xoshiro(42)).

# Returns
- `nothing`

# Details
1. Calls `energy_reset!` to reset all energy storage components.
2. Calls `exogenous_reset!` to reset all exogenous variables.
3. Resets the random number generator and episode timestep.
4. Resets all penalty and cost values to zero:
   - Initial projection penalty
   - Operational projection penalty
   - Grid cost
   - Degradation cost
   - EV cost
5. Clears the state buffer and grid buffer by setting all values to zero.
"""
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