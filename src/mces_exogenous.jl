####################################################################################################
# Exogenous Data
abstract type Exogenous end

"""
    Exogenous_Batch <: Exogenous

Represents a single timestep of exogenous (external) data for a energy system simulation.

# Fields
- `load_e::Float32 = 0.0`: Electrical load in kilowatts [kW]. Must be non-negative.
- `load_th::Float32 = 0.0`: Thermal load in kilowatts [kW]. Must be non-negative.
- `pv::Float32 = 0.0`: Photovoltaic power generation in kilowatts [kW]. Must be non-negative.
- `λ_buy::Float32 = 0.0`: Price to buy electricity in euros per kilowatt-hour [€/kWh]. May be negative.
- `λ_sell::Float32 = 0.0`: Price to sell electricity in euros per kilowatt-hour [€/kWh]. May be negative.
- `p_drive::Float32 = 0.0`: Power used for driving the Electric Vehicle (EV) in kilowatts [kW]. Must be non-negative.
- `γ_ev::Bool = true`: Indicates whether the Electric Vehicle (EV) is present in the house.

# Notes
- All power values (load_e, load_th, pv, p_drive) must be in kilowatts [kW].
- Price values (λ_buy, λ_sell) must be in euros per kilowatt-hour [€/kWh].
- Ensure that all values are in the correct units to maintain consistency and accuracy in calculations.
"""
@kwdef struct Exogenous_Batch <: Exogenous
  # Power
  load_e::Float32 = 0.0    # Electrical load [kW]
  load_th::Float32 = 0.0   # Thermal load [kW]
  pv::Float32 = 0.0        # Photovoltaic power generation [kW]

  # Prices
  λ_buy::Float32 = 0.0     # Price to buy electricity (€/kWh)
  λ_sell::Float32 = 0.0    # Price to sell electricity (€/kWh)

  # EV:
  p_drive::Float32 = 0.0   # Power used for driving the EV [kW]  
  # Is the Electric Vehicle (EV) present in the house?
  γ_ev::Bool = true

  # t::Int = 0                      # Timestep (maybe unnecessary)
end


"""
    Exogenous_BatchCollection{V,B} <: Exogenous

Represents a collection of exogenous (external) data for multiple timesteps in an energy system simulation.

# Fields
- `load_e::V = zeros(0)`: Vector of electrical loads in kilowatts [kW]. Each element must be non-negative.
- `load_th::V = zeros(0)`: Vector of thermal loads in kilowatts [kW]. Each element must be non-negative.
- `pv::V = zeros(0)`: Vector of photovoltaic power generation in kilowatts [kW]. Each element must be non-negative.
- `λ_buy::V = zeros(0)`: Vector of prices to buy electricity in euros (or any currency) per kilowatt-hour [€/kWh]. 
- `λ_sell::V = zeros(0)`: Vector of prices to sell electricity in euros (or any currency) per kilowatt-hour [€/kWh]. 
- `p_drive::V = zeros(0)`: Vector of power used for driving the EV in kilowatts [kW]. Each element must be non-negative.
- `γ_ev::B = [true]`: Vector indicating whether the Electric Vehicle (EV) is present in the house for each timestep.
- `last_timestep::Int64 = 96`: The last timestep included in the collection.

# Notes
- The last timestep must be within bounds of all the vectors.
- All power values (load_e, load_th, pv, p_drive) must be in kilowatts [kW].
- Price values (λ_buy, λ_sell) must be in euros (or any currency) per kilowatt-hour [€/kWh].
- Ensure that all values are in the correct units to maintain consistency and accuracy in calculations.
"""
@kwdef struct Exogenous_BatchCollection{V,B} <: Exogenous
    # Power
    load_e::V = Float32[]   # Electrical load [kW]
    load_th::V = Float32[]  # Thermal load [kW]
    pv::V = Float32[]       # Photovoltaic power generation [kW]

    # Prices
    λ_buy::V = Float32[]    # Price to buy electricity (€/kWh)
    λ_sell::V = Float32[]   # Price to sell electricity (€/kWh)

    p_drive::V = Float32[]  # Power used for driving the EV [kW] 
    # Is the Electric Vehicle (EV) present in the house?
    γ_ev::B = Bool[]       # Expects booleans.

    last_timestep::Int32 = Int32(96)              # Last timestep included in the collection
end  

# function Exogenous_BatchCollection(load_e::Vector{T}, load_th::Vector{T}, pv::Vector{T}, 
#                                    λ_buy::Vector{T}, λ_sell::Vector{T}, p_drive::Vector{T}, 
#                                    γ_ev::Vector{T}, last_timestep::Signed) where T <: Real
#     V32 = Vector{Float32}
#     return Exogenous_BatchCollection{V32, Vector{Bool}}(
#         convert(V32, load_e), convert(V32, load_th), convert(V32, pv),
#         convert(V32, λ_buy), convert(V32, λ_sell), convert(V32, p_drive),
#         convert(Vector{Bool}, γ_ev), convert(Int32, last_timestep)
#     )
# end

function Exogenous_BatchCollection(load_e::Vector, load_th::Vector, pv::Vector, 
                                   λ_buy::Vector, λ_sell::Vector, p_drive::Vector, 
                                   γ_ev::Vector, last_timestep::Signed) 
    V32 = Vector{Float32}
    return Exogenous_BatchCollection{V32, Vector{Bool}}(
        convert(V32, load_e), convert(V32, load_th), convert(V32, pv),
        convert(V32, λ_buy), convert(V32, λ_sell), convert(V32, p_drive),
        convert(Vector{Bool}, γ_ev), convert(Int32, last_timestep)
    )
end


function subset_exog(exo::Exogenous_BatchCollection, s::Int, e::Int)
    s = max(1, s) 
    e = min(e, minimum(length, (exo.load_e, exo.load_th, exo.pv, exo.λ_buy, exo.λ_sell, exo.p_drive)))

    Exogenous_BatchCollection(
        load_e = exo.load_e[s:e],
        load_th = exo.load_th[s:e],
        pv = exo.pv[s:e],
        λ_buy = exo.λ_buy[s:e],
        λ_sell = exo.λ_sell[s:e],
        p_drive = exo.p_drive[s:e],
        γ_ev = exo.γ_ev[s:e],
        last_timestep = e - s + 1
    )
end

####################################################################################################
# Exogenous Update

"""
    exogenous!(env::MCES_Env, W::Exogenous_BatchCollection)

Selects the exogenous information for the next timestep out of a collection.
Creates a data package (::Exogenous_Batch) with just the new timestep. 
Dispatches to `exogenous!(env::MCES_Env, W::Exogenous_Batch)`

# Args:
* `env`: The MCES environment object.
* `W`: An object containing the new load, generation, prices, EV location.

# Returns:
* `nothing`. 

# Details:
1. The first timestep is env.t = 1.
2. Exogenous_BatchCollection will be first indexed with the timestep (1) to get the first element of each field.  
3. With this information, the transition to env.t = 2 can take place. 
"""
function exogenous!(env::MCES_Env, W::Exogenous_BatchCollection)

    @assert env.t <= W.last_timestep "Last timestep of Exogenous_BatchCollection has been reached."

    # Extract data for the next timestep from the collection
    load_e = W.load_e[env.t]
    load_th = W.load_th[env.t]
    pv = W.pv[env.t]
    λ_buy = W.λ_buy[env.t]
    λ_sell = W.λ_sell[env.t]
    p_drive = W.p_drive[env.t]
    γ_ev = W.γ_ev[env.t]

    # Dispatch
    exogenous!(env, Exogenous_Batch(load_e, load_th, pv, λ_buy, λ_sell, p_drive, Bool(γ_ev)))

    nothing
end

"""
    exogenous!(env::MCES_Env, W::Exogenous_Batch)

Updates the environment object (env) with the exogenous information of only the next timestep provided.

# Args:
* `env`: The MCES environment object.
* `W`: An object containing the new load, generation, prices, EV location.

# Returns:
* `nothing`. The function updates the environment object (`env`) in-place.

# Details
1. Increases the timestep
2. Provides the information about the average power consumption and generation in between t and t+1.
3. Uses the information from t+1 and the decisions made in t (for the whole Δt) to calculate the penalties. 
4. Updates the SoC and voltages of storage assets, providing guarantees that they are within the safe limits. 
"""
function exogenous!(env::MCES_Env, W::Exogenous_Batch)
    #Extracting components from the Env
    ev = env.ev           # Electric Vehicle (EV)
    bess = env.bess       # Battery Energy Storage System (BESS)
    hp = env.hp           # Heat Pump
    tess = env.tess       # Thermal Energy Storage System (TESS)
    stc = env.stc         # Solar Thermal Collector (STC)
    pei = env.pei         # Power Electronic Interface (PEI)
    

    env.load_e = W.load_e        # Electrical load
    env.load_th = W.load_th      # Thermal load
    env.pv = W.pv                # Photovoltaic power
    env.st = min(W.pv * stc.η, stc.th_max)  # Solar thermal power

    env.λ_buy = W.λ_buy          # Price to buy electricity
    env.λ_sell = W.λ_sell        # Price to sell electricity

    ev.p_drive = W.p_drive       # Power demanded of the EV for driving [kW]

    ####################################################
    # Updating EV Location    
    ev.departs = ev.γ == 1 && W.γ_ev == 0
    ev.γ = W.γ_ev
    ####################################################

    # Update Power going to/from the Grid
    p_grid = env.load_e + hp.e - env.pv - bess.p - ev.bat.p * ev.γ * ev.in_EMS
    grid_η = p_grid > 0 ? 1/pei.η : pei.η
    env.grid, env.ξp_grid = clamp_with_amount(p_grid*grid_η, - pei.p_max, pei.p_max, abs = true)
    
    # Update SOC of batteries
    env.ξsoc_bess = update_bat_soc!(bess, env)
    update_voltage!(bess)

    if ev.in_EMS
        env.ξsoc_ev = update_ev_soc!(ev, env)
        update_voltage!(ev.bat)
    else
        env.ξsoc_ev = 0f0
    end

    # Update TESS SoC
    tess.p, env.ξp_tess = clamp_with_amount(env.load_th - env.st - hp.th, -tess.p_max, tess.p_max, abs = true)
    env.ξsoc_tess = update_tess_soc!(tess, env)

    env.t += Int32(1)    # Increment general timestep
    env.t_ep += Int32(1) # Increment episode timestep

    nothing
end




@info "Exogenous functionality added"
