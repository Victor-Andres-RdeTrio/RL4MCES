####################################################################################################
# Exogenous Data
abstract type Exogenous end


"""
    Exogenous_Batch <: Exogenous

Represents a single timestep of exogenous (external) data, required for the MCES environment transition.

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
- Prices may be negative in some electricity markets, reflecting grid congestion or oversupply conditions.
- The EV presence indicator (`γ_ev`) determines whether EV-related calculations are performed.
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

end



"""
    Exogenous_BatchCollection{V,B} <: Exogenous

Represents a collection of exogenous (external) data for multiple timesteps, required for the MCES environment transition.

# Type Parameters
- `V`: Vector type for storing numerical data, defaults to `Vector{Float32}`.
- `B`: Vector type for storing boolean data, defaults to `Vector{Bool}`.

# Fields
- `load_e::V = Float32[]`: Vector of electrical loads in kilowatts [kW]. Each element must be non-negative.
- `load_th::V = Float32[]`: Vector of thermal loads in kilowatts [kW]. Each element must be non-negative.
- `pv::V = Float32[]`: Vector of photovoltaic power generation in kilowatts [kW]. Each element must be non-negative.
- `λ_buy::V = Float32[]`: Vector of prices to buy electricity in euros per kilowatt-hour [€/kWh].
- `λ_sell::V = Float32[]`: Vector of prices to sell electricity in euros per kilowatt-hour [€/kWh].
- `p_drive::V = Float32[]`: Vector of power used for driving the EV in kilowatts [kW]. Each element must be non-negative.
- `γ_ev::B = Bool[]`: Vector indicating whether the Electric Vehicle (EV) is present in the house for each timestep.
- `last_timestep::Int32 = Int32(96)`: The last timestep included in the collection, defaults to 96 (representing a typical day with 15-minute intervals).

# Notes
- All vectors must have at least `last_timestep` elements for proper indexing.
- All power values (load_e, load_th, pv, p_drive) must be in kilowatts [kW].
- Price values (λ_buy, λ_sell) must be in euros per kilowatt-hour [€/kWh].
- This structure is typically used for loading and processing time series data for energy system simulations.
- The default values create empty vectors.
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

"""
    subset_exog(exo::Exogenous_BatchCollection, s::Int, e::Int)

Creates a subset of an `Exogenous_BatchCollection` from the specified start and end indices. 

# Arguments
- `exo::Exogenous_BatchCollection`: The source collection to subset.
- `s::Int`: The start index (inclusive). Will be clamped to a minimum of 1.
- `e::Int`: The end index (inclusive). Will be clamped to the maximum available length.

# Returns
- A new `Exogenous_BatchCollection` containing only the elements from index `s` to `e`.

# Details
1. Ensures the start index is at least 1.
2. Ensures the end index doesn't exceed the length of any data vectors.
3. Creates a new collection with the subset of data and updates the `last_timestep` accordingly.
"""
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


"""
    exog_to_dict(exog::Exogenous_BatchCollection)

Convert an Exogenous_BatchCollection object to a dictionary.

# Arguments
- `exog::Exogenous_BatchCollection`: The exogenous batch collection to convert.

# Returns
- `Dict{String, Any}`: A dictionary where keys are field names (as strings) and values are the corresponding field values.
"""

function exog_to_dict(exog::Exogenous_BatchCollection)
    dict = Dict{String, Any}()

    for field in fieldnames(typeof(exog))
        dict[string(field)] = getfield(exog, field)
    end
    
    return dict
end

####################################################################################################
# Exogenous Update


"""
    exogenous!(env::MCES_Env, W::Exogenous_BatchCollection)

Updates the environment with exogenous information for the current timestep from a collection. Must take place after the agent has made its decisions.

# Arguments
- `env::MCES_Env`: The MCES environment object.
- `W::Exogenous_BatchCollection`: Collection of exogenous data for multiple timesteps.

# Returns
- `nothing`: The function updates the environment object in-place.

# Details
1. Verifies that the current timestep doesn't exceed the available data.
2. Extracts the data for the current timestep from the collection.
3. Creates an `Exogenous_Batch` with the extracted data.
4. Dispatches to `exogenous!(env::MCES_Env, W::Exogenous_Batch)` for processing.

# Notes
- The first timestep is `env.t = 1`.
- Indexing starts at 1 (first element of each field).
- After processing, the environment transitions to the next timestep.
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

Updates the environment with exogenous information for a single timestep.

# Arguments
- `env::MCES_Env`: The MCES environment object.
- `W::Exogenous_Batch`: Object containing exogenous data for a single timestep.

# Returns
- `nothing`: The function updates the environment object in-place.

# Details
1. Updates the environment with electrical load, thermal load, PV generation, pricing, etc.
2. Calculates solar thermal power based on PV power and efficiency.
3. Updates the EV status and records departure events.
4. Calculates power flow to/from the grid and applies efficiency adjustments.
5. Updates state of charge (SoC) for batteries and thermal storage.
6. Updates battery voltages.
7. Increments the timestep and episode counters.

# Effects
- Updates multiple fields in the environment: loads, generation, prices, penalties, and component states.
- Advances the simulation to the next timestep.
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
