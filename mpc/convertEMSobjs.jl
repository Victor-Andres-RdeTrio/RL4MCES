

"""
    convert(::STC, t::ElectroThermData)

Convert an `ElectroThermData` object to an `STC`(Solar Thermal Collector) object.

# Arguments
- `t::ElectroThermData`: The input data to convert.

# Returns
- `STC`: The converted `STC` object.
"""
function convert(::Type{STC}, t::ElectroThermData)
    STC(
        η = t.η, 
        th_max = t.RatedPower
    )
end

"""
    convert(::Type{HP}, t::ElectroThermData)

Convert an `ElectroThermData` object to an `HP` (HeatPump) object.

# Arguments
- `t::ElectroThermData`: The input data to convert.

# Returns
- `HP`: The converted `HP` object.
"""
function convert(::Type{HP}, t::ElectroThermData)
    HP(
        e = 0.1f0, 
        th = 0.45f0, 
        η = t.η, 
        e_max = t.RatedPower
    )
end

"""
    convert(::Type{TESS}, tess_data::TESSData)

Convert a `TESSData` object to a `TESS`(Thermal Energy Storage System) object.

# Arguments
- `tess_data::TESSData`: The input data to convert.

# Returns
- `TESS`: The converted `TESS` object.
"""
function convert(::Type{TESS}, tess_data::TESSData)
    TESS(
        soc = clamp(tess_data.SoC0, tess_data.SoCLim[1], tess_data.SoCLim[2]),
        soc_min = tess_data.SoCLim[1],
        soc_max = tess_data.SoCLim[2],
        p_max = tess_data.PowerLim[2],
        q = tess_data.Q,
        η = tess_data.η
    )
end

"""
    convert(::Type{PEI}, pei_data::peiData, η = 0.9)

Convert a `peiData` object to a `PEI`(Power Electric Interface) object.

# Arguments
- `pei_data::peiData`: The input data to convert.
- `η::Float64`: Efficiency value, default is 0.9.

# Returns
- `PEI`: The converted `PEI` object.
"""
function convert(::Type{PEI}, pei_data::peiData, η = 0.9)
    PEI(
        η = η,
        p_max = pei_data.RatedPower
    )
end

"""
    convert(::Type{Battery}, bess_data::BESSData)

Convert a `BESSData` object to a `Battery` object.

# Arguments
- `bess_data::BESSData`: The input data to convert.

# Returns
- `Battery`: The converted `Battery` object.
"""
function convert(::Type{Battery}, bess_data::BESSData)
    gen = bess_data.GenInfo
    Battery(
        soc = clamp(gen.SoC0, gen.SoCLim[1], gen.SoCLim[2]),
        soc_min = gen.SoCLim[1],
        soc_max = gen.SoCLim[2],
        q = gen.initQ,
        p = gen.P0,
        p_max = gen.PowerLim[2],
        i = 0.0,  
        i_max = 7.8,  
        v = 3.6, 
        v_rated = 3.6,
        v_max = gen.vLim[2],
        ocv = 0.0,  
        a = gen.OCVParam.ocvLine[1],
        b = gen.OCVParam.ocvLine[2],
        η_c = gen.η,
        Np = gen.Np,
        Ns = gen.Ns
    )
end


"""
    convert(::Type{EV{Battery}}, ev_data::EVData)

Convert an `EVData` object to an `EV{Battery}` (Electric Vehicle) object.

# Arguments
- `ev_data::EVData`: The input data to convert.

# Returns
- `EV{Battery}`: The converted `EV` object.
"""
function convert(::Type{EV{Battery}}, ev_data::EVData)
    battery = convert(Battery, ev_data.carBatteryPack)
    drive = ev_data.driveInfo
    EV{Battery}(
        bat = battery,
        γ = Bool(drive.γ[1]),    
        p_drive = drive.Pdrive[1],
        soc_dep = drive.SoCdep,
        departs = false  
    )
end

function convert(::Type{EV}, ev_data::EVData)
    convert(EV{Battery}, ev_data)
end

@info "Objects from EMS can be converted to RL Environment"