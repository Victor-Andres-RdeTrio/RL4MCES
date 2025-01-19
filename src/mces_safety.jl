function create_safety_model!(env::MCES_Env; sf = 1.05f0)
    # Select optimizer based on model type
    env.simple_projection == true && return nothing

    @info "Created Safety Projection JuMP Model"
    model_type = env.model_type
    if model_type == "NL"
        model = Model(Ipopt.Optimizer)
        set_attribute(model, "max_wall_time", 5.0)
        set_attribute(model, "max_iter", 300)
    elseif model_type == "MILP"
        model = Model(HiGHS.Optimizer)
        # set_attribute(model, "time_limit", 3.0)
    else
        error("Invalid model type. Use 'NL' for nonlinear or 'MILP' for mixed-integer linear programming.")
    end

    set_silent(model)

    # Extracting info from Environment
    b = env.bess
    ev_b = env.ev.bat
    hp = env.hp
    η_grid = env.pei.η
    grid_max = env.pei.p_max
    tess = env.tess
    soc_ev_lim = (ev_b.soc_min, ev_b.soc_max)
    soc_b_lim = (b.soc_min, b.soc_max)
    soc_tess_lim = (tess.soc_min, tess.soc_max)
    Δt = env.Δt
    γ_ev = 1f0
    

    # Variables
    @variables(model, begin
            0.0 <= p_hp_e <= hp.e_max
            -grid_max / sf <= p_grid <= grid_max / sf
            soc_ev_lim[1] <= soc_ev <= soc_ev_lim[2]
            soc_b_lim[1] <= soc_bess <= soc_b_lim[2]
            soc_tess_lim[1] * sf <= soc_tess <= soc_tess_lim[2] / sf
            -b.i_max <= i_bess <= b.i_max
            -ev_b.i_max <= i_ev <= ev_b.i_max
            ε_grid  # Slack variable for electric balance
            ε_tess  # Slack variable for thermal balance
        end)

    if model_type == "NL"
        @variables(model, begin
            -ev_b.p_max <= p_ev <= ev_b.p_max
            -b.p_max <= p_bess <= b.p_max
            -tess.p_max / sf <= p_tess <= tess.p_max / sf
        end)
    elseif  model_type == "MILP" 
        @variables(model, begin
            0.0 <= p_ev_pos <= ev_b.p_max
            0.0 <= p_ev_neg <= ev_b.p_max
            0.0 <= p_bess_pos <= b.p_max
            0.0 <= p_bess_neg <= b.p_max
            0.0 <= p_tess_pos <= tess.p_max / sf
            0.0 <= p_tess_neg <= tess.p_max / sf
            bess_pos, Bin
            bess_neg, Bin
            ev_pos, Bin
            ev_neg, Bin
            tess_pos, Bin
            tess_neg, Bin
        end)
    else
        error("Invalid Model Type")
    end

    # Parameters (updated every timestep)
    @variables(model, begin
        ϟp_ev in Parameter(0.0)
        ϟp_bess in Parameter(0.0)
        ϟp_hp_e in Parameter(0.0)
        load_e in Parameter(0.0)
        load_th in Parameter(0.0)
        pv in Parameter(0.0)
        st in Parameter(0.0)
        soc_bess_0 in Parameter(0.0)
        soc_ev_0 in Parameter(0.0)
        soc_tess_0 in Parameter(0.0)
    end)

    # Electrical Constraints
    if model_type == "NL"
        @variables(model, begin
        v_bess in Parameter(0.0)
        v_ev in Parameter(0.0)
        # ε_soc_ev  # Slack variable for soc ev
        # ε_soc_bess  # Slack variable for soc bess
        ε_soc_tess  # Slack variable for soc tess
        end)

        @constraints(model, begin
        e_balance, p_grid * (ifelse(p_grid >= 0, η_grid, 1/η_grid)) + ε_grid == load_e + p_hp_e - pv - p_bess - γ_ev * p_ev * env.ev.in_EMS
        soc_bess_dyn, soc_bess == soc_bess_0 - i_bess * Δt / (b.q * 3600)
        p_bess_eq, p_bess == v_bess * i_bess * b.Np * b.Ns * (ifelse(i_bess >= 0, b.η_c, 1/b.η_c)) / 1000
        soc_ev_dyn, soc_ev == soc_ev_0 - i_ev * Δt / (ev_b.q * 3600)
        p_ev_eq, p_ev == v_ev * i_ev * ev_b.Np * ev_b.Ns * (ifelse(i_ev >= 0, ev_b.η_c, 1/ev_b.η_c)) / 1000
        end)

    elseif model_type == "MILP" 
        v_ev = 3.6
        v_bess = 3.6

        @constraints(model, begin
            e_balance, p_grid + ε_grid == load_e + p_hp_e - pv - p_bess_pos + p_bess_neg - γ_ev * (p_ev_pos - p_ev_neg) * env.ev.in_EMS
            bess_bidir, bess_pos + bess_neg == 1
            ev_bidir, ev_pos + ev_neg == 1
            p_bess_pos_eq, p_bess_pos == v_bess * i_bess * b.Np * b.Ns * b.η_c / 1000
            p_bess_neg_eq, -p_bess_neg == v_bess * i_bess * b.Np * b.Ns / b.η_c / 1000
            soc_bess_dyn, soc_bess == soc_bess_0 - (i_bess * Δt) / (b.q * 3600)
            p_ev_pos_eq, p_ev_pos == v_ev * i_ev * ev_b.Np * ev_b.Ns * ev_b.η_c / 1000
            p_ev_neg_eq, -p_ev_neg == v_ev * i_ev * ev_b.Np * ev_b.Ns / ev_b.η_c / 1000
            soc_ev_dyn, soc_ev == soc_ev_0 - (i_ev * Δt) / (ev_b.q * 3600)
        end)
    else
        error("Invalid Model Type")
    end

    # Thermal Constraints
    if model_type == "NL"
        @constraints(model, begin
            th_balance, p_tess + ε_tess == load_th - st - p_hp_e * hp.η
            soc_tess_dyn, soc_tess_0 - soc_tess + ε_soc_tess == p_tess * (Δt / (tess.q * 3600)) * (ifelse(p_tess >= 0, 1/tess.η, tess.η))
        end)

    elseif model_type == "MILP" 
        @constraints(model, begin
            tess_bidir, tess_pos + tess_neg == 1
            p_tess_bidir_pos, p_tess_pos <= tess.p_max * tess_pos / sf
            p_tess_bidir_neg, p_tess_neg <= tess.p_max * tess_neg / sf
            th_balance, p_tess_pos - p_tess_neg + ε_tess == load_th - st - p_hp_e * hp.η
            soc_tess_dyn, soc_tess_0 - soc_tess == (p_tess_pos / tess.η - p_tess_neg * tess.η) * (Δt / (tess.q * 3600))
        end)
    else
        error("Invalid Model Type")
    end

    # Objective function
    if model_type == "NL"
        @objective(model, Min,
            (p_ev - ϟp_ev)^2 + (p_bess - ϟp_bess)^2 + (p_hp_e - ϟp_hp_e)^2 +
            1e7 * (ε_grid^2 + ε_tess^2 + ε_soc_tess^2)  # Penalty for constraint violations
        )
    elseif model_type == "MILP"
        @objective(model, Min,
            (p_ev_pos - p_ev_neg - ϟp_ev)^2 + (p_bess_pos - p_bess_neg - ϟp_bess)^2 + (p_hp_e - ϟp_hp_e)^2 +
            1000 * (ε_grid^2 + ε_tess^2)  # Penalty for constraint violations
        )
    else
        error("Invalid Model Type")
    end

    env.safety_model = model
    return nothing
end

@inline function batt_coefficients(batt::Battery)
    pos_coeff = batt.v * batt.Np * batt.Ns * batt.η_c / 1000
    neg_coeff = batt.v * batt.Np * batt.Ns / (batt.η_c * 1000)
    return pos_coeff, neg_coeff
end

@inline function update_model!(env::MCES_Env)
    env.simple_projection == true && return nothing
    model = env.safety_model
    # Update raw decisions.
    set_parameter_value(model[:ϟp_ev], env.p_ev_raw)
    set_parameter_value(model[:ϟp_bess], env.p_bess_raw)
    set_parameter_value(model[:ϟp_hp_e], env.p_hp_e_raw)

    # Update other parameters.
    set_parameter_value(model[:load_e], env.load_e)
    set_parameter_value(model[:load_th], env.load_th)
    set_parameter_value(model[:pv], env.pv)
    set_parameter_value(model[:st], env.st)
    set_parameter_value(model[:soc_bess_0], env.bess.soc)
    set_parameter_value(model[:soc_ev_0], env.ev.bat.soc)
    set_parameter_value(model[:soc_tess_0], env.tess.soc)
    # set_parameter_value(m[:γ_ev], Float32(env.ev.γ)) # Not used currently, it is safer if the EV is always assumed to be in the EMS. 

    if env.model_type == "MILP"
        bess_v_pos, bess_v_neg = batt_coefficients(env.bess)
        set_normalized_coefficient(model[:p_bess_pos_eq], model[:i_bess], -bess_v_pos)
        set_normalized_coefficient(model[:p_bess_neg_eq], model[:i_bess], -bess_v_neg)

        ev_v_pos, ev_v_neg = batt_coefficients(env.ev.bat)
        set_normalized_coefficient(model[:p_ev_pos_eq], model[:i_ev], -ev_v_pos)
        set_normalized_coefficient(model[:p_ev_neg_eq], model[:i_ev], -ev_v_neg)
    
    elseif env.model_type == "NL"
        set_parameter_value(model[:v_bess], env.bess.v) 
        set_parameter_value(model[:v_ev], env.ev.bat.v)
    else
        error("Invalid Model Type")
    end

    return nothing
end

@inline function warm_start_model!(model::JuMP.Model, p_ev, p_bess, p_hp_e; model_type::String = "NL")
    if model_type == "NL"
        set_start_value(model[:p_ev], p_ev)
        set_start_value(model[:p_bess], p_bess)
    else
        p_ev >= 0 ? set_start_value(model[:p_ev_pos], p_ev) : set_start_value(model[:p_ev_neg], -p_ev)
        p_bess >= 0 ? set_start_value(model[:p_bess_pos], p_bess) : set_start_value(model[:p_bess_neg], -p_bess)
    end

    set_start_value(model[:p_hp_e], p_hp_e)
    set_start_value(model[:ε_grid], 0.0)
    set_start_value(model[:ε_tess], 0.0)

    return nothing
end

function use_projection!(env::MCES_Env)
    env.simple_projection = false
end


"""
    valid_actions(env::MCES_Env, decision)
    
Projects the proposed actions into valid actions (power levels) for the current state of the environment.
  
# Args:
    * `env`: The MCES environment object.
    * `decision`: An array containing proposed power levels for EV, BESS, and Heat Pump (in that order).
  
# Returns:
    A list of valid power levels for EV, BESS, and Heat Pump (electric).
""" 
function valid_actions(dec, env::MCES_Env; simple = false)

    p_ev_valid = limit_battery_power(dec[1], env.ev.bat, env.Δt, simple)
    p_bess_valid = limit_battery_power(dec[2], env.bess, env.Δt, simple)
    p_hp_e_valid = limit_hp_power(dec[3], env, simple)

    (p_ev_valid, p_bess_valid, p_hp_e_valid) # All in kW
end

@inline function valid_actions!(actions_buf, dec, env::MCES_Env; simple = false)
    
    if simple
        actions_buf[1] = limit_battery_power(dec[1], env.ev.bat, env.Δt, true)
        actions_buf[2] = limit_battery_power(dec[2], env.bess, env.Δt, true)
        actions_buf[3] = limit_hp_power(dec[3], env, true)
        return actions_buf
    end

    actions_buf[1] = limit_battery_power(dec[1], env.ev.bat, env.Δt, false)
    actions_buf[2] = limit_battery_power(dec[2], env.bess, env.Δt, false)
    actions_buf[3] = limit_hp_power(dec[3], env, false)

    m = env.safety_model
    warm_start_model!(m, actions_buf...; model_type = env.model_type)
    optimize!(m)
    
    st = termination_status(m)
    if is_solved_and_feasible(m, allow_almost = true) || st == MOI.ITERATION_LIMIT || st == MOI.TIME_LIMIT
        get_safe_decisions!(actions_buf, m; model_type = env.model_type)
        return actions_buf
    end

    @warn "Solver terminated in timestep $(env.t). Cause -> $st)" 
    
    return actions_buf
end

@inline function get_safe_decisions!(actions_buf::Vector{Float32}, model::JuMP.Model; model_type::String = "NL")
    if model_type == "NL"
        actions_buf[1] = JuMP.value(model[:p_ev])
        actions_buf[2] = JuMP.value(model[:p_bess])
    elseif model_type == "MILP"
        actions_buf[1] = JuMP.value(model[:p_ev_pos]) - JuMP.value(model[:p_ev_neg])
        actions_buf[2] = JuMP.value(model[:p_bess_pos]) - JuMP.value(model[:p_bess_neg])
    else
        error("Invalid model type")
    end

    actions_buf[3] = JuMP.value(model[:p_hp_e])

    return nothing
end
