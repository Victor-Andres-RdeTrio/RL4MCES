using StatsBase
include("../mces_core.jl")

function crosscov_unequal(short_series, long_series; corr = false)
    n_short = length(short_series)
    n_long = length(long_series)
    @assert n_short <= n_long "The first argument must be the shortest series"
    
    max_lag = n_long - n_short
    ccf = Vector{Float32}(undef, max_lag + 1)

    func = corr ? crosscor : crosscov
    
    for lag in 0:max_lag
        segment = long_series[lag+1 : lag+n_short]
        ccf[lag+1] = func(short_series, segment, [0], demean = true)[]
        # ccf2[lag+1] = mean((short_series .- mean(short_series)).*(segment .-mean(segment)))
    end
    
    return ccf
end

function plot_cc(short, long; corr::Bool = false, window::Int = 7)
    #day 1
    window = window # days that will be used for the long series
    days = [17, 23, 68]
    
    p = plot(
        xlabel = "Lag [days]", 
        ylabel = corr ? "Cross-Correlation" : "Cross-Covariance",
        dpi = 900
    )

    for i in days
        s = short[96*(i-1) + 1 : 96*(i)] # Collect 1 day
        l =  long[96*(i-window) + 1 : 96*(i)]
        ccv = crosscov_unequal(s, l, corr = corr)
        plot!(p,(-(length(ccv)-1):0)/96, ccv, label = "day_$i")
    end

    display(p)
    p
end
 

function cc_all_days(short, long; corr::Bool = false, window::Int = 7, label::String = "", plot::Bool = true)
    window = window # days that will be used for the long series
    days = window : div(length(long),96)
    
    ccv = Matrix{Float32}(undef, length(days), max(1, (window - 1)*96 + 1))

    for i in days
        s = short[96*(i-1) + 1 : 96*(i)] # Collect 1 day
        l =  long[96*(i-window) + 1 : 96*(i)]
        ccv[i - window + 1, :] = crosscov_unequal(s, l, corr = corr)
    end
    ccv_mean = vec(mean(ccv, dims = 1))

    if plot
        p = Plots.plot(
            title = "Mean for all days",
            xlabel = "Lag [days]", 
            ylabel = corr ? "Cross-Correlation" : "Cross-Covariance",
            dpi = 900,
            legend = true
        )
        Plots.plot!(p,(-(length(ccv_mean)-1):0)/96, ccv_mean, label = label)
        display(p)
    end

    ccv_mean
end

function analyze_crosscor(h_mpc::MCES_Hook, exog::Exogenous; corr::Bool = true, window::Int = 7 )
    decisions = ["p_hp_e", "p_bess", "p_ev"]
    
    # Variables to correlate against
    variables = [
        "load_e", "load_th", "pv", "grid", "λ_buy",
        "p_ev", "p_bess", "p_hp_e", "soc_ev", "soc_bess", "soc_tess", 
        "γ_ev", "p_drive","load_e - pv", "load_th - st", "t_ratio"
        # "λ_sell", "st"
    ]
    # λ_sell is proportional to λ_buy and thus has same correlations
    # st is proportional to pv and thus has same correlations

    exog_d = to_dict(exog)
    
    var_data = Dict{String, Vector{Float32}}()
    for var in variables
        if var in ["load_e", "load_th", "pv", "λ_buy", "λ_sell", "p_drive", "γ_ev"]
            var_data[var] = exog_d[var][1:exog.last_timestep]
        elseif var in ["p_ev", "p_bess", "p_hp_e", "soc_ev", "soc_bess", "soc_tess", "st", "grid"]
            var_data[var] = h_mpc.energy[var]
        elseif var == "load_e - pv"
            var_data[var] = var_data["load_e"] .- var_data["pv"]
        elseif var == "load_th - st"
            var_data[var] = var_data["load_th"] .- h_mpc.energy["st"]
        elseif var == "t_ratio"
            var_data[var] = mod1.(h_mpc.energy["t"], 96)./96
        end
    end
    
    # Calculate cross-correlations
    results = Dict{String, Dict{String, Vector{Float32}}}()
    for decision in decisions
        results[decision] = Dict{String, Vector{Float32}}()
        for var in variables
            results[decision][var] = cc_all_days(
                h_mpc.energy[decision], var_data[var], corr=corr, 
                plot=false, window = window
            )
        end
    end
    
    return results
end

function plot_cross_corr(results; save_path="", autocorr::Bool = false)
    decisions = collect(keys(results))
    all_variables = sort(collect(keys(results[decisions[1]])))
    # There was too much information per plot, so the variables are now spread through 3 plots. 
    var_groups = [all_variables[1:5], all_variables[6:10], all_variables[11:end]]
    
    # High contrast color palette
    colors = [:red, :blue, :green, :gold, :purple, :cyan, :magenta, :lime, :teal, :brown, :pink]
    
    plots = []
    for decision in decisions
        for var_group in var_groups
            p = plot(
                title = "Cross-Correlation for $decision", 
                xlabel="Lag [days]", 
                ylabel="Cross-Correlation coefficient", 
                dpi=2000, 
                legend=:outerright, 
                legendfontsize=8)
            for (i, var) in enumerate(var_group)
                if !autocorr
                    decision == var && continue
                end
                plot!(p, (-(length(results[decision][var])-1):0)/96, results[decision][var], label=var, color=colors[mod1(i, length(colors))])
            end
            push!(plots, p)
        end
    end
    
    plot_combined = plot(plots..., layout = (3, length(decisions)), size = (1500, 400 * length(decisions)), margin = 4mm, link=:y)
    display(plot_combined)
    
    if !isempty(save_path)
        Plots.savefig(plot_combined, save_path)
    end
    
    return plot_combined
end

function plot_mean_cross_corr(results; save_path="")
    decisions = keys(results)
    all_variables = sort(collect(keys(first(values(results)))))
    var_groups = [all_variables[1:5], all_variables[6:10], all_variables[11:end]]
    
    # Calculate mean correlations across all decisions
    mean_correlations = Dict{String, Vector{Float32}}()
    for var in all_variables
        mean_correlations[var] = mean([abs.(results[decision][var]) for decision in decisions])
    end
    
    # High contrast color palette
    colors = [:red, :blue, :green, :gold, :purple, :cyan, :magenta, :lime, :teal, :brown, :pink]
    
    plots = []
    for (group_index, var_group) in enumerate(var_groups)
        p = plot(
            title = "Mean Absolute Cross-Correlation (Group $(group_index))", 
            xlabel="Lag [days]", 
            ylabel="Mean Cross-Correlation coefficient", 
            dpi=2000, 
            legend=:outerright, 
            legendfontsize=8
        )
        for (i, var) in enumerate(var_group)
            plot!(p, (-(length(mean_correlations[var])-1):0)/96, mean_correlations[var], 
                  label=var, color=colors[mod1(i, length(colors))])
        end
        push!(plots, p)
    end
    
    plot_combined = plot(plots..., layout = (3, 1), size = (1000, 1200), margin = 4mm, link=:y)
    display(plot_combined)
    
    if !isempty(save_path)
        Plots.savefig(plot_combined, save_path)
    end
    
    return plot_combined
end

function create_heatmap(results)
    decisions = collect(keys(results))
    variables = collect(keys(results[decisions[1]]))
    
    heatmap_data = zeros(Float32, length(variables), length(decisions))
    for (i, decision) in enumerate(decisions)
        for (j, var) in enumerate(variables)
            decision == var && continue
            filtered = filter(x -> x <= 1, abs.(results[decision][var]))
            heatmap_data[j, i] = maximum(filtered)
        end
    end
    
    heatmap_plot = heatmap(decisions, variables, heatmap_data, 
                           color=:viridis, aspect_ratio=:equal,
                           title="Maximum Absolute Cross-Correlation",
                           size=(600, 1200), dpi=600,
                           xrotation=45, xticks=:all, yticks=:all)
    
    display(heatmap_plot)
    return heatmap_plot
end
