# Print in the REPL


function print_hook_REPL(hook::Union{MCES_Hook,MCES_Fast_Hook})
    println(
            UnicodePlots.lineplot(
                hook.episode_rewards,
                title="Total reward per episode",
                xlabel="Episode",
                ylabel="Rewards"))
end

###########################################################################
# General Utility Plots
Plots.theme(:dao)
#other options are here https://docs.juliaplots.org/latest/generated/plotthemes/#:dao

function plot_basic(a::Vector, start = 1, finish = length(a); smooth::Int = 0, label = "")
    b = Smoothing.binomial(a, smooth)
    plot(start:finish, b[start:finish], label = label, linewidth = 2, dpi = 600)
end


"""
    batches_to_ep(value_per_batch::Vector, batches_per_episode = 1) :: AbstractArray

Calculate the total loss per episode from individual losses per batch.

# Args:

- `value_per_batch`: A vector containing losses for each batch.
- `batches_per_episode`: Number of batches per episode. 

# Returns:

- `AbstractArray`: A matrix containing the total loss per episode.

# Details:
1. Checks if the length of `value_per_batch` is divisible by `batches_per_episode`.
2. Reshapes `value_per_batch` into a matrix with `batches_per_episode` columns.
3. Sums each of the rows.
"""
function batches_to_ep(value_per_batch::Vector, batches_per_episode = 1) :: AbstractArray
    @assert mod(length(value_per_batch), batches_per_episode) == 0 "Array size must be divisible by the number of batches per episode."
    loss_mat = reshape(value_per_batch, :, batches_per_episode)
    sum(loss_mat, dims = 2)
end

function plot_dict(dict::Dict, keys::Vector{String}; start = 1, finish = length(dict[keys[1]]), 
    smooth::Int = 0, layout = (2, round(Int, length(keys)/2, RoundUp)))

    columns = round(Int, length(keys)/2, RoundUp)
    p = plot(layout = layout, size = (300*columns, 800))

    for (i, key) in enumerate(keys)
        smoothed_values = Smoothing.binomial(dict[key], smooth)
        fin = min(finish, length(dict[key])) 
        plot!(p[i], start:fin, smoothed_values[start:fin], linewidth = 2, dpi = 800, title = key, label = "",
        titlefont = font(12), guidefont = font(10), tickfont = font(8))
    end
    
    display(p)
end

function plot_hooks(array; funct = plot_reward, subs = 6, kwargs...)
    plots = []

    for (i, hook) in enumerate(array)
        p = funct(hook; kwargs...)
        plot!(p, linewidth = 2, dpi = 800, title = "Seed: $i", label = "",
        titlefont = font(12), guidefont = font(10), tickfont = font(8))
        push!(plots, p)
        i >= subs && break
    end

    columns = cld(length(plots), 2)
    plot(plots..., layout = (2, columns), size = (300 * columns, 800))
end

##############################################################
# Statistic Plots

function cdf_plot(data::Vector)
    
    ecdf_prob = ecdf(data)
    
    x = 0:0.0001:1
    # Plot the CDF
    p = plot(x, ecdf_prob.(x), title="Empirical Cumulative Distribution Function", xlabel="Value", ylabel="CDF")
end

function plot_boxplots(
    data_arrays::Vector{Vector{T}}; 
    labels::Union{Vector{String}, Nothing} = nothing,
    colors=[:green, :violetred, :blue, :orange, :red, :teal],
    title="Multiple Dataset Comparison",
    size=(1300, 600),
    xlim::Union{Tuple{Real, Real}, Nothing} = nothing,
    ylim::Union{Tuple{Real, Real}, Nothing} = nothing,
    save_path::String = "",
    threshold::Real = 0.0,
    alpha = 0.65
    ) where {T}

    # Set labels if not provided
    if isnothing(labels)
        labels = ["Dataset $i" for i in 1:length(data_arrays)]
    elseif length(labels) != length(data_arrays)
        throw(ArgumentError("Number of labels must match number of arrays"))
    end

    ymax = maximum(maximum.(data_arrays))
    ymin = minimum(minimum.(data_arrays))

    # Initialize the base plot
    p = StatsPlots.boxplot(
        size=size,
        title= iszero(threshold) ? title : title * "(Threshold = $threshold)",
        grid=false,
        legend=:outertopright,
        legend_font_pointsize = 10
    )

    # Process each dataset and add boxplots
    for (i, data) in enumerate(data_arrays)
        color = colors[mod1(i, length(colors))]

        # Separate successes and failures
        successes = filter(x -> x >= threshold, data)

        # Add boxplot for current dataset
        StatsPlots.boxplot!(p,
            Float64.(successes),
            color=color,
            fillalpha=alpha,
            label= iszero(threshold) ? 
            "$(labels[i])" : 
            "$(labels[i]) B(n=$(length(data)), p=$(round(length(successes) / length(data), digits=3)))",
            whisker_range=1.5,
            whisker_width=:half,
            outliers=true,
            notch = false
        )
    end

    # Customize plot
    plot!(p,
        xlabel="Dataset",
        ylabel="Performance",
        xticks=(1:length(data_arrays), labels),
        yticks = (min(ymin, 0) : 0.1 : ymax * 1.2),
        ylim=(min(ymin, 0), ymax * 1.05),
        margin = 10mm,
        grid = :y,
        gridalpha = 0.25,
    )

    # Check if xlim or ylim are provided to create a second plot
    if isnothing(xlim) && isnothing(ylim)
        # Only one plot if no xlim or ylim provided
        final_plot = p
    else
        # Create a second plot with limited xlim and ylim
        p_limited = deepcopy(p)
        !isnothing(xlim) && plot!(p_limited, xlim=xlim)
        !isnothing(ylim) && plot!(p_limited, ylim=ylim, yticks=round.(range(ylim[1], ylim[2], length=10), digits=4))
        
        # Combine the two plots in a (1,2) layout
        final_plot = plot(p, p_limited, layout=(1,2), size=size)
    end

    # Save the plot if a path is provided
    !isempty(save_path) && savefig(final_plot, save_path)
    
    return final_plot
end


function plot_boxplots(
    data_dict::Dict;
    colors=[:green, :violetred, :blue, :orange, :red, :teal],
    title="Multiple Dataset Comparison",
    size=(1300, 600),
    xlim::Union{Tuple{Real, Real}, Nothing} = nothing,
    ylim::Union{Tuple{Real, Real}, Nothing} = nothing,
    save_path::String = "",
    threshold::Real = 0.0
) 
    # Create key-value pairs and sort by key
    sorted_pairs = sort(collect(data_dict), by = first)
    
    # Unzip the sorted pairs into separate arrays
    sorted_keys, data_arrays = first.(sorted_pairs), last.(sorted_pairs)
   
    return plot_boxplots(
        data_arrays;
        labels=string.(round.(sorted_keys, digits = 4)),
        colors=colors,
        title=title,
        size=size,
        xlim=xlim,
        ylim=ylim,
        save_path=save_path,
        threshold=threshold
    )
end

function plot_hyperopt_boxplots(
    hyperopt::Union{Hyperoptimizer, Vector{<:Hyperoptimizer}};
    save_dir::String,
    correct::Bool = false,
    plot_kwargs...
)
    hops = hyperopt isa Vector ? hyperopt : [hyperopt]
    parameters = hops[1].params
    mkpath(save_dir)
    
    for param in parameters
        result_dict = analize_hyperopt(hops, param, correct=correct)
        save_path = joinpath(save_dir, "$(String(param))_boxplot.svg")
        title = length(hops) == 1 ? "Distribution of results for parameter: $(param)" : 
                                   "Combined distribution of results for parameter: $(param)"
        
        plot_boxplots(
            result_dict;
            title=title,
            save_path=save_path,
            plot_kwargs...
        )
    end
end

function log_boxplots(
    data_arrays::Vector{Vector{T}};
    labels::Union{Vector{String}, Nothing} = nothing,
    colors=[:green, :violetred, :blue, :lawngreen, :red, :purple],
    title="Multiple Dataset Comparison",
    size=(800, 600),
    save_path::String = "",
    ) where {T}

    # Set labels if not provided
    if isnothing(labels)
        labels = ["Dataset $i" for i in 1:length(data_arrays)]
    elseif length(labels) != length(data_arrays)
        throw(ArgumentError("Number of labels must match number of arrays"))
    end

    ymax = maximum(maximum.(data_arrays))
    ymin = minimum(minimum.(data_arrays))

    # Initialize the base plot
    p = StatsPlots.boxplot(
        size=size,
        title=title,
        grid=false,
        legend=:outertopright,
        yscale = :ln
    )

    # Process each dataset and add boxplots
    for (i, data) in enumerate(data_arrays)
        color = colors[mod1(i, length(colors))]

        # Add boxplot for current dataset
        StatsPlots.boxplot!(p,
            Float64.(data),
            color=color,
            fillalpha=0.75,
            label=labels[i],
            whisker_range=1.5,
            whisker_width=:half,
            outliers=true,
            notch=false,
        )
    end

    # Customize plot
    plot!(p,
        xlabel="Dataset",
        ylabel="Performance",
        xticks=(1:length(data_arrays), labels),
        # yticks = (min(ymin, 0) : 0.1 : ymax * 1.2),
        # ylim=(min(ymin, 0), ymax * 1.2),
        margin = 10mm,
        grid = :y,
        gridalpha = 0.25,
    )

    !isempty(save_path) && savefig(final_plot, save_path)
    
    return p
end


function plot_gamma_binomial( # Get labels for each dataset. 
    data_arrays::Vector, 
    x_eval, 
    h::Real;
    threshold::Real = 0.05,
    n_bootstrap::Int=100,
    colors=[:violetred, :blue, :green, :orange, :purple],
    title="Multiple Dataset Comparison"
    )
    
    p = plot(layout=(2,1), size=(800,1000), title = title)
    
    # Process each dataset
    for (i, data) in enumerate(data_arrays)
        # Separate successes and failures
        successes = filter(x -> x >= threshold, data)
        failures = filter(x -> x < threshold, data)
        
        color = colors[mod1(i, length(colors))]
        
        # Top plot: KDE for successes using existing bootstrap_kde function
        if !isempty(successes)
            density, lower, upper = bootstrap_kde(successes, x_eval, h; n_bootstrap=n_bootstrap)
            
            # Plot on first subplot
            plot!(p[1], x_eval, density, 
                 color=color, 
                 linewidth=2, 
                 label="Dataset $i KDE")
            
            plot!(p[1], x_eval, [lower upper], 
                 fillrange=upper,
                 fillalpha=0.2,
                 fillcolor=color,
                 linealpha=0,
                 label="Dataset $i 95% CI")
        end
        
        # Bottom plot: Binomial approximation
        n = length(data)
        p_success = length(successes) / n
        
        # Create binomial distribution
        bin_dist = Binomial(n, p_success)
        x_bin = 0:n
        prob = pdf.(bin_dist, x_bin)
        
        # Plot on second subplot
        plot!(p[2], x_bin, prob,
             color=color,
             linewidth=2,
             label="Dataset $i Binomial (n =$n , p=$(round(p_success, digits=3)))")
    end
    
    # Customize top subplot
    plot!(p[1], xlabel="Value", 
         ylabel="Density",
         title="KDE of Successes (threshold = $threshold)",
         grid=false)
    
    # Customize bottom subplot
    plot!(p[2], xlabel="Number of Successes", 
         ylabel="Probability",
         title="Binomial Approximations",
         grid=false)
    
    return p
end

function plot_gamma_kde(
    data_arrays::Vector{Vector{T}}, 
    x_eval, 
    h::Real = 0.05; # bandwidth
    labels::Union{Vector{String}, Nothing} = nothing,
    threshold::Real = 0.05,
    n_bootstrap::Integer = 100,
    colors=[:green, :violetred, :blue, :orange, :purple],
    title="Multiple Dataset Comparison",
    xlims::Union{Tuple{Real,Real}, Nothing} = nothing,
    ylims::Union{Tuple{Real,Real}, Nothing} = nothing,
    quartiles::Bool = true
) where {T}

    # Set labels if not provided
    if isnothing(labels)
        labels = ["Series $i" for i in 1:length(data_arrays)]
    elseif length(labels) != length(data_arrays)
        throw(ArgumentError("Number of labels must match number of arrays"))
    end

    x_eval = filter(x -> x >= threshold , collect(x_eval))
    
    # Precompute all density estimates and quartiles
    density_data = []
    quartile_data = []
    for data in data_arrays
        successes = filter(x -> x >= threshold, data)
        if !isempty(successes)
            density, lower, upper = bootstrap_kde(successes, x_eval, h; n_bootstrap=n_bootstrap)
            push!(density_data, (density=density, lower=lower, upper=upper))
            
            # Calculate 1st and 3rd quartiles
            q1 = quantile(successes, 0.25)
            q3 = quantile(successes, 0.75)
            push!(quartile_data, (q1=q1, q3=q3))
        else
            push!(density_data, nothing)
            push!(quartile_data, nothing)
        end
    end
    
    # Create plot with 2 subplots
    p = plot(layout=(2,1), size=(800,800), margin = 10mm, title = title, grid = :y)
    
    # Plot both views
    for (subplot_idx, show_labels) in enumerate([true, false])
        for (i, data) in enumerate(density_data)
            isnothing(data) && continue
            
            color = colors[mod1(i, length(colors))]
            
            # Plot density
            plot!(p[subplot_idx], x_eval, data.density, 
                color=color, 
                linewidth=2, 
                legend = show_labels ? :outerright : false,
                label=show_labels ? labels[i] * " KDE" : "")
            
            # Plot confidence interval
            plot!(p[subplot_idx], x_eval, [data.lower data.upper], 
                fillrange=data.upper,
                fillalpha=0.2,
                fillcolor=color,
                linealpha=0,
                label=show_labels ? labels[i] * " 95% CI" : "")
            
            # Plot 1st and 3rd quartiles as dashed lines
            if quartiles
                q1, q3 = quartile_data[i].q1, quartile_data[i].q3
                vline!(p[subplot_idx], [q1, q1], 
                    linestyle=:dash, 
                    linewidth=1, 
                    color=color, 
                    label=show_labels ? labels[i] * " Q1" : "")
                vline!(p[subplot_idx], [q3, q3], 
                    linestyle=:dashdot, 
                    linewidth=1, 
                    color=color, 
                    label=show_labels ? labels[i] * " Q3" : "")
            end
        end
        
        # Add labels only to the first plot
        if show_labels
            plot!(p[subplot_idx], 
                xlabel="Performance Metric", 
                ylabel="Prob. Density",
                title="KDE of Successes (threshold = $threshold)",
                )
        end
        
        # Set limits for second plot
        if !show_labels  # second plot
            if !isnothing(xlims)
                plot!(p[subplot_idx], xlim=xlims)
            end
            if !isnothing(ylims)
                plot!(p[subplot_idx], ylim=ylims)
            end
        end
    end

    stats = calculate_binomial_stats(data_arrays, threshold, labels)
    for stat in stats
        println("Binomial Approx. $(stat.label) -> n: $(stat.n); p: $(round(stat.p, digits=4))")
    end
    
    return p
end

##########################################################################
# Plotting Rewards

function plot_reward(hook::Union{MCES_Hook,MCES_Moderate_Hook,MCES_Fast_Hook}; smooth::Int = 0)
    episodes = 1:length(hook.episode_rewards)
    smooth_rwds = Smoothing.binomial(hook.episode_rewards, smooth)
    p = Plots.plot(episodes, smooth_rwds,
            xlabel="Episodes", 
            ylabel="Acc. Reward of Episode", 
            linewidth=1.5,
            dpi = 600)
    
        # Horizontal Line   
    Plots.plot!(p, episodes, zeros(length(episodes)), linestyle=:dash)
    Plots.plot!(p, legend = false)
   
    display(p)
    p
end

function plot_reward(hook1::Union{MCES_Hook,MCES_Moderate_Hook,MCES_Fast_Hook}, 
                     hook2::Union{MCES_Hook,MCES_Moderate_Hook,MCES_Fast_Hook}; 
                     smooth::Int = 0, 
                     label1::String = "Hook 1", 
                     label2::String = "Hook 2")
    
    episodes1 = 1:length(hook1.episode_rewards)
    episodes2 = 1:length(hook2.episode_rewards)
    smooth_rwds1 = Smoothing.binomial(hook1.episode_rewards, smooth)
    smooth_rwds2 = Smoothing.binomial(hook2.episode_rewards, smooth)
    
    p = Plots.plot(episodes1, smooth_rwds1,
            xlabel="Episodes", 
            ylabel="Acc. Reward of Episode", 
            linewidth=1.5,
            label=label1,
            dpi=600)
    
    Plots.plot!(p, episodes2, smooth_rwds2,
            linewidth=1.5,
            label=label2)
    
    # Horizontal Line   
    Plots.plot!(p, episodes1, zeros(length(episodes1)), linestyle=:dash, label=nothing)
    
    Plots.plot!(p, legend = :bottomright)
   
    display(p)
    p
end


function plot_reward_dissect(hook::MCES_Hook; save_path = "", 
    params::Union{MCES_Params, Nothing} = nothing)
    dissection = hook.reward_dissect
    timesteps = 1:length(dissection["EV"])

    w_grid = isnothing(params) ? 1f0 : params.w_grid
    w_soc = isnothing(params) ? 1f0 : params.w_soc
    w_init_proj = isnothing(params) ? 1f0 : params.w_init_proj
    w_op_proj = isnothing(params) ? 1f0 : params.w_op_proj
    weighted = isnothing(params) ? "Unweighted " : "Weighted "

    p1 = plot(timesteps, w_grid * dissection["Grid"],
              xlabel = "Timesteps",
              ylabel = weighted * "Reward",
              linewidth = 2,
              label = "Grid",
              legend = :topleft,
              color = RGB(216/255, 100/255, 84/255),
              dpi = 600)
    
    plot!(p1, timesteps, w_soc * dissection["EV"], label = "EV", color = :green)
    plot!(p1, timesteps, w_init_proj * dissection["Initial Projection"], label = "Initial Projection", color = :blue)
    plot!(p1, timesteps, w_op_proj * dissection["Op. Projection"], label = "Operational Projection", color = :purple1)

    p2 = plot(timesteps, dissection["Clamped Reward"],
              xlabel = "Timesteps",
              ylabel = "Reward",
              linewidth = 2,
              label = "Weighted and Clamped Reward",
              legend = :topleft,
              color = RGB(14/255, 73/255, 133/255),
              dpi = 600)
    
    plot!(p2, timesteps, dissection["Real Reward"], label = "Unweighted Reward", color = :red)

    plot_combined = plot(p1, p2, layout = (1, 2), size = (1100, 400), margin = 8mm)

    if !isempty(save_path)
        Plots.savefig(plot_combined, save_path)
    end

    display(plot_combined)
end


function plot_avg_loss(hook::MCES_Hook, batches_per_episode::Int = 1; smooth::Int = 0, raw = false)
    loss_per_batch = raw ? hook.debug["raw_avg_loss"] : hook.debug["avg_loss"]
    loss_per_ep = batches_to_ep(loss_per_batch, batches_per_episode)
    loss_plot = plot_basic(vec(loss_per_ep), smooth = smooth);
    plot!(loss_plot, xlabel = "Optimization Step")
    plot!(loss_plot, ylabel = raw ? "Avg. Raw Actor Loss" : "Avg. Actor Loss")
end

function plot_reward_and_loss(hook::MCES_Hook, batches_per_episode::Int = 1; smooth::Int = 0)
    rwd_plot = plot_reward(hook, smooth = smooth);
    loss_plot = plot_avg_loss(hook, batches_per_episode, smooth = smooth);

    combined_plot = plot(rwd_plot, loss_plot, layout = (2,1));
    combined_plot
end

function plot_reward_and_loss(hook::MCES_Hook; loss_per_ep::AbstractArray, smooth::Int = 0)
    rwd_plot = plot_reward(hook, smooth = smooth);
    loss_plot = plot_avg_loss(loss_per_ep, smooth = smooth);

    combined_plot = plot(rwd_plot, loss_plot, layout = (2,1));
    combined_plot
end

##########################################################################
# Plot losses

function plot_debug_vpg(hook::MCES_Hook; start = 1, finish = length(hook.debug["actor_loss"]), smooth::Int = 0)
    d = hook.debug ::Dict{String,Vector{Float32}}
    keys =  ["avg_adv", "avg_loss", "raw_avg_loss", "actor_loss", 
    "critic_loss", "actor_norm", "critic_norm", "δ"]
    plot_dict(d, keys, start = start, finish = finish, smooth = smooth, layout = (2,4))
end

function plot_debug_a2c(hook::MCES_Hook; start = 1, finish = length(hook.debug["actor_loss"]), smooth::Int = 0)
    d = hook.debug ::Dict{String,Vector{Float32}}
    keys =  ["avg_adv", "avg_loss", "raw_avg_loss", "actor_loss", "critic_loss", 
    "entropy_loss", "actor_norm", "critic_norm", "δ"]
    plot_dict(d, keys, start = start, finish = finish, smooth = smooth, layout = (2,5))
end

function plot_debug_ppo(hook::MCES_Hook; start = 1, finish = length(hook.debug["actor_loss"]), smooth::Int = 0)
    d = hook.debug ::Dict{String,Vector{Float32}}
    keys =  ["avg_adv", "avg_loss", "raw_avg_loss", "critic_loss", 
    "entropy_loss", "actor_norm", "critic_norm", "δ", "clip_fracs", "approx_kl"]
    plot_dict(d, keys, start = start, finish = finish, smooth = smooth, layout = (2,5))
end

##########################################################################
# Plot Grid, EV, and AuxTESS
# Function to compute the combined grid and EV cost
function compute_grid_ev(hook::MCES_Hook, grid_weight::Real, ev_weight::Real; 
    combined::Bool = false, episodic::Bool = false, timesteps_per_day::Int = 96)

    dissection = hook.reward_dissect
    grid_cost = dissection["Grid"] .* grid_weight
    ev_cost = dissection["EV"] .* ev_weight
    combined_cost = grid_cost .+ ev_cost

    if episodic
        grid_cost = [sum(day) for day in partition(grid_cost, timesteps_per_day)]
        ev_cost = [sum(day) for day in partition(ev_cost, timesteps_per_day)]
        combined_cost = [sum(day) for day in partition(combined_cost, timesteps_per_day)]
    end

    return combined ? combined_cost : (grid_cost, ev_cost)
end

function compute_grid_cost(hook::MCES_Hook; episodic::Bool = false, timesteps_per_day::Int = 96)
    dissection = hook.reward_dissect
    grid_cost = dissection["Grid"]

    if episodic
        grid_cost = [sum(day) for day in partition(grid_cost, timesteps_per_day)]
    end

    return grid_cost
end

function compute_ev_cost(hook::MCES_Hook; episodic::Bool = false, timesteps_per_day::Int = 96)
    dissection = hook.reward_dissect
    ev_cost = dissection["EV"]

    if episodic
        ev_cost = [sum(day) for day in partition(ev_cost, timesteps_per_day)]
    end

    return ev_cost
end




function plot_grid_ev(hook::MCES_Hook; 
                      w_grid::Real = 1.0, 
                      w_ev::Real = 1.0, 
                      episodic::Bool = true,
                      save_path::String = "", 
                      dpi::Int = 600)
    
    grid_cost, ev_cost = compute_grid_ev(hook, w_grid, w_ev, episodic = episodic)
    timesteps = 1:length(grid_cost)

    p1 = plot(
        timesteps, -grid_cost, 
        xlabel = "Episodes", 
        ylabel = "Grid Cost [€]", 
        title = "Total Grid Cost for Episode", 
        linewidth = 2, 
        legend = false,
        color = RGB(216/255, 100/255, 84/255),
        dpi = dpi
    )

    plot!(p1, timesteps, zeros(length(timesteps)), color = :gray)
    
    p2 = plot(
        timesteps, -ev_cost, 
        xlabel = "Episodes", 
        ylabel = "EV Cost [-]", 
        title = "EV Penalty for departure", 
        linewidth = 2, 
        legend = false,
        color = :green,
        dpi = dpi
    )

    plot_combined = plot(p1, p2, layout = (1, 2), size = (1000, 400), bottom_margin = 5mm, left_margin = 5mm)
    
    if !isempty(save_path)
        Plots.savefig(plot_combined, save_path)
    end
    
    display(plot_combined)
end

function plot_grid_ev(hook1::MCES_Hook, hook2::MCES_Hook;
                      w_grid::Real = 1.0,
                      w_ev::Real = 1.0,
                      episodic::Bool = true,
                      save_path::String = "",
                      dpi::Int = 600,
                      label1::String = "Hook 1",
                      label2::String = "Hook 2")
    
    grid_cost1, ev_cost1 = compute_grid_ev(hook1, w_grid, w_ev, combined = false, episodic = episodic)
    grid_cost2, ev_cost2 = compute_grid_ev(hook2, w_grid, w_ev, combined = false, episodic = episodic)
    
    timesteps = 1:length(grid_cost1)

    avg_grid_cost1 = mean(-grid_cost1)
    avg_grid_cost2 = mean(-grid_cost2)

    # Plot the grid costs for hook 1 and hook 2
    p1 = plot(
        timesteps, -grid_cost1,
        xlabel = "Days",
        ylabel = "Grid Cost [€]",
        title = "Total Grid Exchange Cost per day",
        linewidth = 1.5,
        label = label1,
        color = RGB(216/255, 100/255, 84/255),  
        dpi = dpi,
        legend = :best,
        ylim = (0, 40),
        legendfontsize = 9,
        margin = 7mm
    )

    plot!(p1, timesteps, -grid_cost2, linewidth = 1.5, label = label2, color = RGB(84/255, 100/255, 216/255))  

    # Dotted average lines
    plot!(p1, [1, length(timesteps)], [avg_grid_cost1, avg_grid_cost1], 
          linestyle = :dashdot, 
          linewidth = 1.5, 
          label = label1 * " avg.", 
          color = RGBA(216/255, 100/255, 84/255, 0.5))  # Lighter version

    plot!(p1, [1, length(timesteps)], [avg_grid_cost2, avg_grid_cost2], 
          linestyle = :dashdot, 
          linewidth = 1.5, 
          label = label2 * " avg.", 
          color = RGBA(84/255, 100/255, 216/255, 0.5))  # Lighter version
    
    p2 = plot(
        timesteps, (-ev_cost1).^2,
        xlabel = "Days",
        ylabel = "(Desired SoC - SoC at Dep.)"* L"^2",
        title = "Sq. distance to desired SoC at departure",
        linewidth = 1.5,
        label = label1,
        color = :green,
        dpi = dpi,
        legend = :best,
        # ylim = (0, 65),
        legendfontsize = 11
    )

    plot!(p2, timesteps, (-ev_cost2).^2, linewidth = 1.5, label = label2, color = :darkorange2)

    plot_combined = plot(p1, p2, layout = (1, 2), size = (1000, 400), bottom_margin = 5mm, left_margin = 5mm)
    
    if !isempty(save_path)
        Plots.savefig(plot_combined, save_path)
    end
    
    display(plot_combined)
end

function plot_grid_cost(hook1::MCES_Hook, hook2::MCES_Hook;
                        episodic::Bool = true,
                        save_path::String = "",
                        dpi::Int = 600,
                        size = (800,600),
                        label1::String = "Hook 1",
                        label2::String = "Hook 2")
    
    grid_cost1= compute_grid_cost(hook1, episodic = episodic)
    grid_cost2= compute_grid_cost(hook2, episodic = episodic)
    
    timesteps = 1:length(grid_cost1)
    avg_grid_cost1 = mean(-grid_cost1)
    avg_grid_cost2 = mean(-grid_cost2)

    p1 = plot(
        timesteps, -grid_cost1,
        xlabel = "Day",
        ylabel = L"C_{grid}" *" [€]",
        title = "Total Grid Exchange Cost per day",
        linewidth = 1.5,
        label = label1,
        color = RGB(216/255, 100/255, 84/255),  
        dpi = dpi,
        size = size,
        legend = :outerright,
        ylim = (0, Inf),
        xticks = 1:5:length(grid_cost1),
        # legendfontsize = 9,
        margin = 7mm
    )

    plot!(p1, timesteps, -grid_cost2, linewidth = 1.5, label = label2, color = RGB(84/255, 100/255, 216/255))  

    plot!(p1, [1, length(timesteps)], [avg_grid_cost1, avg_grid_cost1], 
          linestyle = :dash, 
          linewidth = 1.5, 
          label = label1 * " avg.", 
          color = RGBA(216/255, 100/255, 84/255, 0.5))

    plot!(p1, [1, length(timesteps)], [avg_grid_cost2, avg_grid_cost2], 
          linestyle = :dash, 
          linewidth = 1.5, 
          label = label2 * " avg.", 
          color = RGBA(84/255, 100/255, 216/255, 0.5))

    if !isempty(save_path)
        Plots.savefig(p1, save_path)
    end

    return p1
end


function plot_ev_cost(hook1::MCES_Hook, hook2::MCES_Hook;
                      episodic::Bool = true,
                      save_path::String = "",
                      dpi::Int = 600,
                      size = (800,600),
                      label1::String = "Hook 1",
                      label2::String = "Hook 2")
    
    ev_cost1 = compute_ev_cost(hook1, episodic = episodic)
    ev_cost2 = compute_ev_cost(hook2, episodic = episodic)
    
    timesteps = 1:length(ev_cost1)

    p2 = plot(
        timesteps, (-ev_cost1).^2,
        xlabel = "Day",
        ylabel = L"p_{SoCDep}" *"  " * L"[-]",
        title = "Sq. distance to desired SoC at departure",
        linewidth = 1.5,
        label = label1,
        color = :green,
        dpi = dpi,
        margin = 7mm,
        size = size,
        xticks = 1:5:length(ev_cost1),
        legend = :outerright,
        # legendfontsize = 11
    )

    plot!(p2, timesteps, (-ev_cost2).^2, linewidth = 1.5, label = label2, color = :darkorange2)

    if !isempty(save_path)
        Plots.savefig(p2, save_path)
    end

    return p2
end

function plot_auxtess(hook1::MCES_Hook, hook2::MCES_Hook;
                      daylength::Integer = 96,
                      save_path::String = "",
                      dpi::Int = 600,
                      size = (800, 600),
                      label1::String = "Hook 1",
                      label2::String = "Hook 2")

    auxtess1 = compute_auxtess(hook1, daylength)[2]
    auxtess2 = compute_auxtess(hook2, daylength)[2]

    timesteps = 1:length(auxtess1)

    p1 = plot(
        timesteps, auxtess1,
        xlabel = "Days",
        ylabel = L"Aux_{TESS}" * " [s]",
        title = "Daily Value for " * L"Aux_{TESS}",
        linewidth = 1.5,
        label = label1,
        color = RGBA(153/255, 51/255, 204/255, 0.8) ,
        dpi = dpi,
        margin = 7mm,
        size = size,
        xticks = 1:5:length(auxtess1),
        legend = :outerright
    )

    # Add the second dataset to the plot
    plot!(
        p1, timesteps, auxtess2, linewidth = 1.5, 
        label = label2, color = RGBA(255/255, 204/255, 51/255, 0.8)
    )

    # Save the plot if a save path is specified
    if !isempty(save_path)
        Plots.savefig(p1, save_path)
    end

    return p1
end



##########################################################################
# Plot SoCs
function plot_soc(hook::MCES_Hook; save_path::String = "")
    energy = hook.energy
    timesteps = Int32.(energy["t"])
    n_days = div(length(timesteps), 96)  # Assuming 96 timesteps per day

    full_range = 1:length(timesteps)
    summer_day = div(n_days, 2) * 96 + 1 : div(n_days, 2) * 96 + 96
    winter_day = (n_days - 1) * 96 + 1 : n_days * 96

    # Plot 1: Full Range
    p1 = plot(timesteps[full_range], energy["soc_ev"][full_range] .* 100, 
             xlabel = "Timesteps", 
             ylabel = "SoC [%]",
             yticks = 0:10:120,
             ylims = (0, 120),
             linewidth = 2,
             label = L"SoC_{EV}",
             legend = :topleft,
             color = :green,  
             dpi = 600)
    
    plot!(p1, timesteps[full_range], energy["soc_bess"][full_range] .* 100, label = L"SoC_{BESS}", color = :blue)
    plot!(p1, timesteps[full_range], energy["soc_tess"][full_range] .* 100, label = L"SoC_{TESS}", color = :orange)  
    Plots.title!(p1, "State of Charge - Full Year")

    # Plot 2: Summer Day
    p2 = plot(timesteps[summer_day], energy["soc_tess"][summer_day] .* 100, 
             xlabel = "Timesteps", 
             ylabel = "SoC [%]",
             yticks = 0:10:120,
             ylims = (0, 120),
             linewidth = 2,
             label = L"SoC_{TESS}",
             legend = :topleft,
             color = :orange,
             dpi = 600)
    
    plot!(p2, timesteps[summer_day], energy["soc_bess"][summer_day] .* 100, label = L"SoC_{BESS}", color = :blue)
    plot!(p2, timesteps[summer_day], energy["soc_ev"][summer_day] .* 100, label = L"SoC_{EV}", color = :green)
    Plots.title!(p2, "State of Charge - Summer Day")

    # Plot 3: Winter Day
    p3 = plot(timesteps[winter_day], energy["soc_tess"][winter_day] .* 100, 
             xlabel = "Timesteps", 
             ylabel = "SoC [%]",
             yticks = 0:10:120,
             ylims = (0, 120),
             linewidth = 2,
             label = L"SoC_{TESS}",
             legend = :topleft,
             color = :orange,
             dpi = 600)
    
    plot!(p3, timesteps[winter_day], energy["soc_bess"][winter_day] .* 100, label = L"SoC_{BESS}", color = :blue)
    plot!(p3, timesteps[winter_day], energy["soc_ev"][winter_day] .* 100, label = L"SoC_{EV}", color = :green)
    Plots.title!(p3, "State of Charge - Winter Day")

    combined_plot = plot(p1, p2, p3, layout = (2, 2), size = (1400, 850), margin = 6mm)

    if !isempty(save_path)
        Plots.savefig(combined_plot, save_path)
    end
    
    display(combined_plot)
end

function plot_soc(hook1::MCES_Hook, hook2::MCES_Hook;
                    save_path::String = "",
                    label1::String = "Hook 1",
                    label2::String = "Hook 2",
                    day_to_plot::Integer = 1,
                    daylength::Integer = 96)

    energy1 = hook1.energy
    energy2 = hook2.energy
    timesteps = Int32.(energy1["t"])
    
    L = min(length(timesteps), 
            length(energy1["soc_bess"]),
            length(energy1["soc_ev"]),
            length(energy1["soc_tess"]),
            length(energy2["soc_bess"]),
            length(energy2["soc_ev"]),
            length(energy2["soc_tess"]))
    
    n_days = div(L, daylength)
    day_to_plot = clamp(day_to_plot, 1, n_days)
    day_range = ((day_to_plot - 1) * daylength + 1):(day_to_plot * daylength)

    function create_soc_plot(time_range, title)
        plot_timesteps = (1:length(time_range))/(daylength/24)
        
        p = plot(xlabel = "Hours",
                ylabel = "SoC [%]",
                # yticks = 0:5:105,
                ylims = (0, 105),
                title = title,
                legend = :outerright,
                legendfontsize = 10,
                dpi = 900)
        
        hook1_base = RGBA(208/255, 32/255, 144/255, 0.75)
        hook1_dark = RGBA(208/255, 32/255, 144/255, 1)
        hook1_bright = RGBA(208/255, 32/255, 144/255, 0.5)
        
        hook2_base = RGBA(84/255, 100/255, 216/255, 0.75)
        hook2_dark = RGBA(84/255, 100/255, 216/255, 1.5)
        hook2_bright = RGBA(84/255, 100/255, 216/255, 0.5)
        
        plot!(p, plot_timesteps, energy1["soc_bess"][time_range] .* 100,
              linewidth = 1.75,
              label = L"SoC_{BESS}" * " ($label1)",
              color = hook1_base,
              linestyle = :dashdot)

        plot!(p, plot_timesteps, energy2["soc_bess"][time_range] .* 100,
        linewidth = 1.75,
        label = L"SoC_{BESS}" * " ($label2)",
        color = hook2_base,
        linestyle = :dashdot)
        
        plot!(p, plot_timesteps, energy1["soc_ev"][time_range] .* 100,
              linewidth = 1.5,
              label = L"SoC_{EV}" * " ($label1)",
              color = hook1_bright)
        
        plot!(p, plot_timesteps, energy2["soc_ev"][time_range] .* 100,
              linewidth = 1.5,
              label = L"SoC_{EV}" * " ($label2)",
              color = hook2_bright)


        plot!(p, plot_timesteps, energy1["soc_tess"][time_range] .* 100,
        linewidth = 1.5,
        label = L"SoC_{TESS}" * " ($label1)",
        color = hook1_dark)
        
        plot!(p, plot_timesteps, energy2["soc_tess"][time_range] .* 100,
              linewidth = 1.5,
              label = L"SoC_{TESS}" * " ($label2)",
              color = hook2_dark)

        return p
    end
    p = create_soc_plot(day_range, "State of Charge Comparison - Day $day_to_plot")

    !isempty(save_path) && savefig(p, save_path)
    display(p)
    return p
end


function plot_soc_year(hook1::MCES_Hook, hook2::MCES_Hook; 
                       save_path1::String = "", 
                       save_path2::String = "", 
                       label1::String = "Hook 1", 
                       label2::String = "Hook 2", 
                       daylength::Int = 96)

    # Extract energy data for each hook
    energy1, energy2 = hook1.energy, hook2.energy
    total_days = div(length(energy1["soc_bess"]), daylength)
    time_range = 1:(total_days * daylength)
    days = (0:length(time_range)-1)/daylength

    # SoC data for Hook 1
    soc_bess1 = energy1["soc_bess"][time_range] .* 100
    soc_ev1 = energy1["soc_ev"][time_range] .* 100
    soc_tess1 = energy1["soc_tess"][time_range] .* 100

    # SoC data for Hook 2
    soc_bess2 = energy2["soc_bess"][time_range] .* 100
    soc_ev2 = energy2["soc_ev"][time_range] .* 100
    soc_tess2 = energy2["soc_tess"][time_range] .* 100

    # Plot for Hook 1
    p1 = plot(
    days, soc_bess1, label = "SoC BESS ($label1)", 
    xlabel = "Day", ylabel = "SoC [%]", title = "Yearly SoC for $label1",
    color = RGB(76/255, 0/255, 156/255), linewidth = 1.5, xticks = 0:5:total_days,
    fg_minor_grid = :white,
    grid = :x
    )
    plot!(p1, days, soc_ev1, label = "SoC EV ($label1)", color = RGBA(178/255, 45/255, 150/255, 0.5))
    plot!(p1, days, soc_tess1, label = "SoC TESS ($label1)", color = RGBA(51/255, 0/255, 25/255, 1.0))

    # Plot for Hook 2
    p2 = plot(
        days, soc_bess2, label = "SoC BESS ($label2)", 
        xlabel = "Day", ylabel = "SoC [%]", title = "Yearly SoC for $label2",
        color = RGBA(15/255, 65/255, 150/255, 1.0), linewidth = 1.5, xticks = 0:5:total_days,
        fg_minor_grid = :white,
        grid = :x
    )
    plot!(p2, days, soc_ev2, label = "SoC EV ($label2)", color = RGBA(36/255, 140/255, 200/255, 0.8))
    plot!(p2, days, soc_tess2, label = "SoC TESS ($label2)", color = RGBA(0/255, 0/255, 51/255, 1.0))

    # Save plots if paths are provided
    !isempty(save_path1) && savefig(p1, save_path1)
    !isempty(save_path2) && savefig(p2, save_path2)

    return p1, p2
end



##########################################################################
# Plot Actions and Unfiltered actions

function plot_decisions(hook::MCES_Hook, timesteps_range::Union{UnitRange{Int64}, Nothing} = nothing; raw::Bool = false, save_path="")
    energy = hook.energy
    timesteps = Int64.(energy["t"])
    
    if timesteps_range !== nothing
        timesteps = timesteps[timesteps_range]
    end

    p = plot(timesteps, energy["p_hp_e"][timesteps], 
             xlabel = L"Timesteps", 
             ylabel = L"P [kW]",
            #  ylims = (-25, 25),
             linewidth = 2,
             label = L"P^{e}_{HP}",
             legend = :topright,
             color = :green,
             dpi = 600)
    
    plot!(timesteps, energy["p_bess"][timesteps], label = L"P_{BESS}", color = :blue)
    plot!(timesteps, energy["p_ev"][timesteps], label = L"P_{EV}", color = RGB(1, 0.5, 0.3))
    
    if !raw && !isempty(save_path)
        Plots.savefig(p, save_path)
    end

    !raw && return display(p)

    plot!(timesteps, energy["p_hp_e_raw"][timesteps],
          label = L"Raw P^{e}_{HP}",
          linestyle=:dash,
          color=:green
    )
    plot!(timesteps, energy["p_bess_raw"][timesteps],
          label = L"Raw P_{BESS}",
          linestyle=:dash,
          color=:blue      
    )
    plot!(timesteps, energy["p_ev_raw"][timesteps],
          label = L"Raw P_{EV}",
          linestyle=:dash,
          color=RGB(1, 0.5, 0.3)
    )
    
    if !isempty(save_path)
        Plots.savefig(p, save_path)
    end

    display(p)
end

##########################################################################
# Plot Power balance

function plot_power_balance(hook::MCES_Hook; save_path::String="")
    energy = hook.energy
    timesteps = Int32.(energy["t"])
    n_days = div(length(timesteps), 96)

    function plot_range(range, title)
        p = plot(timesteps[range], energy["load_e"][range], 
                 xlabel = "Timesteps", ylabel = L"P [kW]",
                 linewidth = 2, label = L"P^{e}_{Load}",
                 legend = :topright, color = RGB(1, 0.85, 0.35),
                 dpi = 900)
        
        plot!(p, timesteps[range], energy["pv"][range], label = L"P_{PV}", color = RGB(0.8, 0.3, 0.3))
        plot!(p, timesteps[range], energy["grid"][range], label = L"P_{grid}", color = RGB(0.4, 0.4, 0.4))
        plot!(p, timesteps[range], energy["p_hp_e"][range], label = L"P^{e}_{HP}", color = :orange)
        plot!(p, timesteps[range], energy["p_bess"][range], label = L"P_{BESS}", color = :blue)
        plot!(p, timesteps[range], energy["p_ev"][range] .* energy["γ_ev"][range], label = L"P_{EV}", color = :green)
        Plots.title!(p, title)
        return p
    end

    full_range = 1:length(timesteps)
    summer_day = div(n_days, 2) * 96 + 1 : div(n_days, 2) * 96 + 96
    winter_day = (n_days - 1) * 96 + 1 : n_days * 96

    p1 = plot_range(full_range, "Power Balance - Full Year")
    p2 = plot_range(summer_day, "Power Balance - Summer Day")
    p3 = plot_range(winter_day, "Power Balance - Winter Day")

    combined_plot = plot(p1, p2, p3, layout = (2, 2), size = (1400, 850), margin = 6mm)

    !isempty(save_path) && Plots.savefig(combined_plot, save_path)
    display(combined_plot)
end

function plot_power_balance(hook1::MCES_Hook, hook2::MCES_Hook;
                       save_path::String = "",
                       label1::String = "Hook 1",
                       label2::String = "Hook 2",
                       day_to_plot::Integer = 1,
                       daylength::Integer = 96)  # Timesteps per day

    energy1 = hook1.energy
    energy2 = hook2.energy
    timesteps = Int32.(energy1["t"])
    
    L = min(length(timesteps), 
            length(energy1["p_bess"]),
            length(energy1["p_ev"]),
            length(energy1["γ_ev"]),
            length(energy2["p_bess"]),
            length(energy2["p_ev"]),
            length(energy2["γ_ev"]),
            length(energy1["load_e"]),
            length(energy1["pv"]))
    
    n_days = div(L, daylength)
    day_to_plot = clamp(day_to_plot, 1, n_days)
    
    day_range = ((day_to_plot - 1) * daylength + 1):(day_to_plot * daylength)

    function create_plot(time_range, title)
        plot_timesteps = (1:length(time_range))/(daylength/24)
        
        # Create base plot with white background
        p = plot(xlabel = "Hours",
                ylabel = "Power [kW]",
                legend = :outerright,
                dpi = 900,
                legendfontsize = 10,
                title = title,
                background_color = :white)
        
        # Add background rectangles for γ_ev
        y_min = minimum([
            minimum(energy1["load_e"][time_range]),
            minimum(energy1["pv"][time_range]),
            minimum(energy1["p_bess"][time_range]),
            minimum(energy1["p_ev"][time_range]),
            minimum(energy2["p_bess"][time_range]),
            minimum(energy2["p_ev"][time_range])
        ]) - 0.5  # Add padding
        
        y_max = maximum([
            maximum(energy1["load_e"][time_range]),
            maximum(energy1["pv"][time_range]),
            maximum(energy1["p_bess"][time_range]),
            maximum(energy1["p_ev"][time_range]),
            maximum(energy2["p_bess"][time_range]),
            maximum(energy2["p_ev"][time_range])
        ]) + 0.5  # Add padding


        
        
        # Add hook1 components
        plot!(p, plot_timesteps, energy1["p_bess"][time_range],
              linewidth = 1.5,
              label = L"P_{BESS}" * " ($label1)",
              color = RGBA(208/255, 32/255, 144/255, 1),
              style = :solid)
        
        plot!(p, plot_timesteps, energy1["p_ev"][time_range] .* energy1["γ_ev"][time_range],
              linewidth = 1.75,
              label = L"P_{EV}" * " ($label1)",
              color =RGBA(208/255, 32/255, 144/255, 0.75),
              style = :dashdot)
        
        # Add hook2 components
        plot!(p, plot_timesteps, energy2["p_bess"][time_range],
              linewidth = 1.5,
              label = L"P_{BESS}" * " ($label2)",
              color = RGBA(84/255, 100/255, 216/255, 1),
              style = :solid)
        
        plot!(p, plot_timesteps, energy2["p_ev"][time_range] .* energy2["γ_ev"][time_range],
              linewidth = 1.75,
              label = L"P_{EV}" * " ($label2)",
              color = RGBA(84/255, 100/255, 216/255, 0.75),
              style = :dashdot)

        # Add background rectangles
        for i in 1:length(time_range)-1
            if energy1["γ_ev"][time_range][i] == 0
                rectangle = Shape([
                    plot_timesteps[i], 
                    plot_timesteps[i+1], 
                    plot_timesteps[i+1], 
                    plot_timesteps[i]
                ], [y_min, y_min, y_max, y_max])
                plot!(p, rectangle, fillcolor=:grey90, linewidth=0, label="", alpha=0.35)
            end
        end
        
        # Add primary curves (load_e, pv)
        plot!(p, plot_timesteps, energy1["load_e"][time_range],
              linewidth = 2,
              label = L"P^e_{load}",
              color = RGBA(85/255, 135/255, 85/255, 0.8),
              style = :solid,
            )
        
        plot!(p, plot_timesteps, energy1["pv"][time_range],
              linewidth = 2,
              label = L"P_{PV}",
              color = RGBA(1, 0.75, 0, 0.8),
                )
        

        # Legend for gamma_ev
        plot!(p, [NaN], [NaN],
            fillrange = [NaN],
            fillcolor = :grey90,
            fillalpha = 0.35,
            linewidth = 0,
            label = L"\gamma_{EV} = 0")

        return p
    end

    p = create_plot(day_range, "Electrical Power Balance Comparison - Day $day_to_plot")

    !isempty(save_path) && savefig(p, save_path)
    display(p)
    return p
end

##########################################################################
# Thermal balance
function plot_thermal_balance(hook::MCES_Hook; save_path::String = "")
    energy = hook.energy
    timesteps = Int32.(energy["t"])
    n_days = div(length(timesteps), 96)

    function plot_range(range, title)
        p = plot(timesteps[range], energy["load_th"][range], 
                 xlabel = "Timesteps", ylabel = L"P [kW_t]",
                 linewidth = 2, label = L"P^{th}_{Load}",
                 legend = :bottomleft, color = RGB(1, 0.85, 0.35),
                 dpi = 600)
        
        plot!(p, timesteps[range], energy["st"][range], label = L"P_{ST}", color = RGB(0.8, 0.3, 0.3))
        plot!(p, timesteps[range], energy["p_tess"][range], label = L"P_{TESS}", color = :green)
        plot!(p, timesteps[range], energy["p_hp_th"][range], label = L"P^{th}_{HP}", color = :blue)
        Plots.title!(p, title)
        return p
    end

    full_range = 1:length(timesteps)
    summer_day = div(n_days, 2) * 96 + 1 : div(n_days, 2) * 96 + 96
    winter_day = (n_days - 1) * 96 + 1 : n_days * 96

    p1 = plot_range(full_range, "Thermal Balance - Full Year")
    p2 = plot_range(summer_day, "Thermal Balance - Summer Day")
    p3 = plot_range(winter_day, "Thermal Balance - Winter Day")

    combined_plot = plot(p1, p2, p3, layout = (2, 2), size = (1400, 850), margin = 6mm)

    !isempty(save_path) && Plots.savefig(combined_plot, save_path)
    display(combined_plot)
end


function plot_thermal_balance(hook1::MCES_Hook, hook2::MCES_Hook;
                            save_path::String = "",
                            label1::String = "Hook 1",
                            label2::String = "Hook 2",
                            day_to_plot::Integer = 1,
                            daylength::Integer = 96)  # Timesteps per day

    energy1 = hook1.energy
    energy2 = hook2.energy
    timesteps = Int32.(energy1["t"])
    
    L = min(length(timesteps), 
            length(energy1["load_th"]),
            length(energy1["st"]),
            length(energy1["p_hp_th"]),
            length(energy1["p_tess"]),
            length(energy2["p_hp_th"]),
            length(energy2["p_tess"]))
    
    n_days = div(L, daylength)
    day_to_plot = clamp(day_to_plot, 1, n_days)
    day_range = ((day_to_plot - 1) * daylength + 1):(day_to_plot * daylength)

    function create_plot(time_range, title)
        plot_timesteps = (1:length(time_range))/(daylength/24)

        p = plot(xlabel = "Hours",
            ylabel = "Power" * L"\quad [kW_{th}]",
            legend = :outerright,
            legendfontsize = 10,
            title = title,
            dpi = 900,
            background_color = :white)
              
        
        # Add hook1 components
        plot!(p, plot_timesteps, energy1["p_hp_th"][time_range],
              linewidth = 1.35,
              label = L"P^{th}_{HP}" * " ($label1)",
              color = RGBA(208/255, 32/255, 144/255, 1),
              style = :solid)
        
        plot!(p, plot_timesteps, energy1["p_tess"][time_range],
              linewidth = 1.5,
              label = L"P_{TESS}" * " ($label1)",
              color = RGBA(208/255, 32/255, 144/255, 0.75),
              style = :dashdot)
        
        # Add hook2 components
        plot!(p, plot_timesteps, energy2["p_hp_th"][time_range],
              linewidth = 1.35,
              label = L"P^{th}_{HP}" * " ($label2)",
              color = RGBA(84/255, 100/255, 216/255, 1),
              style = :solid)
        
        plot!(p, plot_timesteps, energy2["p_tess"][time_range],
                linewidth = 1.5,
                label = L"P_{TESS}" * " ($label2)",
                color = RGBA(84/255, 100/255, 216/255, 0.75),
                style = :dashdot)
        
        plot!(p, plot_timesteps, energy1["st"][time_range],
            linewidth = 2,
            label = L"P_{ST}",
            color = RGB(1, 0.75, 0))  

        plot!(p, plot_timesteps, energy1["load_th"][time_range],
            linewidth = 2,
            label = L"P^{th}_{load}",
            color = RGBA(85/255, 135/255, 85/255, 0.85),  
        )

        return p
    end

    p = create_plot(day_range, "Thermal Power Balance Comparison - Day $day_to_plot")

    !isempty(save_path) && savefig(p, save_path)
    display(p)
    return p
end

#################################################################################################
# Plot exogenous info

function plot_exog(hook::MCES_Hook;
                    save_path::String="",
                    day_to_plot::Integer = 1,
                    daylength::Integer = 96)  # Timesteps per day
    
    # Extract required data from hook
    energy = hook.energy
    load_e = energy["load_e"]
    load_th = energy["load_th"]
    pv = energy["pv"]
    γ_ev = energy["γ_ev"]
    
    # Calculate number of days and validate day_to_plot
    L = min(length(load_e), length(load_th), length(pv))
    n_days = div(L, daylength)
    day_to_plot = clamp(day_to_plot, 1, n_days)

    # Calculate the range for the selected day
    day_range = ((day_to_plot - 1) * daylength + 1):(day_to_plot * daylength)
    
    
    plot_timesteps = (1:length(day_range))/(daylength/24) # Time axis in hours
       
    p = plot(plot_timesteps, load_e[day_range],
             xlabel = "Hours",
             ylabel = "Power [kW]",
             linewidth = 2,
             label = L"P^{e}_{Load}",
             legend = :outerright,
             color = :darkseagreen3,
             dpi = 900,
             title = "Power Balance - Day $day_to_plot",
             legendfontsize = 10,
             background_color_legend = :white)

    plot!(p, plot_timesteps, load_th[day_range],
          linewidth = 2,
          label = L"P^{th}_{Load}",
          color = :orangered)

    plot!(p, plot_timesteps, pv[day_range],
          linewidth = 1.65,
          label = L"P_{PV}",
          color = :violetred)

    plot!(p, plot_timesteps, γ_ev[day_range] .* 0.5,
          linewidth = 1.35,
          label = L"\gamma_{EV}",
          color = :gray60,
          linestyle = :dash)

    !isempty(save_path) && savefig(p, save_path)
    display(p)
    return p
end

################################################################################################
# Plot Mean and Std
function plot_mean_and_std(hook::MCES_Hook, timesteps_range::Union{UnitRange{Int64}, Nothing} = nothing; 
    downsample::Bool = true, save_path = "")

    dbg = hook.debug
    means = dbg["mean"]
    stds = dbg["std"]

    mean_ordered = reshape(means, 3, :)
    std_ordered = reshape(stds, 3, :)

    timesteps = 1:size(mean_ordered, 2)

    if timesteps_range !== nothing
        timesteps = timesteps[timesteps_range]
    end

    
    if downsample
        timesteps = timesteps[1:10:end]
    end

    # Create the plot for means
    p1 = plot(timesteps, mean_ordered[1, timesteps],
              xlabel="Optimization Timesteps",
              ylabel="Mean",
              linewidth=2,
              label=L"Mean_{EV}",
              legend=:topright,
              color=RGB(0.1216, 0.4667, 0.7059),  # Blue
              dpi=600)

    plot!(p1, timesteps, mean_ordered[2, timesteps], label=L"Mean_{BESS}", color=RGB(1.0, 0.498, 0.0549))  # Orange
    plot!(p1, timesteps, mean_ordered[3, timesteps], label=L"Mean_{HPe}", color=RGB(0.1725, 0.6275, 0.1725))  # Green

    # Create the plot for standard deviations
    p2 = plot(timesteps, std_ordered[1, timesteps],
              xlabel="Optimization Timesteps",
              ylabel="Standard Deviation",
              linewidth=2,
              label=L"Std_{EV}",
              legend=:topright,
              linestyle=:dash,
              color=RGB(0.8392, 0.1529, 0.1569),  # Red
              dpi=600)

    plot!(p2, timesteps, std_ordered[2, timesteps], label=L"Std_{BESS}", color=RGB(0.5804, 0.4039, 0.7412))  # Purple
    plot!(p2, timesteps, std_ordered[3, timesteps], label=L"Std_{HPe}", color=RGB(0.549, 0.3373, 0.2941))  # Brown

    plot_combined = plot(p1, p2, layout = (1, 2), size = (1400, 500), margin = 10mm)

    if !isempty(save_path)
        Plots.savefig(plot_combined, save_path)
    end

    display(plot_combined)
end
#################################################################################################
# Plot Projections
function plot_projections(hook::MCES_Hook, timesteps_range::Union{UnitRange{Int64}, Nothing} = nothing; 
    params::Union{MCES_Params, Nothing} = nothing, save_path = "")

    reward_dissect = hook.reward_dissect
    energy = hook.energy
    timesteps = Int32.(energy["t"])

    # Assign weights based on params or default to 1f0
    w_ξsoc_ev = isnothing(params) ? 1f0 : params.w_ξsoc_ev
    w_ξsoc_bess = isnothing(params) ? 1f0 : params.w_ξsoc_bess
    w_ξsoc_tess = isnothing(params) ? 1f0 : params.w_ξsoc_tess
    w_ξp_tess = isnothing(params) ? 1f0 : params.w_ξp_tess
    w_ξp_grid = isnothing(params) ? 1f0 : params.w_ξp_grid
    w_init_proj = isnothing(params) ? 1f0 : params.w_init_proj
    w_op_proj = isnothing(params) ? 1f0 : params.w_op_proj
    weighted = isnothing(params) ? "" : "Weighted "


    op_projection = 1/5 .* (
        w_ξsoc_bess .* energy["ξsoc_bess"] + 
        w_ξsoc_ev   .* energy["ξsoc_ev"] + 
        w_ξsoc_tess .* energy["ξsoc_tess"] + 
        w_ξp_grid   .* energy["ξp_grid"] + 
        w_ξp_tess   .* energy["ξp_tess"]
    )

    if timesteps_range !== nothing
        timesteps = timesteps[timesteps_range]
    end

    p1 = plot(timesteps, energy["ξp_hp_e"][timesteps],
          xlabel = "Timesteps",
          ylabel = weighted * "Absolute Projection [kW]",
          linewidth = 2,
          label = "HPe Power",
          legend = :topleft,
          color = :orange,
          title = "Agent's decisions (pre-Environment)",
          dpi = 600
        )

    plot!(p1, timesteps, energy["ξp_bess"][timesteps], label = "BESS Power", color = :blue)
    plot!(p1, timesteps, energy["ξp_ev"][timesteps], label = "EV Power", color = :green)

    p2 = plot(timesteps, w_init_proj * (-reward_dissect["Initial Projection"][timesteps]),
            xlabel = "Timesteps",
            ylabel = weighted * "Absolute Projection [kW]",
            linewidth = 2,
            label = "Initial Projection",
            legend = :topleft,
            color = :red3,
            title = "Initial Projection",
            dpi = 600)

    p3 = plot(timesteps, w_ξsoc_ev * energy["ξsoc_ev"][timesteps],
              xlabel = "Timesteps",
              ylabel = weighted * "Absolute Projection [-]",
              linewidth = 2,
              label = L"SoC_{EV}",
              legend = :topleft,
              color = :green,
              dpi = 600)

    plot!(p3, timesteps, w_ξsoc_tess * energy["ξsoc_tess"][timesteps], label = L"SoC_{TESS}", color = :orange)
    plot!(p3, timesteps, w_ξsoc_bess * energy["ξsoc_bess"][timesteps], label = L"SoC_{BESS}", color = :blue)

    p4 = plot(timesteps, w_op_proj * (op_projection[timesteps]),
              xlabel = "Timesteps",
              ylabel = weighted * "Absolute Projection [kW]",
              linewidth = 2,
              label = "Operational Projection [-]",
              legend = :topleft,
              color = :purple1,
              dpi = 600)

    plot!(p4, timesteps, w_ξp_tess * energy["ξp_tess"][timesteps], label = "TESS Power", color = :orange)
    plot!(p4, timesteps, w_ξp_grid * energy["ξp_grid"][timesteps], label = "Grid Power", color = :black)

    plot_combined = plot(p1, p2, p3, p4, layout = (2, 2), size = (1400, 850), margin = 6mm)

    if !isempty(save_path)
        Plots.savefig(plot_combined, save_path)
    end

    display(plot_combined)
end


function plot_proj_year(hook::MCES_Hook, daylength::Int64 = 96)
    energy = hook.energy
    total_days = div(length(energy["ξp_hp_e"]), daylength)
    time_range = 1:(total_days * daylength)
    days = collect(1:length(time_range)) / daylength

    # Initial Projection
    initial_projection = energy["ξp_hp_e"][time_range] + energy["ξp_bess"][time_range] + energy["ξp_ev"][time_range]
    
    # Plot 1: Initial Projection
    p1 = plot(days, initial_projection,
              xlabel = "Day",
              ylabel = "Absolute Projection [kW]",
              linewidth = 2,
              label = "Initial Projection",
              legend = :outerright,
              color = :red3,
            #   xticks = 1:5:length(days),
              title = "Initial Projection",
              dpi = 600)

    # Plot 2: Agent's Decisions (pre-Environment)
    p2 = plot(days, energy["ξp_hp_e"][time_range],
              xlabel = "Day",
              ylabel = "Absolute Projection [kW]",
              linewidth = 2,
              label = "HPe Power",
              legend = :outerright,
              color = :orange,
              title = "Agent's decisions (pre-Environment)",
            #   xticks = 1:5:length(days), 
              dpi = 600)

    plot!(p2, days, energy["ξp_bess"][time_range], label = "BESS Power", color = :blue)
    plot!(p2, days, energy["ξp_ev"][time_range], label = "EV Power", color = :green)

    return p1, p2
end

#################################################################################
# Plot arbitrage
function plot_arbitrage(hook1::MCES_Hook, hook2::MCES_Hook; 
                        save_path::String="", 
                        prices = nothing, 
                        label1::String = "Hook 1", 
                        label2::String = "Hook 2",
                        day_to_plot::Integer = 1,
                        daylength::Integer = 96  # Timesteps per day
                        )
    
    grid_power1 = hook1.energy["grid"]
    grid_power2 = hook2.energy["grid"]
    
    if isnothing(prices)
        # Extracting price information out of the grid rewards from hook1. 
        L = min(length(grid_power1), length(hook1.reward_dissect["Grid"])) 
        prices = - hook1.reward_dissect["Grid"][1:L] ./ (grid_power1[1:L] * 0.25f0)  # 0.25 = 15 min / 60 min
    else
        L = min(length(grid_power1), length(grid_power2), length(prices))
    end
    
    n_days = div(L, daylength)  # Assuming 96 timesteps per day
    day_to_plot = clamp(day_to_plot, 1, n_days)

    # Calculate the range for the selected day
    day_range = ((day_to_plot - 1) * daylength + 1):(day_to_plot * daylength)

    function create_plot(time_range, title)
        plot_timesteps = (1:length(time_range))/(daylength/24) 
        p = plot(plot_timesteps, grid_power1[time_range],
                 xlabel = "Hours",
                 ylabel = "Grid Power Exchange [kW]",
                 linewidth = 1.5,
                 label = L"P_{grid}" * " ($label1)",
                 legend = :topleft,
                 color = RGBA(208/255, 32/255, 144/255, 1),
                 dpi = 900,
                 xticks = 0:5:24,
                 title = title,
                #  size = (1200, 600),
                 legendfontsize = 10,
                 background_color_legend = :white,
                #  ylim = (-5.001, 15)  # Can cause problems
                 )

        plot!(p, plot_timesteps, grid_power2[time_range],
              linewidth = 1.5,
              label = L"P_{grid}" * " ($label2)",
              color = RGBA(84/255, 100/255, 216/255, 1)
            )

        plot!(p, plot_timesteps, fill(0f0, length(time_range)),
              linewidth = 1,
              label = "",
              color = :grey,
              style = :solid
              )
        
        plot!(twinx(p), plot_timesteps, prices[time_range], 
              ylabel = "Electricity Price [€/kWh]",
              linewidth = 2,
              xticks = 0:5:24,
              label = "Prices",
              color = RGBA(85/255, 135/255, 85/255, 0.85),
              legend = :bottomleft,
              legendfontsize = 10,
              linestyle = :dashdot,
              dpi = 900,
            #   ylim = (0.1,0.4) # Can cause problems
        )

        return p
    end

    p = create_plot(day_range, "Arbitrage - Day $day_to_plot")

    !isempty(save_path) && savefig(p, save_path)
    display(p)
    return p
end


function plot_grid_powers(hook1::MCES_Hook, hook2::MCES_Hook;
                         save_path::String="",
                         label1::String="Hook 1",
                         label2::String="Hook 2",
                         day_to_plot::Integer=1,
                         daylength::Integer=96)  # Timesteps per day
    
    grid_power1 = hook1.energy["grid"]
    grid_power2 = hook2.energy["grid"]
    L = min(length(grid_power1), length(grid_power2))
    
    n_days = div(L, daylength)
    day_to_plot = clamp(day_to_plot, 1, n_days)
    
    # Calculate the range for the selected day
    day_range = ((day_to_plot - 1) * daylength + 1):(day_to_plot * daylength)
    plot_timesteps = (1:length(day_range))/(daylength/24)
    
    p = plot(plot_timesteps, grid_power1[day_range],
             xlabel="Hours",
             ylabel="Grid Power Exchange [kW]",
             linewidth=1.5,
             label=L"P_{grid}" * " ($label1)",
             legend=:outerright,
             color=RGBA(208/255, 32/255, 144/255, 1),
             dpi=900,
            #  xticks=0:5:24,
             title="Grid Power Exchange - Day $day_to_plot",
             legendfontsize=10,
             background_color_legend=:white)
    
    plot!(p, plot_timesteps, grid_power2[day_range],
          linewidth=1.5,
          label=L"P_{grid}" * " ($label2)",
          color=RGBA(84/255, 100/255, 216/255, 1))
    
    plot!(p, plot_timesteps, fill(0f0, length(day_range)),
          linewidth=1,
          label="",
          color=:grey,
          style=:solid)
    
    !isempty(save_path) && savefig(p, save_path)
    display(p)
    return p
end

function plot_grid_powers_and_cumcost(hook1::MCES_Hook, hook2::MCES_Hook;
                         save_path::String="",
                         label1::String="Hook 1",
                         label2::String="Hook 2",
                         day_to_plot::Integer=1,
                         daylength::Integer=96)
    
    # Grid power data
    grid_power1 = hook1.energy["grid"]
    grid_power2 = hook2.energy["grid"]
    L = min(length(grid_power1), length(grid_power2))
    
    # Extract grid costs for both hooks
    grid_cost1 = -hook1.reward_dissect["Grid"]
    grid_cost2 = -hook2.reward_dissect["Grid"]
    
    n_days = div(L, daylength)
    day_to_plot = clamp(day_to_plot, 1, n_days)
    
    # Calculate the range for the selected day
    day_range = ((day_to_plot - 1) * daylength + 1):((day_to_plot+1) * daylength)
    plot_timesteps = (1:length(day_range)) / (daylength / 24)
    
    # Calculate the cumulative sum of grid costs for the selected day
    grid_cost_cumsum1 = cumsum(grid_cost1[day_range])
    grid_cost_cumsum2 = cumsum(grid_cost2[day_range])
    
    # Create the main plot for grid power exchange
    p = plot(plot_timesteps, [grid_power1[day_range] grid_power2[day_range]],
             xlabel="Hours",
             ylabel="Grid Power Exchange [kW]",
             linewidth=1.35,
             legend= false,
             color=[RGBA(208/255, 32/255, 144/255, 0.8) RGBA(84/255, 100/255, 216/255, 0.8)],
             dpi=900,
             title="Grid Power Exchange and Cumulative Cost - Days $day_to_plot - $(day_to_plot + 1)",
             fg_minor_grid = :white,
             grid = :x,
        )
    
    plot!(p, plot_timesteps, fill(0f0, length(day_range)),
          linewidth=0.8,
          label="",
          color=:grey,
          style=:solid)
    
    # Add cumulative grid costs for both hooks on secondary y-axis
    plot!(
        twinx(p), plot_timesteps, [grid_cost_cumsum1 grid_cost_cumsum2], 
        ylabel="Cumulative Grid Cost [€]",
        linewidth=2,
        color=[RGBA(255/255, 128/255, 192/255, 1) RGBA(128/255, 160/255, 255/255, 1)],
        linestyle=:dashdot,
        fg_minor_grid = :white,
        legend=false,
        dpi=900
    )
    
    function generate_legend(label1::String, label2::String)
        # Define the colors and styles used in the main plot
        power_colors = [RGBA(208/255, 32/255, 144/255, 0.8), RGBA(84/255, 100/255, 216/255, 0.8)]
        cumcost_colors = [RGBA(255/255, 128/255, 192/255, 1), RGBA(128/255, 160/255, 255/255, 1)]
        linestyle_cumcost = [:dashdot, :dashdot]
        linestyle_power = [:solid, :solid]
        linewidth_power = 1.35
        linewidth_cumcost = 2
        
        # Create a dummy plot to display the legend
        p = plot(legend=:outerright, 
                 title="Legend for Grid Power and Cumulative Cost")
        
        # Add grid power entries
        plot!(p, [0], [0], color=power_colors[1], linewidth=linewidth_power,
              linestyle=linestyle_power[1], label=L"P_{grid}" * " ($label1)")
        plot!(p, [0], [0], color=power_colors[2], linewidth=linewidth_power,
              linestyle=linestyle_power[2], label=L"P_{grid}" * " ($label2)")
        
        # Add cumulative cost entries
        plot!(p, [0], [0], color=cumcost_colors[1], linewidth=linewidth_cumcost,
              linestyle=linestyle_cumcost[1], label="Cumulative Grid Cost ($label1)")
        plot!(p, [0], [0], color=cumcost_colors[2], linewidth=linewidth_cumcost,
              linestyle=linestyle_cumcost[2], label="Cumulative Grid Cost ($label2)")
        
        return p
    end
   
    # Save plot if a path is specified
    !isempty(save_path) && savefig(p, save_path)
    display(p)
    return p, generate_legend(label1, label2)
end

function plot_grid_prices(hook::MCES_Hook;
                         save_path::String="",
                         prices=nothing,
                         day_to_plot::Integer=1,
                         daylength::Integer=96)  # Timesteps per day
    
    grid_power = hook.energy["grid"]
    
    if isnothing(prices)
        # Extracting price information out of the grid rewards
        L = min(length(grid_power), length(hook.reward_dissect["Grid"]))
        prices = -hook.reward_dissect["Grid"][1:L] ./ (grid_power[1:L] * 0.25f0)  # 0.25 = 15 min / 60 min
    else
        L = min(length(grid_power), length(prices))
    end
    
    n_days = div(L, daylength)
    day_to_plot = clamp(day_to_plot, 1, n_days)
    
    # Calculate the range for the selected day
    day_range = ((day_to_plot - 1) * daylength + 1):(day_to_plot * daylength)
    plot_timesteps = (1:length(day_range))/(daylength/24)
    
    p = plot(plot_timesteps, prices[day_range],
             xlabel="Hours",
             ylabel="Electricity Price [€/kWh]",
             linewidth=2,
             label="Electricity Price",
             legend=:outerright,
             color=RGBA(85/255, 135/255, 85/255, 0.85),
             dpi=900,
             xticks=0:5:24,
             title="Electricity Prices - Day $day_to_plot",
             legendfontsize=10,
             background_color_legend=:white,
             linestyle=:dashdot)
    
    !isempty(save_path) && savefig(p, save_path)
    display(p)
    return p
end

function plot_arbitrage_sw(hook1::MCES_Hook, hook2::MCES_Hook;
                           save_path::String="",
                           prices=nothing,
                           label1::String="Hook 1",
                           label2::String="Hook 2")
    
    # Create summer plot (day 50)
    summer_plot = plot_arbitrage(hook1, hook2,
                                 prices=prices,
                                 label1=label1,
                                 label2=label2,
                                 day_to_plot=50)
    
    # Create winter plot (day 95)
    winter_plot = plot_arbitrage(hook1, hook2,
                                 prices=prices,
                                 label1=label1,
                                 label2=label2,
                                 day_to_plot=95)
    
    # Combine the plots side by side
    combined_plot = plot(summer_plot, winter_plot,
                         layout = (1, 2),
                         size = (1600, 600),
                         bottommargin = 10mm,
                         topmargin = 5mm,
                         leftmargin = 10mm,
                         rightmargin = 10mm,
                         link = :y)  # Link y-axes for easier comparison
    
    # Update titles
    plot!(combined_plot[1], title="Arbitrage - Summer Day (Day 50)")
    plot!(combined_plot[2], title="Arbitrage - Winter Day (Day 95)")
    
    # Save the combined plot if a save path is provided
    if !isempty(save_path)
        savefig(combined_plot, save_path)
    end
    
    # Display the combined plot
    display(combined_plot)
end





#####################################################################################
# Get many plots for testing results.

function generate_plots(hook::MCES_Hook, folder_path::String; 
    params::Union{MCES_Params, Nothing} = nothing, extra::String = "", suffix::String = "", prices = nothing)

    plot_functions = [
        (plot_grid_ev, "grid_ev.png"),
        (plot_reward_dissect, "reward_dissect.png"),
        (plot_projections, "projections.png"),
        (plot_soc, "soc.png"),
        (plot_decisions, "decisions.png"),
        (plot_power_balance, "power_balance.png"),
        (plot_thermal_balance, "thermal_balance.png"),
        (plot_arbitrage, "arbitrage_example.png"),
        # (plot_mean_and_std, "mean_and_std.png")
    ]

    for (func, filename) in plot_functions
        # Add suffix to the filename if provided
        filename_with_suffix = replace(filename, r"\.png$" => "_$suffix.png")
        full_path = joinpath(folder_path, filename_with_suffix)
        try
            if !isnothing(prices) && func == plot_arbitrage
                func(hook, save_path=full_path, prices = prices);
                println("Success in generating: $filename_with_suffix")
                continue
            end
            
            func(hook, save_path=full_path);
            println("Success in generating: $filename_with_suffix")


            if !isnothing(params) && func in [plot_reward_dissect, plot_projections]
                # Create filename with "w_" prefix and user-provided suffix
                weighted_filename = replace(filename, r"\.png$" => "_w_$suffix.png")
                weighted_full_path = joinpath(folder_path, weighted_filename)
                func(hook, save_path=weighted_full_path, params=params);
                println("Success in generating: weighted $weighted_filename")
            end
        catch e
            println("Error generating $filename_with_suffix: $(typeof(e))")
            println("Error message: $e")
        end
    end

    if !isnothing(params)
        params_file_path = joinpath(folder_path, "params_log.txt")
        try
            open(params_file_path, "w") do file
                println(file, fieldnames(typeof(params)))
                println(file, "Params: ")
                println(file, params)
                println(file, "Extra Information:")
                println(file, extra)
            end
            println("Successfully logged params to params_log.txt")
        catch e
            println("Error writing params to params_log.txt: $(typeof(e))")
            println("Error message: $e")
        end
    end

end

function plot2hooks(;
    day::Integer = 45,
    save_path::String = "",
    h1::MCES_Hook,
    h2::MCES_Hook,
    ex = exog_test_90,
    label1::String = "Expert MPC",
    label2::String = "RL Agent (PPO)",
    width = 1500,
    height = 550,
    )
    
    p1 = plot_power_balance(h1, h2, label1=label1, label2=label2, day_to_plot=day)
    p2 = plot_thermal_balance(h1, h2, label1=label1, label2=label2, day_to_plot=day)
    p3 = plot_soc(h1, h2, label1=label1, label2=label2, day_to_plot=day)
    p4 = plot_grid_powers(h1, h2, label1=label1, label2=label2, day_to_plot=day)
    p5 = plot_grid_prices(h1, prices = ex.λ_buy, day_to_plot=day)
    p6 = plot_grid_cost(h1, h2, label1=label1, label2=label2, episodic = true)
    p7 = plot_ev_cost(h1, h2, label1=label1, label2=label2, episodic = true)
    p8, p9 = plot_soc_year(h1, h2, label1=label1, label2=label2)
    _, p10 = plot_proj_year(h2)
    p11 = plot_auxtess(h1, h2, label1=label1, label2=label2, daylength = 96)
    p12, p12_legend = plot_grid_powers_and_cumcost(h1, h2, label1=label1, label2=label2, daylength = 96, day_to_plot=day)

    n_plots = 13
    n_cols = 1
    n_rows = ceil(Int, n_plots/n_cols)

    # Combine all plots in a layout
    plot_layout = plot(p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12,  p12_legend,
                    layout=(n_rows, n_cols),
                    size=(width*n_cols, height*n_rows),
                    # margin = 10mm,
                    bottommargin = 10mm,
                    topmargin = 7mm,
                    leftmargin = 53mm,
                    rightmargin = 50mm,
                    legendfontsize = 12,
    )
    
    !isempty(save_path) && savefig(plot_layout, save_path)

    return plot_layout
end


###################################################################
# Plots of Neural Network Parameters
function plot_multiple_archs(widths::Union{Vector{Int}, AbstractRange}, types::Vector{Int})
    # Dictionary mapping type numbers to architecture names
    type_names = Dict(
        1 => "Constant Width (CW)",
        2 => "Mid CW",
        3 => "Deep CW",
        4 => "CW + Std",
        5 => "Pyramid",
        6 => "Deep Pyramid",
        7 => "Bottleneck",
        8 => "Residual",
        9 => "Deep Residual",
        10 => "Residual + Bottleneck",
        11 => "3 CW Branches ",
        12 => "3 Pyramid Branches "
    )

    p = plot(
        xlabel="Input Width [-]",
        ylabel="Number of Parameters",
        legend= :outerright,
        yaxis = :log10,
        margin = 5mm,
        dpi = 600
    )


    for type in types
        parameters = count_nn_params(widths, type, ns = 30)
        plot!(p, widths, parameters, 
              label=get(type_names, type, "Type $type"),
              linewidth=1.5,
              linestyle = type < 7 ? :solid : :dot,
              )
    end
    display(p)
    # Uncomment the following line if you want to save the plot
    # savefig(p,joinpath(@__DIR__, "..","Figs\\widthvsparams.svg"))
    
    return p
end

########################################################################################





@info "Finished preparing the plotting functions"

