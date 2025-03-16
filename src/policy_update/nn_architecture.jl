"""
    custom_chain(
        ns::Integer, na::Integer, rng::AbstractRNG = Xoshiro(42); 
        width::Integer = 32, type::Integer = 1, 
        gain::Float32 = 0.1f0, activation::Integer = 1
    )

Creates a neural network with a customizable architecture.

# Arguments
- `ns::Integer`: Number of input features.
- `na::Integer`: Number of output features.
- `rng::AbstractRNG`: Random number generator for weight initialization (default: `Xoshiro(42)`).
- `width::Integer`: Base width parameter for hidden layers (default: `32`).
- `type::Integer`: Architecture type, controlling the network structure (default: `1`).
- `gain::Float32`: Gain value for the output layer weight initialization (default: `0.1f0`).
- `activation::Integer`: Activation function selector (default: `1`).

# Returns
- A Flux.jl neural network model with the specified architecture.

# Details
The function supports multiple network architectures by specifying the `type` parameter:
1. Constant width: 3-layer network with uniform hidden layer width
2. Mid constant width: 5-layer network with adjusted width
3. Deep constant width: 7-layer network with adjusted width
4. Constant width + std: Parallel networks for mean and standard deviation
5. Pyramid: Decreasing width through layers
6. Deep pyramid: Gradually decreasing width through many layers
7. Bottleneck: Wide-narrow-wide layer pattern
8. Residual network: Network with skip connections
9. Deep residual: Network with multiple skip connections
10. Residual + bottleneck: Combines residual and bottleneck patterns
11. 3 CW branches: Three parallel constant width networks
12. 3 pyramid branches: Three parallel pyramid networks

The width of hidden layers is automatically adjusted to maintain similar parameter counts across different architectures.

Activation functions are selected via the `activation` parameter:
1. `tanh_fast`: Fast approximation of hyperbolic tangent
2. `mish`: Mish activation function

All layers use orthogonal initialization, with the output layer using the specified `gain` parameter.
"""
function custom_chain(
    ns::Integer, na::Integer, rng::AbstractRNG = Xoshiro(42); 
    width::Integer = 32, type::Integer = 1, 
    gain::Float32 = 0.1f0, activation::Integer = 1
    )
    # Widths are adjusted to mantain a similar parameter count to "constant width"
    w = width
    
    # Define activation functions
    activation_functions = Dict(
        1 => tanh_fast,
        2 => mish, 
    )

    # Select activation function
    act_fn = get(activation_functions, activation) do
        error("Unsupported activation function: $activation")
    end

    if type == 1 # constant width
        c = Chain(
            Dense(ns, w, act_fn; init=Flux.orthogonal(rng)),
            Dense(w, w, act_fn; init=Flux.orthogonal(rng)),
            Dense(w, na; init=Flux.orthogonal(rng, gain = gain))
        )

    elseif type == 2 # Mid constant width
        adj_w = round(Int, w * sqrt(3/9)) 
        c = Chain(
            Dense(ns, adj_w, act_fn; init=Flux.orthogonal(rng)),
            Dense(adj_w, adj_w, act_fn; init=Flux.orthogonal(rng)),
            Dense(adj_w, adj_w, act_fn; init=Flux.orthogonal(rng)),
            Dense(adj_w, adj_w, act_fn; init=Flux.orthogonal(rng)),
            Dense(adj_w, na; init=Flux.orthogonal(rng, gain = gain))
        )

    elseif type == 3 # Deep constant width
        adj_w = round(Int, w * sqrt(3/14))  
        c = Chain(
            Dense(ns, adj_w, act_fn; init=Flux.orthogonal(rng)),
            Dense(adj_w, adj_w, act_fn; init=Flux.orthogonal(rng)),
            Dense(adj_w, adj_w, act_fn; init=Flux.orthogonal(rng)),
            Dense(adj_w, adj_w, act_fn; init=Flux.orthogonal(rng)),
            Dense(adj_w, adj_w, act_fn; init=Flux.orthogonal(rng)),
            Dense(adj_w, adj_w, act_fn; init=Flux.orthogonal(rng)),
            Dense(adj_w, na; init=Flux.orthogonal(rng, gain = gain))
        )

    elseif type == 4 # constant width + std
        adj_w = round(Int, w * sqrt(1/2)) 
        mean_network = Chain(
        Dense(ns, adj_w, act_fn; init=Flux.orthogonal(rng)),
        Dense(adj_w, adj_w, act_fn; init=Flux.orthogonal(rng)),
        Dense(adj_w, na; init=Flux.orthogonal(rng, gain = gain)),
        x -> reshape(x, 1, :)  # Reshape to 1 x na
        )

        std_network = Chain(
            Dense(ns, adj_w, act_fn; init=Flux.orthogonal(rng)),
            Dense(adj_w, adj_w, act_fn; init=Flux.orthogonal(rng)),
            Dense(adj_w, na; init=Flux.orthogonal(rng)),
            x -> softplus.(x) .+ 1f-2,
            x -> reshape(x, 1, :)  # Reshape to 1 x na
        )

        c = Parallel(
            vcat,
            mean_network,
            std_network
        )

    elseif type == 5 # pyramid
        adj_w = round(Int, w * 1.25) 
        half_w = adj_w ÷ 2
        qrt_w = adj_w ÷ 4
        c = Chain(
            Dense(ns, adj_w, act_fn; init=Flux.orthogonal(rng)),
            Dense(adj_w, half_w, act_fn; init=Flux.orthogonal(rng)),
            Dense(half_w, qrt_w, act_fn; init=Flux.orthogonal(rng)),
            Dense(qrt_w, na; init=Flux.orthogonal(rng, gain = gain))
        )

    elseif type == 6 # deep pyramid
        adj_w = round(Int, w * 0.71)  
        c = Chain(
            Dense(ns, adj_w, act_fn; init=Flux.orthogonal(rng)),
            Dense(adj_w, div(adj_w * 5, 6), act_fn; init=Flux.orthogonal(rng)),
            Dense(div(adj_w * 5, 6), div(adj_w * 2, 3), act_fn; init=Flux.orthogonal(rng)),
            Dense(div(adj_w * 2, 3), div(adj_w, 2), act_fn; init=Flux.orthogonal(rng)),
            Dense(div(adj_w, 2), div(adj_w, 3), act_fn; init=Flux.orthogonal(rng)),
            Dense(div(adj_w, 3), div(adj_w, 4), act_fn; init=Flux.orthogonal(rng)),
            Dense(div(adj_w, 4), div(adj_w, 6), act_fn; init=Flux.orthogonal(rng)),
            Dense(div(adj_w, 6), na; init=Flux.orthogonal(rng, gain = gain))
        )

    elseif type == 7 # bottleneck
        adj_w = round(Int, w * 1.55) 
        small_w = adj_w ÷ 5
        c = Chain(
            Dense(ns, adj_w, act_fn; init=Flux.orthogonal(rng)),
            Dense(adj_w, small_w, act_fn; init=Flux.orthogonal(rng)),
            Dense(small_w, adj_w, act_fn; init=Flux.orthogonal(rng)),
            Dense(adj_w, na; init=Flux.orthogonal(rng, gain = gain))
        )

    elseif type == 8 # Residual network
        adj_w = round(Int, w * sqrt(3/6))  
        c = Chain(
            Dense(ns, adj_w, act_fn; init=Flux.orthogonal(rng)),
            SkipConnection(
                Chain(
                    Dense(adj_w, adj_w, act_fn; init=Flux.orthogonal(rng)),
                    Dense(adj_w, adj_w, act_fn; init=Flux.orthogonal(rng))
                ),
                +
            ),
            Dense(adj_w, na; init=Flux.orthogonal(rng, gain = gain))
        )
    
    elseif type == 9 # Deep Residual
        adj_w = round(Int, w * sqrt(3/12))  # Adjust width to maintain similar parameter count
        c = Chain(
            Dense(ns, adj_w, act_fn; init=Flux.orthogonal(rng)),
            SkipConnection(
                Chain(
                    Dense(adj_w, adj_w, act_fn; init=Flux.orthogonal(rng)),
                    Dense(adj_w, adj_w, act_fn; init=Flux.orthogonal(rng))
                ),
                +
            ),
            SkipConnection(
                Chain(
                    Dense(adj_w, adj_w, act_fn; init=Flux.orthogonal(rng)),
                    Dense(adj_w, adj_w, act_fn; init=Flux.orthogonal(rng))
                ),
                +
            ),
            Dense(adj_w, na; init=Flux.orthogonal(rng, gain = gain))
        )
        
    elseif type == 10 # Residual + Bottleneck
        adj_w = round(Int, w * 1.48)  
        small_w = adj_w ÷ 5
        c = Chain(
            Dense(ns, adj_w, act_fn; init=Flux.orthogonal(rng)),
            SkipConnection(
                Chain(
                    Dense(adj_w, small_w, act_fn; init=Flux.orthogonal(rng)),
                    Dense(small_w, small_w, act_fn; init=Flux.orthogonal(rng)),
                    Dense(small_w, adj_w, act_fn; init=Flux.orthogonal(rng))
                ),
                +
            ),
            Dense(adj_w, na; init=Flux.orthogonal(rng, gain = gain))
        )

    elseif type == 11 # 3 CW Branches 
        adj_w = round(Int, w * sqrt(3/10)) 
        contract = na < 3 
        base_output = max(na,3) ÷ 3
        remainder = max(na,3) % 3
        branch_outputs = [
            base_output + (remainder > 0 ? 1 : 0),
            base_output + (remainder > 1 ? 1 : 0),
            base_output
        ]
    
        branches = [
            Chain(
                Dense(ns, adj_w, act_fn; init=Flux.orthogonal(rng)),
                Dense(adj_w, adj_w, act_fn; init=Flux.orthogonal(rng)),
                Dense(adj_w, branch_outputs[1]; init=Flux.orthogonal(rng, gain = gain))
            ),
            Chain(
                Dense(ns, adj_w, act_fn; init=Flux.orthogonal(rng)),
                Dense(adj_w, adj_w, act_fn; init=Flux.orthogonal(rng)),
                Dense(adj_w, branch_outputs[2]; init=Flux.orthogonal(rng, gain = gain))
            ),
            Chain(
                Dense(ns, adj_w, relu; init=Flux.orthogonal(rng)),
                Dense(adj_w, adj_w, relu; init=Flux.orthogonal(rng)),
                Dense(adj_w, branch_outputs[3], relu; init=Flux.orthogonal(rng, gain = gain))
            )
        ]
        c = Chain(
            Parallel(vcat, branches...),
            contract ? Dense(3, na) : identity  # In case na <3
        )


    elseif type == 12 # 3 Pyramid Branches 
        adj_w = round(Int, w * sqrt(1/2)) 
        half_w = adj_w ÷ 2
        qrt_w = adj_w ÷ 4
        contract = na < 3 
        base_output = max(na,3) ÷ 3
        remainder = max(na,3) % 3
        branch_outputs = [
            base_output + (remainder > 0 ? 1 : 0),
            base_output + (remainder > 1 ? 1 : 0),
            base_output
        ]
    
        branches = [
            Chain(
                Dense(ns, adj_w, act_fn; init=Flux.orthogonal(rng)),
                Dense(adj_w, half_w, act_fn; init=Flux.orthogonal(rng)),
                Dense(half_w, qrt_w, act_fn; init=Flux.orthogonal(rng)),
                Dense(qrt_w, branch_outputs[1]; init=Flux.orthogonal(rng, gain = gain))
            ),
            Chain(
                Dense(ns, adj_w, act_fn; init=Flux.orthogonal(rng)),
                Dense(adj_w, half_w, act_fn; init=Flux.orthogonal(rng)),
                Dense(half_w, qrt_w, act_fn; init=Flux.orthogonal(rng)),
                Dense(qrt_w, branch_outputs[2]; init=Flux.orthogonal(rng, gain = gain))
            ),
            Chain(
                Dense(ns, adj_w, relu; init=Flux.orthogonal(rng)),
                Dense(adj_w, half_w, relu; init=Flux.orthogonal(rng)),
                Dense(half_w, qrt_w, relu; init=Flux.orthogonal(rng)),
                Dense(qrt_w, branch_outputs[3], relu; init=Flux.orthogonal(rng, gain = gain))
            )
        ]
        c = Chain(
            Parallel(vcat, branches...),
            contract ? Dense(3, na) : identity  # In case na <3
        )

    else
        error("Invalid chain type parameter. Choose a number between 1 and 11.")
    end

    return c
end


"""
    count_nn_params(widths::Union{Vector{Int}, AbstractRange}, type::Integer; ns = 14)

Counts the number of parameters in neural networks of various widths for a given architecture type.

# Arguments
- `widths::Union{Vector{Int}, AbstractRange}`: A collection of width values to evaluate.
- `type::Integer`: The architecture type to use (corresponds to the `type` parameter in `custom_chain`).
- `ns::Integer`: Number of input features (default: `14`).

# Returns
- A vector of integers representing the parameter count for each width in the input collection.

# Details
This utility function helps compare parameter counts across different network widths while keeping
the architecture type constant. It creates a network for each width using `custom_chain`, counts
the parameters, and returns the results.
"""
function count_nn_params(widths::Union{Vector{Int}, AbstractRange}, type::Integer; ns = 14)
    parameters = Int64[]
    for width in widths
        network = custom_chain(ns, 3; width=width, type = type)
        num_params = sum(length, Flux.params(network))
        push!(parameters, num_params)
    end
    
    return parameters
end

function plot_multiple_architectures(widths::Union{Vector{Int}, AbstractRange}, types::Vector{Int})
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
    # savefig(p,joinpath(@__DIR__, "widthvsparams.svg"))
    
    return p
end

@info "All NN architectures are available for the agent"