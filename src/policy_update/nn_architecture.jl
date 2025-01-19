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

function count_nn_params(widths::Union{Vector{Int}, AbstractRange}, type::Integer; ns = 14)
    parameters = Int64[]
    for width in widths
        network = custom_chain(ns, 3; width=width, type = type)
        num_params = sum(length, Flux.params(network))
        push!(parameters, num_params)
    end
    
    return parameters
end

@info "All NN architectures are available for the agent"