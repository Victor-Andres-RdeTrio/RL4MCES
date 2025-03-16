# File to include all the basic functioning for the MCES environment

using Revise
using Base.Threads
using Base.Iterators: partition
using BSON
using ChainRulesCore: ignore_derivatives
using CircularArrayBuffers
using CSV
using DataFrames
using Dates
using Distributions: Normal, Categorical, logpdf, ContinuousDistribution, Distributions, LogNormal, Gamma, Weibull, Exponential, Pareto, InverseGaussian, Laplace, Rayleigh, Frechet 
# using DomainSets
using DomainSets: ×, (..), fullspace, TupleProductDomain
using ElasticArrays
using Flux
using Flux: glorot_uniform, normalise, params, gradient, Flux
using Functors: @functor
import HiGHS
using Hyperopt
import HypothesisTests
using Ipopt
using JLD2
using JSON3
using JuMP
using LaTeXStrings
using LinearAlgebra: norm
using OnlineStats
using Plots
using Plots: px, mm
using PlotThemes
using Profile
using Random
# using SparseArrays
import SpecialFunctions
using Statistics
import StatsPlots # Needed for the boxplot 
using StatsBase: crosscor, crosscov, StatsBase, sample
using Smoothing
using Test
using Zygote: @showgrad, Buffer


# ReinforcementLearning packages
using ReinforcementLearningBase
using ReinforcementLearningCore
using ReinforcementLearningEnvironments: AbstractEnv
using ReinforcementLearningCore: AbstractHook, PostActStage, PostEpisodeStage, PreExperimentStage
import ReinforcementLearningCore: global_norm, clip_by_global_norm!, run, _run
using ReinforcementLearningZoo
import ReinforcementLearningZoo: policy_gradient_estimate, action_distribution, VPG


function try_includet(filename)
    try 
        includet(filename)
    catch e
        @warn "Could not evaluate $filename"
        println("The error is: $e")
    end
end

# Define statistics functions and Types
try_includet("mces_stats.jl")

# Define the MCES struct and most components
try_includet("mces_env.jl")


# Add Policy Types
try_includet("policy_update/policy_types.jl")

# Add Exogenous Data and needed functions
try_includet("mces_exogenous.jl")


try_includet("mces_load_data.jl")

# Add custom Stop Conditions
try_includet("policy_update/stop.jl")

# Add the Personalized Hook for tracking progress and rewards
try_includet("mces_hook.jl")

# Modify ReinforcementLearning functionality
try_includet("mces_run.jl")

# Add plotting functions
try_includet("mces_plot.jl")

# Adding Policies 
try_includet("policy_update/replay.jl")
try_includet("policy_update/myvpg.jl")
try_includet("policy_update/a2cgae.jl")
try_includet("policy_update/ppo.jl")
try_includet("policy_update/policy.jl")
try_includet("policy_update/policy_test.jl")
try_includet("policy_update/behav_cloning.jl")

# Add helper functions
try_includet("mces_helper.jl")
try_includet("mces_state_buffer.jl")
try_includet("policy_update/nn_architecture.jl")
try_includet("mces_hypertuning.jl")
try_includet("mces_reward.jl")

# Add safety projection model
try_includet("mces_safety.jl")
