# Reinforcement Learning Algorithms

This folder contains the core implementation of reinforcement learning algorithms used within the MCES Energy Management System project. It provides the policy gradient methods, neural network architectures, and evaluation tools necessary for training and deploying RL-based energy management solutions.

The code in this folder enables the system to learn near-optimal control strategies for balancing grid costs, EV charging satisfaction, and safety constraints.

## File Descriptions

- **policy_types.jl**: Defines the core policy structures including PPO, A2CGAE, VPG, Replay, and Clone policies, establishing the abstract types and interfaces used throughout the framework.

- **a2cgae.jl**: Implements the Advantage Actor-Critic with Generalized Advantage Estimation algorithm for enhanced training stability through improved advantage estimation.

- **ppo.jl**: Contains the Proximal Policy Optimization implementation with clipped surrogate objective, designed for stable and sample-efficient policy updates.

- **myvpg.jl**: Provides the Vanilla Policy Gradient with Critic implementation, serving as a baseline policy gradient approach with separate policy and value networks.

- **nn_architecture.jl**: Offers utilities for creating customizable neural network architectures with adjustable depths, widths, and activation functions to support various learning requirements.

- **policy.jl**: Contains functions for creating, training, and evaluating RL agents, including hyperparameter optimization capabilities using the BOHB algorithm.

- **policy_test.jl**: Provides testing utilities for trained policies, including validation evaluation, performance assessment, and safety impact analysis.

- **replay.jl**: Implements a replay mechanism for reproducing predefined action sequences from historical data, supporting direct comparisons with MPC experts.

- **behav_cloning.jl**: Enables behavioral cloning for imitation learning. 
  - *Warning: this functionality was not included in the master thesis and was only tested during initial stages of development*.

- **stop.jl**: Implements early stopping mechanisms based on performance metrics. 
  - *Warning: this functionality was not included in the master thesis and was only tested during initial stages of development*.
