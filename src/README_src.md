This folder is crucial for implementing a Reinforcement Learning (RL)-based Energy Management System (EMS) for multi-carrier residential buildings. It provides a modular, efficient framework for simulating and optimizing energy flows among various assets—including PV systems, batteries, EVs, heat pumps, and thermal storage—while enforcing safety constraints. Researchers and practitioners can use these components to experiment with custom RL policies, safety projections, and performance metrics in small-scale energy systems.

This implementation benchmarks against an "Expert" system (a day-ahead MPC planner) and demonstrates that RL can effectively manage small-scale MCES applications where traditional MPC may be impractical and computationally costly.

### Getting started

To use this framework, first load the core components:

```julia
include("mces_core.jl")
```

This will import all necessary dependencies and prepare the environment for running RL experiments. All relevant functions are clearly documented in hopes it will be of help to the user. 


## File Details

- **mces_core.jl**  
  Acts as the main configuration and setup script. Run this file first (e.g., using `include("mces_core.jl")`) to load dependencies, integrate RL algorithms, and initialize the environment.

- **mces_env.jl**  
  Defines the simulation framework for the MCES, presenting all of the system's components. It models the dynamic interactions between electrical and thermal subsystems, supports RL control, and enforces safety through projection-based methods.

- **mces_exogenous.jl**  
  Manages the integration and processing of external time-series data (loads, PV generation, pricing, EV status...) into the simulation, necessary to afford state transitions.

- **mces_helper.jl**  
  Provides utility functions for data comparison, environment state updates, gradient clipping, and conversions between EMS modules and RL replay objects, alongside file I/O operations.

- **mces_hook.jl**  
  Implements monitoring and debugging hooks to track episode rewards, energy metrics, and constraint violations, with options for memory-efficient data recording.

- **mces_hypertuning.jl**  
  Contains utilities for processing hyperparameter optimization results. It offers statistical analysis, visualization of optimization performance, and the option to analyze parameter results across multiple optimization runs.

- **mces_load_data.jl**  
  Loads training, testing, and validation datasets from `.jld2` files. It constructs exogenous data structures essential for the simulation environment.

- **mces_plot.jl**  
  Provides a suite of plotting functions for visualizing time series data, power and thermal balances, state-of-charge, rewards, and other performance metrics.

- **mces_reward.jl**  
  Implements the reward system for the RL agent, computing penalties for grid costs, EV charging deviations, constraint violations, and includes mechanisms for discounting and performance evaluation.

- **mces_run.jl**  
  Serves as the main execution framework for both training and evaluating RL agents. It supports various run modes, safety projections, and customizable hooks for data collection.

- **mces_safety.jl**  
  Implements optimization-based action projection using JuMP models to enforce operational constraints across energy assets. It offers both simple and advanced projection methods.

- **mces_state_buffer.jl**  
    Implements utilities to extract and buffer state features for the MCES environment. It maps feature variables to indices and time lags, enables efficient state information extraction across various time steps, and updates the state buffer with normalized state vectors using online agent statistics.

- **mces_stats.jl**  
  Computes and aggregates statistical performance metrics from simulation runs, enabling detailed analysis of agent performance and system behavior.
