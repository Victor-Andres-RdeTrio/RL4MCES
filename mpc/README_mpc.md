# MPC Expert Agent Testing Module

This folder contains utilities for testing and comparing the Model Predictive Control (MPC) expert agent from the EMSModule against a Reinforcement Learning (RL) agent in the same environment.

## Overview

The main purpose of this module is to provide a fair comparison between an MPC expert agent and an RL agent by:

1. Feeding identical information to the MPC that was provided to the RL agent
2. Having the MPC generate optimal control decisions
3. Implementing these decisions in the RL environment (to avoid discrepancies that would arise from using the MPC's own environment)

## Key Components

### Main Functions

- `runEMS(steps; save=true, RLEnv=true)`: Runs the Energy Management System simulation with the MPC Expert Agent for a specified number of steps, optionally using the RL environment for direct comparison.

- `simTransitionEnv!(results, data, s; typeOpt="day-ahead")`: Simulates the transition function, updating the state of various storage assets based on optimal control decisions and exogenous information, it uses the dynamics within the RL environment.

### Utility Functions

- Conversion functions (in `convertEMSobjs.jl`): Convert between MCES components used by the MPC and those accepted by the RL environment.

- Modified EMSModule functions (in `EMS_mods.jl`): Custom versions of EMSModule functions. 
  - Example: `build_data()` -> ensures identical test conditions for both agents.

### Dependencies

- Package imports required for the EMSModule are contained in `EMS_pkg.jl`

## Workflow

1. The `runEMS` function initializes the environment with specific parameters
2. It executes rolling horizon optimization using either the RL environment or EMSModule's own environment
3. Control decisions from the MPC are implemented in the environment via `simTransitionEnv!`
4. Results are collected, concatenated, and optionally saved to disk

## Implementation Notes

- This module was designed to work with a single EV configuration
- Fixed test data is used to ensure consistent comparisons
- Most EMSModule functions have been modified to maintain compatibility while ensuring identical test conditions
- The implementation supports direct comparison by running the MPC expert's decisions through the same environment used by the RL agent

## Usage

Load the necessary packages and then call `runEMS()` with the desired number of simulation steps. Set `RLEnv=true` to use the RL environment for simulation (recommended for fair comparison) or `false` to use EMSModule's own environment.