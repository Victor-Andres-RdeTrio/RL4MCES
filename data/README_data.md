# MCES Data Processing

## Overview

The `data` folder contains Julia code for processing Multi-Carrier Energy System (MCES) data used in training a Reinforcement Learning (RL) agent. The code handles various energy-related data streams including PV generation, electricity prices, electrical and thermal loads, and electric vehicle (EV) charging profiles. 

Its **main purpose** is to use the same inputs that the MPC expert has access to and generate/expand the data needed for training and testing the RL agent. 

## Functions

The codebase includes utility functions for:

- Data partitioning (train/test splits)
- Data augmentation and expansion
- Statistical transformations (mean/max replacements)
- EV charging behavior simulation
- Cross-validation dataset generation

## Inputs

The system requires the following input files in the `input` directory:

- `pv_data.jld2`: Solar PV generation data
- `prices_2022.jld2`: Electricity buying and selling prices
- `load_e.jld2`: Electrical load profiles
- `load_th.jld2`: Thermal load profiles
- `gmmElaadFit.dat`: Serialized Gaussian Mixture Model for EV arrival patterns
- `mean-session-length-per.csv`: Mean EV connection durations by arrival time

## Outputs

The system generates several JLD2 files containing processed datasets:

- `train.jld2`: Base training data (274 days, 26,304 time steps)
- `train_ex.jld2`: Expanded training data (365 days, 35,040 time steps)
- `test.jld2`: Test data (91 days, 8,736 time steps)
- `cv.jld2`: Cross-validation data with moderate difficulty
- `cv_hard.jld2`: Cross-validation data with increased difficulty
- `pdrive_MPC.jld2`: EV demand profiles from MPC expert

## Data Structure

Each output dataset contains processed versions of:

- PV generation profiles
- Electricity buying/selling prices
- Electrical and thermal loads
- EV availability signal (γ)
- EV departure and arrival times
- EV power demand (when leaving the MCES)

## 
