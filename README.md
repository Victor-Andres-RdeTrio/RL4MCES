# Reinforcement Learning for Multi-Carrier Energy Management Systems 🌱⚡

## Overview

This repository contains the implementation and research findings of a **Reinforcement Learning (RL)-based Energy Management System (EMS)** for multi-carrier residential buildings. The project was conducted at **TU Delft's Green Village**, focusing on integrating and optimizing multiple energy carriers including electricity, heat, and mobility.

### Key Components
- 🔋 Lithium-ion battery storage
- 🚗 Electric Vehicle (EV) integration
- ☀️ Photovoltaic and solar thermal systems
- 🌡️ Heat pump and thermal storage
- 🤖 RL-based energy management system 


## Key Findings

The research validates RL as a viable alternative to traditional MPC approaches for residential energy management:

### Advantages
- **Computational Efficiency**: Reduced resource requirements
- **Accessibility**: Lower barrier to entry for developers
- **Performance**: Near-optimal results 

### Applications
- Residential buildings
- Office complexes
- Small-scale multi-carrier energy systems

## Technical Implementation

The project is implemented in **Julia**, leveraging the `ReinforcementLearning.jl` package. While the package provided fundamental structures for:
- Agent abstractions
- Environment modeling
- Policy definitions
- Runtime staging

Due to ongoing package refactoring, the **VPG with Critic, A2CGAE and PPO** policy update algorithms were custom-implemented to maintain Julia's performance benefits.

The codebase makes use of Julia's performance benefits through:
- Efficient memory management for policy updates
- Optimized tensor operations for DNN computations
- Reduced computational overhead during deployment

## Research Highlights

This section summarises the **key contributions** and **findings** of the thesis.

### Performance & Safety
- Achieved **96% cost efficiency** in grid exchange costs compared to the MPC (Expert) benchmark
- Demonstrated **improved safety constraint adherence**, with negligible operational projections across critical components
- Achieved **superior performance** in maintaining Thermal Energy Storage System (TESS) State of Charge (SoC) bounds
- Maintained **consistent performance** throughout the test set
- Satisfied EV charging demands while ensuring operational safety

### Computational & Implementation Benefits
- **Significantly reduced computational costs** during deployment
- Achieved **real-time operation capability**
- Avoids explicit physics-based modeling for decision making (which was the white-box MPC approach)
- Demonstrated effectiveness for practitioners without deep knowledge of optimal control theory

### Technical Innovations
- Investigated the impact of advanced RL policy update algorithms for EMS
- Analyzed Deep Neural Network (DNN) architectures
- Explored temporal feature engineering strategies
- Developed reward component shaping approaches, treating combinations of reward components as a hyperparameter
- Implementation of safe projection mechanism

### Comprehensive Analysis
- Provided detailed literature review of RL applications in Energy Management Systems (EMS)
- Conducted direct comparison between MPC and RL approaches
- Evaluated performance across multiple metrics:
  - Operational safety
  - Grid exchange costs
  - EV State of Charge (SoC) demand satisfaction
  - Energy storage utilization patterns

## Repository Structure

[*Note: Add your repository structure here*]

## Getting Started

[*Note: Add installation and usage instructions here*]

## Citation

If you use this work in your research, please cite:

[*Note: Add citation information here*]

## License
This repository is licensed under the MIT License. If you use this code or research in your work, please provide proper attribution by citing this repository or the corresponding author. 


## Contact
- MSc Thesis author: Victor Andrés -> vandres.trio@proton.me

## Master Thesis Abstract  
**Decarbonisation** and the need to reduce living expenses have driven the integration of **Multicarrier Energy Systems (MCES)**, where *electricity, heat, and mobility* converge. These systems demand **sophisticated energy management strategies** to address uncertainties in energy supply, demand, and weather conditions.  

This thesis develops a **Reinforcement Learning (RL)-based Energy Management System (EMS)** for a multicarrier residential building at **TU Delft's Green Village**. The case study household integrates:  
- **Photovoltaic and solar thermal systems**,  
- An **electric vehicle (EV)**,  
- A **lithium-ion battery**,  
- **Heat pump** 
- **Thermal Energy Storage System** 
- A **Power Electronic Interface** 

Current management strategies employ **white-box Model Predictive Control (MPC)**, which is effective but computationally intensive and requires **expertise in optimal control** and physics-based modelling.  

The **RL-based EMS** presents a *computationally efficient, data-driven alternative* that learns system dynamics autonomously. Benchmarking against the **Expert** (a day-ahead MPC planner), the RL agent demonstrated:  
- Comparable **operational performance**,  
- **Only a 4% increase** in grid energy exchange costs,  
- Improved **EV state of charge (SoC) compliance**, and  
- Enhanced **safety constraint adherence**.  

A **literature review** of RL applications for residential EMS is also presented, alongside investigations into **advanced policy update algorithms**, **deep neural network architectures**, **temporal feature engineering**, and **reward shaping strategies**, to evaluate their impact on performance.  

The **RL-based EMS** effectively manages both **electrical** and **thermal subsystems** while maintaining **safety** and minimising costs. Its computational efficiency and reduced modelling requirements position it as a practical solution for **small-scale MCES applications** such as residential or office buildings, where **MPC** may not be feasible.  

---





[Previous sections remain the same until Technical Implementation]

## Technical Implementation

This repository implements a Reinforcement Learning-based Energy Management System in **Julia**, building upon the `ReinforcementLearning.jl` framework. The implementation focuses on computational efficiency and practical deployment for a Small Multi Carrier Energy System (MCES) in TU Delft's Green Village.  

### Core Components

#### Policy Gradient Implementations
- Custom implementation of advanced policy update algorithms
- Progressive complexity from basic advantage estimation to GAE
- Implementation of trust region constraints for optimization stability
- Efficient policy storage mechanisms for PPO implementation

#### Deep Neural Network Architectures
- Implementation of multiple DNN architectures:
  - Residual networks
  - Three-Branched structures
  - Deep Common-Weight (DCW) networks
- Flexible architecture configuration system
- Robust weight initialization strategies

#### Feature Engineering Framework
- Efficient temporal feature processing
- State information management system
- Cross-correlation analysis tools for state-action relationships
- Optimized feature vector configurations

#### Reward Component System
- Modular reward shaping framework
- Multiple reward versions for multi-objective optimization
- Implementation of margin reward mechanism
- Safety-constraint integration in reward calculation

### Performance Optimizations



### Key Features

- **Modular Design**: Easily extendable for different energy system configurations
- **Hyperparameter Optimization**: Integration with BOHB for efficient parameter tuning
- **Safety Mechanisms**: Implementation of safe projection mechanisms
- **Performance Monitoring**: Comprehensive metrics tracking and validation
- **Storage Management**: Sophisticated handling of multiple energy storage systems:
  - Electric Vehicle (EV) battery management
  - Battery Energy Storage System (BESS) control
  - Thermal Energy Storage System (TESS) optimization

### Development Tools

[*Note: Add specific development tools, testing frameworks, and documentation generators used in your project*]

### Package Dependencies

[*Note: List key Julia packages and versions required for the project*]

[Rest of the README remains the same...]