# Project Title
Autonomously Learning Systems - Model-free Reinforcement Learning

# Project Overview
This project focuses on implementing and analyzing model-free reinforcement learning algorithms to solve various tasks. The tasks involve grid world environments and function approximation techniques using neural networks, aiming to develop agents that learn optimal policies through interaction with their environments.

# Key Features
Grid World Environment (Frozen Lake)

Implemented the SARSA (State-Action-Reward-State-Action) algorithm to solve the FrozenLake environment from the OpenAI Gym package.
Transitioned from deterministic to stochastic environments to evaluate algorithm robustness.
Analyzed the impact of parameters like ε (epsilon) and γ (gamma) on agent performance.
Enhanced the reward structure to improve learning efficiency.
Function Approximation for Continuous State Spaces

Utilized linear models and neural networks for Q-function approximation in environments with continuous state spaces.
Implemented SARSA and Q-learning algorithms using PyTorch to handle environments like CartPole.
Experimented with different neural network architectures and hyperparameters to optimize performance.
# Implementation Details
Frozen Lake (Grid World) Tasks:

Implemented SARSA and Q-learning algorithms.
Tested the influence of ε and γ on the average accumulated reward.
Modified rewards to guide the agent more effectively in challenging environments.
Non-tabular Q Function Tasks:

Implemented SARSA with a linear model for the CartPole environment.
Extended the Q function to a neural network with hidden layers to enhance learning capabilities.
Conducted experiments with different gym environments (MountainCar-v0, Pendulum-v0, Acrobot-v1) using both linear models and neural networks.
# Learning Outcomes
Understanding of Reinforcement Learning Algorithms:

Gained in-depth knowledge of on-policy (SARSA) and off-policy (Q-learning) algorithms.
Learned to balance exploration and exploitation through parameter tuning.
Skill Development in Python and PyTorch:

Enhanced coding skills in Python, focusing on clean, efficient, and effective code.
Developed proficiency in PyTorch for implementing neural networks and reinforcement learning models.
Critical Analysis and Optimization:

Analyzed the impact of various parameters and reward structures on the learning process.
Optimized algorithms for better performance in both discrete and continuous state spaces.
# Conclusion
This project showcases the implementation and analysis of fundamental model-free reinforcement learning algorithms. Through a series of experiments in different environments, it highlights the challenges and strategies in developing autonomous learning systems. The insights gained contribute to the broader understanding and application of reinforcement learning in more complex and real-world scenarios.
