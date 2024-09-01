# **Classic-RL-Algorithms**

This repository provides a comprehensive collection of classic reinforcement learning algorithms. It features simple implementations of **traditional methods** such as **dynamic programming, monte carlo methods, and td-learning**, including both **tabular approaches and those using function approximation.** All classic algorithms here have been tested using environments from [Sutton and Barto's book: "Reinforcement Learning: An Introduction"](http://incompleteideas.net/book/the-book-2nd.html). Additionally, this repository will include **algorithms implemented with deep neural networks**, covering both **Q-function-based methods and policy gradient methods**. Some of these implementations have borrowed code from the homworks of [Berkeley's CS 285 course on Deep Reinforcement Learning](https://rail.eecs.berkeley.edu/deeprlcourse/). 

What's more, this repository includes notes for each algorithm to share my **findings from the experiments**. For beginners, these notes provide insights into why certain algorithms are used, their advantages and disadvantages, and potential issues you might encounter during implementation. Additionally, in the **Blog** folder, you can find some of my reflections on reinforcement learning. 

This repository is still in development, with the traditional algorithms section largely complete. Currently, I am working on implementing the deep reinforcement learning algorithms. In the future, I plan to add more algorithms, including those related to model-based RL, visual RL, and real-world RL.

**Reinforcement learning is fascinating, and I wish this repository will be helpful to others who share the same interest!**

## The list of traditional ones is:
- [Epsilon Greedy](https://github.com/cc299792458/Classic-RL-Algorithms/tree/main/traditional_algos/epsilon_greedy)
- Dynamic Programming
  - Policy Iteration 
  - Value Iteration
- Monte Carlo Methods 
  - Monte Carlo, Gradient Monte Carlo
- Temporal Difference Learning 
  - SARSA, Expected-SARSA, N-Step SARSA, Semi-Gradient SARSA
  - Q-Learning: Q-Learning, Double Q-Learning
- Policy Gradient Methods:
  - REINFORCE, REINFORCE with Baseline 
  - Actor-Critic
