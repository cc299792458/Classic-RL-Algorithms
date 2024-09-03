Dynamic Programming (DP) refers to a set of algorithms designed to compute optimal policies when a perfect model of the environment is available, typically represented as a Markov Decision Process (MDP).

However, DP has two notable drawbacks: it assumes the availability of a perfect model and requires significant computational resources.

Despite its significant computational demands, it remains a relatively feasible method in practice.

Essentially, when the state transition function P of an environment is known, the optimal value function V can be obtained by solving a system of linear equations:
$V = R + \gamma P V \iff (I - \gamma P) V = R \iff V = (I - \gamma P)^{-1} R$

Dynamic Programming methods leverage the Policy Improvement Theorem and use iterative techniques to efficiently solve this system of linear equations.