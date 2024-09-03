**N-Step Sarsa** is an intermediate approach between Sarsa and Monte Carlo, updating the action-value function after a fixed number of steps. This method balances the bias-variance trade-off by incorporating more information than one-step Sarsa while updating more frequently than Monte Carlo. By influencing up to N states in a single update, N-Step Sarsa can achieve better performance than either one-step Sarsa or Monte Carlo, especially when an appropriate value for N is chosen.