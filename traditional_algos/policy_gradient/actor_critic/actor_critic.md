**Actor-Critic** is a reinforcement learning method that combines policy optimization and value estimation. It consists of two main components:

- Actor: Responsible for selecting actions and updating the policy direction. It uses policy gradient methods to directly learn the optimal policy, providing a probability distribution of actions based on the current state.

- Critic: Responsible for evaluating the value of actions, typically by estimating the action-value function or state-value function, which guides the Actor's updates. The Critic uses Temporal Difference (TD) error to calculate value estimates, helping the Actor update its policy more effectively.

The core idea of Actor-Critic is that the Actor uses the value information provided by the Critic to reduce the variance of gradients, thereby accelerating policy learning while retaining the flexibility of policy optimization.