What is Reinforcement Learning?

Chi Chu

Once upon a time, there is an agent. The agent can interact with the environment. Once it takes an action, the environment will return it a reward evaluating how good this action is, and the state of agent changes. This process will continue until the agent reaches the end of its life, for example, if the black pieces lose the Go game, or, if the agent is a robot, until it completes its task. The goal of reinforcement learning algorithms are to maximize the agent's total reward.

So here are some key concepts or elements of reinforcement learning:

Agent:

	Policy: Fundamentally, a policy is a mapping from state to action. It can be deterministic or stochastic, or it can depend on values (which we will explain shortly). Although the term ‘policy’ encompasses various types, such as optimal policy and target policy, when we refer to a ‘policy,’ it by defalut means the same thing as the "agent’s policy".

Environment:

	State:

	Action:

	Reward: Someone may be confuse about where does this reward come from? For example, in a go game, there isn't an critic to provide feedback and give you a reward for each move you make. So, for a beginner, we should note, reward is designed by us, and they can significantly influence the performance of reinforcement learning algorithms.

	Value:
	



Markov Decision Process:



Bellman Equation:
