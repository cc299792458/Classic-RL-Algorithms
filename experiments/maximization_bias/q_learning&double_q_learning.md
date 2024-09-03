**Use Q-Learning and Double Q-Learning to solve Maximization Bias**

From this experiment, we can clearly see the overestimation inherent in Q-Learning. Using Double Q-Learning successfully addresses this issue.

Due to the stochastic nature of this environment, I discovered during debugging that a sufficiently small alpha is necessary to ensure stable training. Otherwise, unexpected behaviors can occur. For instance, with alpha set to 0.1, even when the environment sets the mean reward for moving left to 0.1, the learned probability of moving left may gradually decrease. I suspect this is because a larger alpha prevents the algorithm from accurately learning the correct mean, leading to instability.