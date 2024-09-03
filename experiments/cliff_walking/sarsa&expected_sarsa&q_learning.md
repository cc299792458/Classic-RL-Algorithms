**Use Sarsa, Expected Sarsa and Q-Learning to solve Cliff Walking**

This example perfectly illustrates why Q-Learning is an off-policy algorithm. It shows that Q-Learning consistently learns the optimal policy for the environment, regardless of the epsilon-greedy behavior policy being used. Because of that, it may end up with a policy that doesn’t achieve the highest reward under that specific behavior policy, unlike Sarsa.

Sarsa, on the other hand, is an on-policy method, which means it learns the best policy according to the behavior policy used during training. Similarly, Expected Sarsa is also an on-policy learning algorithm. From the plots of Sarsa and Expected Sarsa, we can observe that the latter effectively reduces variance during the learning process.

Additionally, this experiment highlights the importance of setting an appropriate alpha (learning rate) for effective learning in TD methods.