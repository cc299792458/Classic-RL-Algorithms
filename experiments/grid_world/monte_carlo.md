**Use Monte Carlo Method to solve a 4 by 4 Grid World**

In this experiment, an interesting finding is that without epsilon-greedy, the policy (can find the optimal path but) fails to thoroughly explore all actions. As a result, some states may have incorrect optimal actions (you can see this effect by comparing setting epsilon to 0 and to 0.1). This highlights the importance of the exploitation.