**Use REINFORCE to solve Corridor Gridworld**

REINFORCE utilizes the Monte Carlo method, which leads to high variance. When solving this problem, I initially couldn't train the model stably. Later, during debugging, I realized that I needed to reduce the alpha value. The current alpha is the largest value that allows for stable training.