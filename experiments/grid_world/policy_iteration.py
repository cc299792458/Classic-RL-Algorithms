import time

from envs.grid_world import GridWorld
from traditional_algos.dynamic_programming.policy_iteration.policy_iteration import PolicyIteration

if __name__ == '__main__':
    ##### Step 0: Build environment and initiate policy #####
    env = GridWorld()

    theta = 1e-4
    agent = PolicyIteration(env=env, theta=theta)

    ##### Step 1: Try policy evaluation with a random policy #####
    ### Step 1.1: Update value function inplacely ---> 114 iterations
    inplace = True
    agent.policy_evaluation(inplace=inplace, print_each_iter=True)
    
    ### Step 1.2: Update value function non-inplacely ---> 173 iterations
    agent.reset()
    inplace = False
    agent.policy_evaluation(inplace=inplace, print_each_iter=True)

    #### Step 2: Get the Optimal Policy using Policy Iteration #####
    ## Step 2.1 Record iterations ###
    agent.reset()
    agent.iterate(eval_print_flag=True, impro_print_flag=True)  # 3 iterations
    
    ### Step 2.2 Record time consuming ###
    agent.reset()
    start_time = time.time()
    agent.iterate()
    end_time = time.time()
    print(f"Time consuming: {end_time-start_time}") # about 9e-3 seconds