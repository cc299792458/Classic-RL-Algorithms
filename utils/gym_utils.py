import gymnasium as gym

def get_observation_shape(observation_space):
    """ Determine the shape of the state space from the environment's observation space. """
    if isinstance(observation_space, gym.spaces.Tuple):
        # Handle tuple spaces (e.g., Blackjack's state)
        return tuple(space.n for space in observation_space.spaces)
    elif isinstance(observation_space, gym.spaces.Discrete):
        # Handle discrete spaces
        return (observation_space.n,)
    elif isinstance(observation_space, gym.spaces.Box):
        # Handle continuous spaces (assuming discretization is required)
        return observation_space.shape
    else:
        raise ValueError("Unsupported observation space type")