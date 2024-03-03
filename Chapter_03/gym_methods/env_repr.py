import gymnasium as gym
import random

env = gym.make('Blackjack-v1')

seed = 1
random.seed(seed)
env.reset(seed=seed)
