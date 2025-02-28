import random

import gymnasium as gym

from policy import q_greedy_policy
from q_monte_carlo import train_q_monte_carlo
from plot_q import plot_q
import numpy as np

env = gym.make('Blackjack-v1')

seed = 0
random.seed(seed)
np.random.seed(seed)

Q = train_q_monte_carlo(env, 50_000, gamma = .5)
plot_q(Q)

episodes = 1000
wins = 0
loss = 0
draws = 0

for e in range(1, episodes + 1):
    reward = q_greedy_policy(env, Q)
    if reward > 0:
        wins += 1
    elif reward < 0:
        loss += 1
    else:
        draws += 1

print(f'Wins: {wins} | Loss: {loss} | Draws: {draws}| Total: {episodes}')
