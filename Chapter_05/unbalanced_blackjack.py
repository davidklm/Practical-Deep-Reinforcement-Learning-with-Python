import random
import numpy as np
from blackjack_hidden_deck_env import BlackjackHiddenDeckEnv
from policy import q_greedy_policy
from q_monte_carlo import train_q_monte_carlo
from plot_q import plot_q

hidden_deck = {
    1:  10,
    2:  30,
    3:  30,
    4:  20,
    5:  10,
    6:  5,
    7:  3,
    8:  1,
    9:  1,
    10: 1,
    10: 1,
    10: 1,
    10: 1
}
env = BlackjackHiddenDeckEnv(hidden_deck)

seed = 6
random.seed(seed)
env.seed(seed)
np.random.seed(seed)

Q = train_q_monte_carlo(env, 200_000)
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

print(f'Wins: {wins} | Loss: {loss} | Draws: {draws}')
