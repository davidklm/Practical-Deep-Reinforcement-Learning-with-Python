import numpy as np


def monte_carlo_policy(env):
    state_history = []
    state, info = env.reset()
    actions = [0, 1]
    while True:
        player_sum = state[0]
        if player_sum < 12:
            action = 1
        else:
            probs = [0.8, 0.2] if player_sum > 17 else [0.2, 0.8]
            action = np.random.choice(actions, p = probs)

        next_state, reward, terminated, truncated, _ = env.step(action)
        state_history.append([state, action, reward, next_state])
        state = next_state
        if terminated or truncated:
            break

    return state_history


def q_greedy_policy(env, Q):
    state, info = env.reset()
    while True:
        agent_sum = state[0]
        if agent_sum < 12:
            action = 1
        else:
            action = np.argmax(Q[state])
        next_state, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            break

        state = next_state

    return reward
