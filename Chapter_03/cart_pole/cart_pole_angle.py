from time import sleep
import gymnasium as gym
import random

env = gym.make("CartPole-v1", render_mode="human")

seed = 1
random.seed(seed)

episodes = 10

for i in range(episodes):
    state, info = env.reset(seed=i)
    reward_sum = 0
    while True:
        env.render()
        action = 1 if state[2] > 0 else 0
        state, reward, terminated, truncated, info = env.step(action)
        reward_sum += reward
        sleep(0.01)
        if terminated or truncated:
            print(f"Episode {i} reward: {reward_sum}")
            sleep(1)
            break

env.close()
