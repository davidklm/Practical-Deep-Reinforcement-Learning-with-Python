from time import sleep
import gymnasium as gym
import random

env = gym.make("CartPole-v1", render_mode="human")

seed = 1
random.seed(seed)

print(f"Action Space: {env.action_space}")
print(f"Observation Space: {env.observation_space}")

episodes = 10

for i in range(episodes):
    init_state = env.reset(seed=i)
    reward_sum = 0
    while True:
        env.render()
        random_action = random.randint(0, 1)
        state, reward, terminated, truncated, info = env.step(random_action)
        reward_sum += reward
        sleep(0.01)
        if terminated or truncated:
            print(f"Episode {i} reward: {reward_sum}")
            sleep(1)
            break

env.close()
