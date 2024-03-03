from time import sleep
import gymnasium as gym

env = gym.make("Phoenix-v0", render_mode="human")

episodes = 10

for i in range(episodes):
    init_state = env.reset()
    reward_sum = 0
    while True:
        env.render()
        random_action = env.action_space.sample()
        state, reward, terminated, truncated, info = env.step(random_action)
        reward_sum += reward
        sleep(0.01)
        if terminated or truncated:
            print(f"Episode {i} reward: {reward_sum}")
            sleep(1)
            break

env.close()
