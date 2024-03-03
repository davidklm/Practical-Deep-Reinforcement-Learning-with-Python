import gymnasium as gym

env = gym.make('CartPole-v1')
env.reset()
state, reward, terminated, truncated, info = env.step(1)
