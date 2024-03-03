from time import sleep
import gymnasium as gym

env = gym.make('CartPole-v1')
env.reset()
env.render()
sleep(10)
