import pickle
import gym
import numpy as np

env = gym.make('Reacher-v2')



observation = env.reset()



while True:
    env.render()
    action = env.action_space.sample()


    observation, reward, done, info = env.step(action)
    # print(observation['achieved_goal'])
    # next_state = observation["observation"]



