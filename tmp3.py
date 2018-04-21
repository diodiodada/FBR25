import pickle
import gym
import numpy as np

env = gym.make('FetchReach-v0')



observation = env.reset()
state = observation["observation"]
done = False

action = np.ndarray((4,))

while True:
    # env.render()
    # action = env.action_space.sample()

    a = input("输入: ")
    a = a.split(" ")
    action[0] = a[0]
    action[1] = a[1]
    action[2] = a[2]
    action[3] = a[3]

    observation, reward, done, info = env.step(action)
    print(observation['achieved_goal'])
    next_state = observation["observation"]



