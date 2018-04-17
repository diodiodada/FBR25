
import pickle
import gym
import numpy as np
env = gym.make('FetchReach-v0')

# state, action, next_state
#    10,      4,         10
data = np.zeros((10000,24))


observation = env.reset()
state = observation["observation"]

for i in range(10000):
    # env.render()
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
    next_state = observation["observation"]

    data[i,0:10] = state
    data[i,10:14] = action
    data[i,14:24] = next_state

    state = next_state

pickle.dump( data, open( "FetchReach-v0-sample2.p", "wb" ) )