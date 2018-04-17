from keras.models import Model
from keras.layers import Input, Dense, Concatenate,Average
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras.optimizers import *
from keras.utils.vis_utils import plot_model
import pickle
import gym
import numpy as np



def behavior_cloning():

    state = Input(shape=(17,))

    fc1 = Dense(50, activation='relu')(state)
    fc2 = Dense(50, activation='relu')(fc1)
    fc3 = Dense(50, activation='relu')(fc2)
    fc4 = Dense(50, activation='relu')(fc3)
    fc5 = Dense(50, activation='relu')(fc4)
    fc6 = Dense(50, activation='relu')(fc5)
    fc7 = Dense(50, activation='relu')(fc6)
    fc8 = Dense(50, activation='relu')(fc7)
    fc9 = Dense(50, activation='relu')(fc8)
    output = Dense(6)(fc9)

    MODEL = Model(inputs=state, outputs=output, name = 'behavior_cloning')

    return MODEL

def train(model):

    # (state,action,next_state,reward,done)
    #    17     6          17      1    1
    data = pickle.load(open('/home/zj/Desktop/4zj_HalfCheetah-v2_expert_traj.p', 'rb'))

    state_feed = data[:, 0:17]
    action_feed = data[:, 17:23]

    model.compile(optimizer = Adam(lr = 1e-4),
                              loss = 'mean_squared_error',
                              metrics=['mse'])

    tf_board = TensorBoard(log_dir='./logs',
                           histogram_freq=0,
                           write_graph=True,
                           write_images=False,
                           embeddings_freq=0,
                           embeddings_layer_names=None,
                           embeddings_metadata=None)

    early_stop = EarlyStopping(monitor='val_loss',
                               patience=2,
                               verbose=0,
                               mode='auto')

    model_checkpoint = ModelCheckpoint('behavior_cloning.{epoch:02d}-{val_loss:.2f}.hdf5',
                                       monitor='val_loss',                        # here 'val_loss' and 'loss' are the same
                                       verbose=1,
                                       save_best_only=True,
                                       save_weights_only=True)

    model.fit(state_feed,
                          action_feed,
                            batch_size=50,
                            epochs=100,
                            verbose=1,
                            validation_split=0.2,
                            shuffle=True,
                            callbacks=[tf_board, early_stop , model_checkpoint])


def test(model_for_25_nets):
    env = gym.make('HalfCheetah-v2')

    model_for_25_nets.compile(optimizer=Adam(lr=1e-4),
                              loss='mean_squared_error',
                              metrics=['mse'])

    model_for_25_nets.load_weights('behavior_cloning.40-0.10.hdf5', by_name=True)


    two_state = np.zeros((2,17))
    observation = env.reset()
    two_state[0,:] = observation

    # print(two_state)

    for _ in range(1000):

        env.render()
        action_two = model_for_25_nets.predict_on_batch(two_state)
        action = action_two[0,:]

        for i in range(6):
            if action[i] < -1 :
                action[i] = -1
            elif action[i] > 1:
                action[i] = 1

        observation, reward, done, info = env.step(action)
        two_state[0, :] = observation

        print(reward)




if __name__ == '__main__':

    model = behavior_cloning()
    # train(model)
    test(model)