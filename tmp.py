# (state,action,next_state,reward,done)
#    17     6          17      1    1
data = pickle.load(open('/home/zj/Desktop/4zj_HalfCheetah-v2_expert_traj.p', 'rb'))

state_feed = data[:, 0:17]
action_feed = data[:, 17:23]
next_state_feed = data[:, 23:40]

model = create_forward_model()

model.compile(optimizer=Adam(lr=1e-4),
              loss='mean_squared_error',
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

model_checkpoint = ModelCheckpoint('weights_one_model.{epoch:02d}-{val_loss:.2f}.hdf5',
                                   monitor='val_loss',  # here 'val_loss' and 'loss' are the same
                                   verbose=1,
                                   save_best_only=True,
                                   save_weights_only=True)

model.fit([state_feed, action_feed],
          next_state_feed,
          batch_size=50,
          epochs=100,
          verbose=1,
          validation_split=0.2,
          shuffle=True,
          callbacks=[tf_board, early_stop, model_checkpoint])