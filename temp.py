# =================== define forward model ===================
# state + action  -> next_state
def create_forward_model():

    f_state = Input(shape=(17,))
    f_action = Input(shape=(6,))

    F_dense_s_1 = Dense(50, activation='relu', name='F_dense_s_1')(f_state)
    F_dense_s_2 = Dense(50, activation='relu', name='F_dense_s_2')(F_dense_s_1)
    F_dense_s_3 = Dense(50, activation='relu', name='F_dense_s_3')(F_dense_s_2)

    F_dense_a_1 = Dense(50, activation='relu', name='F_dense_a_1')(f_action)
    F_dense_a_2 = Dense(50, activation='relu', name='F_dense_a_2')(F_dense_a_1)
    F_dense_a_3 = Dense(50, activation='relu', name='F_dense_a_3')(F_dense_a_2)

    F_input_1 = Concatenate(axis=-1, name='F_input_1')([F_dense_s_3, F_dense_a_3])

    F_input_2 = Dense(50, activation='relu', name='F_input_2')(F_input_1)
    F_input_3 = Dense(50, activation='relu', name='F_input_3')(F_input_2)
    F_input_4 = Dense(50, activation='relu', name='F_input_4')(F_input_3)
    F_next_state_output = Dense(17, name='F_next_state_output')(F_input_4)

    MODEL_F = Model(inputs=[f_state, f_action], outputs=F_next_state_output)

    return MODEL_F

# =================== define backward model ===================
# state + next_state -> action
def create_backward_model():

    b_state = Input(shape=(17,))
    b_next_state = Input(shape=(17,))

    B_dense_s_1 = Dense(50, activation='relu', name='B_dense_s_1')(b_state)
    B_dense_s_2 = Dense(50, activation='relu', name='B_dense_s_2')(B_dense_s_1)
    B_dense_s_3 = Dense(50, activation='relu', name='B_dense_s_3')(B_dense_s_2)

    B_dense_a_1 = Dense(50, activation='relu', name='B_dense_a_1')(b_next_state)
    B_dense_a_2 = Dense(50, activation='relu', name='B_dense_a_2')(B_dense_a_1)
    B_dense_a_3 = Dense(50, activation='relu', name='B_dense_a_3')(B_dense_a_2)

    B_input_1 = Concatenate(axis=-1, name='B_input_1')([B_dense_s_3, B_dense_a_3])

    B_input_2 = Dense(50, activation='relu', name='B_input_2')(B_input_1)
    B_input_3 = Dense(50, activation='relu', name='B_input_3')(B_input_2)
    B_input_4 = Dense(50, activation='relu', name='B_input_4')(B_input_3)
    B_action_output = Dense(6, name='B_action_output')(B_input_4)

    MODEL_B = Model(inputs=[b_state, b_next_state], outputs=B_action_output)

    return MODEL_B

# =================== define recover model ===================
# action + next_state -> state
def create_recover_model():

    r_action = Input(shape=(6,))
    r_state = Input(shape=(17,))

    R_dense_a_1 = Dense(50, activation='relu', name='R_dense_a_1')(r_action)
    R_dense_a_2 = Dense(50, activation='relu', name='R_dense_a_2')(R_dense_a_1)
    R_dense_a_3 = Dense(50, activation='relu', name='R_dense_a_3')(R_dense_a_2)

    R_dense_s_1 = Dense(50, activation='relu', name='R_dense_s_1')(r_state)
    R_dense_s_2 = Dense(50, activation='relu', name='R_dense_s_2')(R_dense_s_1)
    R_dense_s_3 = Dense(50, activation='relu', name='R_dense_s_3')(R_dense_s_2)

    R_input_1 = Concatenate(axis=-1, name='R_input_1')([R_dense_s_3, R_dense_a_3])

    R_input_2 = Dense(50, activation='relu', name='R_input_2')(R_input_1)
    R_input_3 = Dense(50, activation='relu', name='R_input_3')(R_input_2)
    R_input_4 = Dense(50, activation='relu', name='R_input_4')(R_input_3)
    R_original_state_output = Dense(17, name='R_original_state_output')(R_input_4)

    MODEL_R = Model(inputs=[r_action, r_state], outputs=R_original_state_output)

    return MODEL_R
