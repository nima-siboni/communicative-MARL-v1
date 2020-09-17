import pdb
import numpy as np
import tensorflow as tf
from keras import layers
import keras
from utilfunctions import scale_state
from utilfunctions import one_hot


class Agent():
    '''
    A class for the agents

    the agent class which has the Q nets:
    one is the one whihc is learned and the other one is the target one
    The Q nets have similar structures. The input for the Qnetwork is
    (state, action) pair and the output is the Q value for the input.
    The dimesion of the input (None, nr_features+nr_actions) and the
    outputs are of shape (None, 1), where None is the batchsize dimension

    '''

    def __init__(self, nr_features=2, nr_actions=2, nr_comm_condensed=1, gamma=0.99, stddev=0.2, learning_rate=0.01, agent_id=0):
        '''
        initializes the Q nets
        the random seed is set to the agent_id, to assure the heterogenous initialization of the agents
        '''
        self.nr_features = nr_features
        self.nr_actions = nr_actions
        self.nr_comm_condensed = nr_comm_condensed
        # the Q network
        initializer_Q = tf.keras.initializers.RandomNormal(mean=0.0, stddev=stddev, seed=agent_id)
        identity = tf.keras.initializers.Identity()
        optimizer_Q = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        initializer_Q_t = tf.keras.initializers.RandomNormal(mean=0.0, stddev=stddev, seed=agent_id)
        optimizer_Q_t = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        # Here it starts
        inputs_comm = keras.layers.Input(shape=(nr_features), name='Q_input_comm')
        inputs_sa = keras.layers.Input(shape=(nr_features + nr_actions), name='Q_input_sa')

        # comm part
        x_c = layers.Dense(nr_features, activation='linear', kernel_initializer=initializer_Q, name='Q_com_lin_den_1')(inputs_comm)
        x_c = layers.Dense(16, activation='sigmoid', kernel_initializer=initializer_Q, name='Q_com_sigmoid_den_1')(x_c)
        x_c = layers.Dense(16, activation='sigmoid', kernel_initializer=initializer_Q, name='Q_com_sigmoid_den_2')(x_c)
        x_c = layers.Dense(16, activation='sigmoid', kernel_initializer=initializer_Q, name='Q_com_sigmoid_den_3')(x_c)
        x_c = layers.Dense(nr_comm_condensed, activation='linear', kernel_initializer=initializer_Q, name='Q_com_lin_den_2')(x_c)

        x_all = layers.Concatenate(name='concat')(
            [inputs_sa,
             x_c])

        x_all = layers.Dense(nr_comm_condensed + nr_features + nr_actions, activation='linear', kernel_initializer=initializer_Q, name='Q_sac_lin_den_1')(x_all)
        x_all = layers.Dense(128, activation='relu', kernel_initializer=initializer_Q, name='Q_sac_relu_den_1')(x_all)
        x_all = layers.Dense(64, activation='relu', kernel_initializer=initializer_Q, name='Q_sac_relu_den_2')(x_all)
        x_all = layers.Dense(32, activation='relu', kernel_initializer=initializer_Q, name='Q_sac_relu_den_3')(x_all)
        x_all = layers.Dense(16, activation='relu', kernel_initializer=initializer_Q, name='Q_sac_relu_den_4')(x_all)
        x_all = layers.Dense(1, activation='linear', kernel_initializer=initializer_Q, name='Q_sac_lin_den_2')(x_all)

        self.Q = keras.Model(inputs=[inputs_sa, inputs_comm], outputs=x_all, name='Q_model')
        self.Q.compile(optimizer=optimizer_Q, loss=['mse'])
        # EnD

        # Now lets define the Q-target

        inputs_t_comm = keras.layers.Input(shape=(nr_features), name='Qt_input_comm')
        inputs_t_sa = keras.layers.Input(shape=(nr_features + nr_actions), name='Qt_input_sa')

        # comm part
        x_t_c = layers.Dense(nr_features, activation='linear', kernel_initializer=initializer_Q_t, name='Qt_com_lin_den_1')(inputs_t_comm)
        x_t_c = layers.Dense(16, activation='sigmoid', kernel_initializer=initializer_Q_t, name='Qt_com_sigmoid_den_1')(x_t_c)
        x_t_c = layers.Dense(16, activation='sigmoid', kernel_initializer=initializer_Q_t, name='Qt_com_sigmoid_den_2')(x_t_c)
        x_t_c = layers.Dense(16, activation='sigmoid', kernel_initializer=initializer_Q_t, name='Qt_com_sigmoid_den_3')(x_t_c)
        x_t_c = layers.Dense(nr_comm_condensed, activation='linear', kernel_initializer=initializer_Q_t, name='Qt_com_lin_den_2')(x_t_c)

        x_t_all = layers.Concatenate(name='concat')(
            [inputs_t_sa,
             x_t_c])

        x_t_all = layers.Dense(nr_comm_condensed + nr_features + nr_actions, activation='linear', kernel_initializer=initializer_Q_t, name='Qt_sac_lin_den_1')(x_t_all)
        x_t_all = layers.Dense(128, activation='relu', kernel_initializer=initializer_Q_t, name='Qt_sac_relu_den_1')(x_t_all)
        x_t_all = layers.Dense(64, activation='relu', kernel_initializer=initializer_Q_t, name='Qt_sac_relu_den_2')(x_t_all)
        x_t_all = layers.Dense(32, activation='relu', kernel_initializer=initializer_Q_t, name='Qt_sac_relu_den_3')(x_t_all)
        x_t_all = layers.Dense(16, activation='relu', kernel_initializer=initializer_Q_t, name='Qt_sac_relu_den_4')(x_t_all)
        x_t_all = layers.Dense(1, activation='linear', kernel_initializer=initializer_Q_t, name='Qt_sac_lin_den_2')(x_t_all)

        self.Q_t = keras.Model(inputs=[inputs_t_sa, inputs_t_comm], outputs=x_t_all, name='Q_t_model')
        self.Q_t.compile(optimizer=optimizer_Q_t, loss=['mse'])
        
        self.gamma = gamma
        self.terminated = False

    def update_Q_t_to_Q(self):
        '''
        set the weights of Q_t to the weights of Q
        '''
        self.Q_t.set_weights(self.Q.get_weights())

    def update_Q_to_Q_t(self):
        '''
        set the weights of Q_t to the weights of Q
        '''
        self.Q.set_weights(self.Q_t.get_weights())

    def action_based_on_Q_target(self, agent_state, comm, env, epsilon):
        '''
        takes an action based on the epsilon greedy policy using the Q-target
        1 - for each agent_state/comm checks the predicted Q values for all the actions
        2 - pick the largest Q value
        3 - pick an action based on the largest Q value and epsilon

        Keyword arguments:

        agent_state -- current agent_state
        env -- the environment
        epsilon -- the epsilon in epsilon greedy approach

        returns:

        the id of the chosen action
        '''

        debug = False
        nr_samples = 1
        nr_actions = self.nr_actions
        scaled_state = scale_state(agent_state, env)
        scaled_state = np.array(scaled_state)
        scaled_state = np.reshape(scaled_state, (nr_samples, -1))

        # this part might need to change, depending on what is the communicated.
        scaled_comm = scale_state(comm, env)
        scaled_comm = np.array(scaled_comm)
        scaled_comm = np.reshape(scaled_comm, (nr_samples, -1))

        if debug:
            print("scaled_state", scaled_state)
            pdb.set_trace()

        for action_id in range(nr_actions):

            action = one_hot(action_id, nr_actions=nr_actions)

            inputs_for_Q_t = {
                'Qt_input_sa': np.concatenate((scaled_state, action), axis=1),
                'Qt_input_comm': scaled_comm}

            if action_id == 0:
                tmp = self.Q_t.predict(inputs_for_Q_t)
                if debug:
                    print("the predicted Q for action", action_id, " is ", tmp)
            else:
                tmp = np.concatenate((tmp, self.Q_t.predict(inputs_for_Q_t)), axis=1)
                if debug:
                    print("the predicted Q for action", action_id, " is ", tmp)

        tmp = tmp[0]
        probabilities = tf.math.softmax(tmp)
        probabilities = (probabilities + epsilon) / (1.0 + epsilon * nr_actions)
        probabilities = probabilities.numpy()
        probabilities = probabilities / np.sum(probabilities)
        if debug:
            print(probabilities, np.sum(probabilities) - 1)
        chosen_act = np.random.choice(nr_actions, p=probabilities)
        if debug:
            print("chosen_act", chosen_act)
            pdb.set_trace()

        return chosen_act

    def prepare_learning_materials(self, events, env):
        '''
        Creating the y vector for learning.
        The y vector is
        y(s,c,a) := r(s,c,a) + gamma * Q_t(s',c', argmax_a'(Q(s',c', a'))
        with  Q_t -- the target net

        Keyword arguments:
        events -- a list of events
        env -- the environment

        returns:
        y vector
        '''
        debug = False

        nr_samples = len(events)

        s_primes = [x.scaled_state_prime for x in events]
        s_primes = np.array(s_primes)
        s_primes = np.reshape(s_primes, (nr_samples, -1))

        c_primes = [x.scaled_comm_prime for x in events]
        c_primes = np.array(c_primes)
        c_primes = np.reshape(c_primes, (nr_samples, -1))

        r = [x.reward for x in events]
        r = np.array(r)
        r = np.reshape(r, (nr_samples, 1))

        done = [x.done for x in events]
        done = np.array(done)
        done = np.reshape(done, (nr_samples, 1))

        nr_actions = env.nr_actions

        if (debug):
            pdb.set_trace()

        for action_id in range(nr_actions):
            action = one_hot(action_id, nr_actions=env.nr_actions)
            actions = np.full((nr_samples, nr_actions), action)
            inputs_for_Q = {
                'Q_input_sa': np.concatenate((s_primes, actions), axis=1),
                'Q_input_comm': c_primes}

            if action_id == 0:
                tmp = self.Q.predict(inputs_for_Q)
            else:
                tmp = np.concatenate((tmp, self.Q.predict(inputs_for_Q)), axis=1)

        tmp = np.argmax(tmp, axis=1)

        if (debug):
            pdb.set_trace()

        inputs_for_Q_t = {
            'Qt_input_sa': np.concatenate((s_primes, tf.one_hot(tmp, depth=nr_actions)), axis=1),
            'Qt_input_comm': c_primes}

        y = r + self.gamma * self.Q_t.predict(inputs_for_Q_t) * (1 - done)

        return y

    def learn(self, events, env):
        '''
        fits the Q using events:
        1- creates the y vector
        2- creates the X vector which is made of the state/comm action pairs
        3- fits the Q network using X, y
        '''

        # 1
        y = self.prepare_learning_materials(events, env)

        # 2
        nr_samples = len(events)
        s = [x.scaled_state for x in events]
        s = np.array(s)
        s = np.reshape(s, (nr_samples, -1))

        c = [x.scaled_comm for x in events]
        c = np.array(c)
        c = np.reshape(c, (nr_samples, -1))

        actions = [x.action for x in events]
        actions = np.reshape(actions, (nr_samples, -1))

        X = np.concatenate((s, actions), axis=1)
        X = {
            'Q_input_sa': np.concatenate((s, actions), axis=1),
            'Q_input_comm': c}

        # my_callbacks = [
        #    tf.keras.callbacks.TensorBoard(log_dir='./logs'),
        # ]

        # 3
        # self.Q.fit(X, y, epochs=1, verbose=0, callbacks=my_callbacks)
        self.Q.fit(X, y, epochs=1, verbose=0)
        # import pdb; pdb.set_trace()
