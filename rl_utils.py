from collections import deque
import numpy as np
import os
from random import sample
from utilfunctions import one_hot
from utilfunctions import scale_state
from utilfunctions import initializer
from utilfunctions import single_shape_adaptor
from utilfunctions import update_state_step


class event:
    '''
    the class event offers a container for 
    s, c, a, r, s_prime, c_prime tuple to gether with done
    
    one should note that the s, and s' are saved in scaled form
    (which is expected as the input for the networks)
    '''
    def __init__(self, state, comm, action_id, reward, state_prime, comm_prime, done, env):
        # converting the state, state_prime to scaled ones
        scaled_state = scale_state(state, env)
        scaled_state_prime = scale_state(state_prime, env)

        # This might need to be changed based on what is communicated
        scaled_comm = scale_state(comm, env)
        scaled_comm_prime = scale_state(comm_prime, env)
        # one_hot the action
        action = one_hot(action_id, nr_actions=env.nr_actions)

        self.scaled_state = scaled_state
        self.scaled_comm = scaled_comm
        self.action = action
        self.reward = reward
        self.scaled_state_prime = scaled_state_prime
        self.scaled_comm_prime = scaled_comm_prime
        self.done = done

    def print(self):
        '''
        printing the state
        '''
        print("scaled_state: ", self.scaled_state)
        print("scaled_comm: ", self.scaled_comm)
        print("action: ", self.action)
        print("reward: ", self.reward)
        print("scaled_state_prime: ", self.scaled_state_prime)
        print("scaled_comm_prime: ", self.scaled_comm_prime)
        print("done: ", self.done)


class Histories:
    '''
    a class for creation and manipulation of the buffer
    '''
    def __init__(self, max_size=10_000):
        self.size = 0
        self.events = deque([])
        self.max_size = max_size

    def reset_the_buffer(self):
        '''
        reset the buffer
        '''
        self.events = ([])
        self.size = 0

    def consider_this_event(self, event):
        if (self.size < self.max_size):
            self.fill_by_appending(event)
        else:
            self.roll_and_replace(event)

    def fill_by_appending(self, event):
        '''
        filling a new buffer or a resetted one by appending to it
        '''
        self.events.append(event)
        # import pdb; pdb.set_trace()
        if (len(self.events) > self.size):
            self.size += 1

    def roll_and_replace(self, event):
        '''
        rolls the buffer, pushing the oldest experience out and adding a new one at the end of the list
        '''
        self.events.rotate(-1)
        self.events[0] = event

    def return_a_batch(self, batchsize=32):
        '''
        returns a random batch from the bucket, note that it first shuffles the bucket
        and then picks the sampels.
        '''
        return sample(self.events, k=batchsize)


def extract_state_and_comm_from_global_state(global_state, agent_id):
    '''
    extract the state and the comm for a specific agent
    this part needs a better implementations such that works for any number of agents
    when that is done the assert expression can be removed.
    '''
    assert agent_id < 2

    if agent_id == 0:
        state = global_state[:, 0:2]
        comm = global_state[:, 2:4]

    if agent_id == 1:
        comm = global_state[:, 0:2]
        state = global_state[:, 2:4]

    return state, comm


def update_replay_buffer_with_episodes(nr_episodes, agents_lst, replay_buffer_lst, env, epsilon):
    '''
    fills the main_buffer with events, i.e. (s, a, r, s', done)
    which are happened during some rounds of experiments
    for the agent. The actions that the agent took are based on the Q-target network with epsilon greedy approach

    Keyword arguments:

    rounds_data_exploration -- number of experiment rounds done
    agent -- the agent
    main_buffer -- the replay buffer
    env -- environement
    epsilon -- the epsilon for the epsilon greedy approach

    returns:

    the replay buffer
    '''

    global_state_space_size = len(env.states)
    nr_features_per_agent = agents_lst[0].nr_features
    nr_comm_features_per_agent = agents_lst[0].nr_features
    # + agents_lst[0].nr_actions

    for training_id in range(nr_episodes):

        print("\nround: " + str(training_id))

        initial_state_global = env.reset()

        for agent_id in range(len(agents_lst)):
            agents_lst[agent_id].terminated = False

        state_global, terminated, steps = initializer(initial_state_global)
        state_global = single_shape_adaptor(state_global, global_state_space_size)

        while not terminated:

            action_id_lst = []
            # choosing actions for each agent
            for agent_id in range(env.nr_agents):
                agent = agents_lst[agent_id]
                state, comm = extract_state_and_comm_from_global_state(state_global, agent_id)
                action_id = agent.action_based_on_Q_target(state, comm, env, epsilon)
                action_id_lst.append(action_id)

            # STEP : taking a step with all the agents
            new_state_global, reward_lst, terminated_lst, info = env.step(action_id_lst)
            new_state_global = single_shape_adaptor(new_state_global, global_state_space_size)

            # extracting the current event for each agent and saving it in its replay buffer
            for agent_id in range(env.nr_agents):

                # extraction
                new_state, new_comm = extract_state_and_comm_from_global_state(new_state_global, agent_id)
                state, comm = extract_state_and_comm_from_global_state(state_global, agent_id)

                # reshaping ?
                state = single_shape_adaptor(state, nr_features_per_agent)
                comm = single_shape_adaptor(comm, nr_comm_features_per_agent)
                new_state = single_shape_adaptor(new_state, nr_features_per_agent)
                new_comm = single_shape_adaptor(new_comm, nr_comm_features_per_agent)

                action_id = action_id_lst[agent_id]

                # putting the data in event format
                this_event = event(state,
                                   comm,
                                   action_id,
                                   reward_lst[agent_id],
                                   new_state,
                                   new_comm,
                                   terminated_lst[agent_id],
                                   env)
                if np.array_equal(this_event.action, np.array([[0, 1]])) and np.array_equal(this_event.scaled_state, this_event.scaled_state_prime) and this_event.done == False:
                    import pdb; pdb.set_trace()
                # throwing the event into the replay buffer
                # as long as the agent is not terminated.
                # the next if which updates the state of the agent (i.e. agent.terminated)
                # is very important.
                # All these manuvers are done in order that the last step is written in the buffer,
                # but all the rest are not.
                # This is a multi-agent specific problem as the in the single agent case both the agent and
                # the environment finish at the same time.

                if agents_lst[agent_id].terminated == False:
                    replay_buffer_lst[agent_id].consider_this_event(this_event)

                if terminated_lst[agent_id] == True:
                    agents_lst[agent_id].terminated = True

            terminated = np.array(terminated_lst).all()
            state_global = new_state_global
            steps = steps + 1

        print("...    the terminal_state is reached after " + str(steps))

    return replay_buffer_lst


def testing_performance(nr_episodes, agents_lst, env, epsilon):
    ''' runs a number of episodes and returns the average performance
    The actions that the agent took are based on the Q-target network with epsilon greedy approach

    Keyword arguments:

    rounds_data_exploration -- number of experiment rounds done
    agent -- the agent
    main_buffer -- the replay buffer
    env -- environement
    epsilon -- the epsilon for the epsilon greedy approach

    returns:

    the the total rewards and the individual rewards
    '''
    print("...    the test is started")

    accumulative_rewards = np.zeros((1, env.nr_agents))
    global_state_space_size = len(env.states)

    for training_id in range(nr_episodes):

        print("\nround: " + str(training_id))

        initial_state_global = env.reset()

        state_global, terminated, steps = initializer(initial_state_global)
        state_global = single_shape_adaptor(state_global, global_state_space_size)

        while not terminated:

            action_id_lst = []
            # choosing actions for each agent
            for agent_id in range(env.nr_agents):
                agent = agents_lst[agent_id]
                state, comm = extract_state_and_comm_from_global_state(state_global, agent_id)
                action_id = agent.action_based_on_Q_target(state, comm, env, epsilon)
                action_id_lst.append(action_id)

            # taking a step with all the agents
            new_state_global, reward_lst, terminated_lst, _ = env.step(action_id_lst)
            new_state_global = single_shape_adaptor(new_state_global, global_state_space_size)

            accumulative_rewards += reward_lst

            terminated = np.array(terminated_lst).all()
            state_global = new_state_global
            steps = steps + 1

        print("...    the terminal_state is reached after " + str(steps))

    accumulative_rewards = accumulative_rewards / (nr_episodes + 0.0)

    output = np.array(accumulative_rewards.flatten())
    output = np.append(output, np.sum(accumulative_rewards))
    return output


def logging_performance(log, training_id, performance_info, write_to_disk=True):
    '''
    returns a log (a numpy array) which has some analysis of the training.

    Key arguments:

    training_id -- the id of the iteration which is just finished.
    peroformance info -- the total number of steps before failing
    write_to_disk -- a flag for writting the performance to the disk

    Output:

    a numpy array with info about the iterations and the learning
    '''

    performance_info = np.array(performance_info)
    performance_info = performance_info.flatten()
    if training_id == 0:
        tmp = np.array([training_id])
        tmp = np.append(tmp, performance_info)
        log = np.array([tmp])
    else:
        tmp = np.array([training_id])
        tmp = np.append(tmp, performance_info)
        log = np.append(log, np.array([tmp]), axis=0)

    if write_to_disk:
        perfdir = './performance-and-animations/'
        if not os.path.exists(perfdir):
            os.makedirs(perfdir)

        np.savetxt(perfdir + 'steps_vs_iteration.dat', log)

    return log
