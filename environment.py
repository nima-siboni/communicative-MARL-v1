import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
from matplotlib import interactive


def get_the_max_min_passenger_count(states, nr_agents):
    lst = []
    for agent_id in range(nr_agents):
        lst.append(states[2 * agent_id + 1])
    return np.max(np.array(lst)), np.min(np.array(lst))


class Environment():
    '''
    the class environment with close resemblence to gym
    '''
    def __init__(self,
                 ls1=10,
                 lc=10,
                 ls2=10,
                 nr_agents=2,
                 states=[0.0, 1.0, 0.0, 10.0],
                 time_cost=1,
                 destinations=[29.0, 29.0],
                 conflict_cost=100.0,
                 nr_actions=2):
        '''
        keyword arguments:

        ls1 -- length of the first segment where the agents can allocate separately
        lc  -- length of the segment which is common
        ls2 -- length of the second separately allocable segment
        nr_agents -- the number of trains
        states -- a list of size (nr_agent * property per agent) 
        where every consequent two numbers present the position and the number of
        passengers in each train
        [0, 1, 0, 10] means:
        both agents are at position zero, and
        the 1st agent is currently at 1
        the 2nd agent is currently at 10
          __________               ___________      
         /          \             /            \
        /<---ls1-->  \____lc_____/  <--ls2 -->  \
        \            /           \              /
         \__________/             \____________/

        '''

        self.ls1 = ls1
        self.lc = lc
        self.ls2 = ls2
        self.nr_agents = nr_agents
        self.states = np.array(states).astype('float')
        self.time_cost = time_cost
        self.destinations = np.array(destinations).astype('float')
        self.terminated_lst = np.full(nr_agents, False)
        self.conflict_cost = float(conflict_cost)
        assert (self.destinations < ls1 + lc + ls2).all()
        self.low = 0.0
        self.high = ls1 + lc + ls2 - 1.0
        self.high_passenger, self.low_passenger = get_the_max_min_passenger_count(self.states, self.nr_agents)
        self.nr_actions = nr_actions
        self.figure = plt.figure(figsize=(6, 6))
        self.ax = self.figure.add_subplot()
        # self.figure, self.ax = plt.subplots()
        self.timestamp = 0

    def reset(self):
        '''
        reset the agents to zero positions
        '''
        for agent_id in range(self.nr_agents):
            self.states[2 * agent_id] = 0
            self.terminated_lst[agent_id] = False
        self.timestamp = 0
        return self.states

    def get_state_agent(self, agent_id):
        '''
        for the agent with id = agent_id
        return the position and number of passengers of the agent
        '''
#        return state(self.positions[id], self.n_list[agent_id])
        return self.states[2 * agent_id: 2 * agent_id + 2]

    def step_agent(self, agent_id, action_id):
        '''
        takes an step with action_id for agent_id
        and returns
        the new state, reward, terminated (for this agent) and info (empty)
        the reward is set to 0 if the agent reaches its terminal
        '''

        if self.terminated_lst[agent_id] == False:

            new_states = (self.states).astype('float') + 0.
            new_states[2 * agent_id] = new_states[2 * agent_id] + action_id

            nr_passengers = self.states[2 * agent_id + 1]
            cost_passenger_time = nr_passengers * self.time_cost + (1 - action_id) * nr_passengers * self.time_cost
            reward = -1.0 * cost_passenger_time

            if (new_states[2 * agent_id] == self.destinations[agent_id]):
                self.terminated_lst[agent_id] = True
                terminated = True
                reward = 0.0
            else:
                terminated = False

            self.states = new_states

        else:

            reward = 0.0
            terminated = True

        return self.states, reward, terminated, ['']

    def conflict_detector(self):
        '''
        finds out how many agents are conflicting
        '''
        agents_in_conflict_zones = 0
        for agent_id in range(self.nr_agents):
            pos = self.states[2 * agent_id]
            agents_in_conflict_zones += (pos >= self.ls1) and (pos < self.lc + self.ls1)

        if agents_in_conflict_zones == 0:
            return agents_in_conflict_zones
        else:
            return agents_in_conflict_zones - 1

    def step(self, action_id_lst):

        '''
        takes the actions for both agents and return an output similar to open-ai gym.
        Basically every input/output of the step function in open-ai gym is replaced here
        with a list. For example, step(action) --> step(action_lst), where action_lst is
        the list of actions associated to each agent to be taken at this step.

        keyword inputs:

        actions_id_lst -- list of action_ids to be taken by agents.

        returns:
        states -- the global state of the system after the actions
        reward_lst -- the list of reward of different agents
        terminated_lst -- the list of "done"s for all the agents

        side-effects:

        - the self.timestamp is increased by 1
        - the self.states is updated.

        takes the actions for both agents and return bela bela
        '''
        reward_lst = np.zeros(self.nr_agents)
        assert len(action_id_lst) == self.nr_agents

        for agent_id in range(self.nr_agents):

            action_id = action_id_lst[agent_id]
            _, reward_lst[agent_id], _, _ = self.step_agent(agent_id, action_id)

        conflicting_agents = self.conflict_detector()

        # Here the cost of the conflict is added to both of them with a distribution
        # equally punished would be [1, 1]
        # punish the second one which comes on the track is implemented here (ONLY for the case of two agents)
        assert self.nr_agents == 2

        if (self.states[0] > self.states[2]):
            conflict_reward_distribution = np.array([10, 1])
            #conflict_reward_distribution = np.array([0, 1])
        else:
            conflict_reward_distribution = np.array([1, 0])

        # conflict_reward_distribution = [1.0, 1.0]
        
        rewards = reward_lst - 1.0 * conflicting_agents * self.conflict_cost * conflict_reward_distribution
        self.timestamp += 1
        return self.states, rewards, self.terminated_lst, ['']

    def get_x_y(self, epsilon=0.1, extension_x_of_connections=0):

        x = []
        y = []
        x_original = []
        for agent_id in range(self.nr_agents):
            x_original.append(self.states[2 * agent_id])
            if (x_original[-1] < self.ls1):
                x.append(self.states[2 * agent_id])
                y.append(agent_id)
            elif (self.ls1 <= x_original[-1] and x_original[-1] < self.ls1 + self.lc):
                x.append(self.states[2 * agent_id] + extension_x_of_connections)
                y.append((self.nr_agents - 1) / 2. + epsilon * (agent_id - 1.0) / self.nr_agents)
            elif (x_original[-1] >= self.ls1 + self.lc):
                x.append(self.states[2 * agent_id] + 2 * extension_x_of_connections)
                y.append(agent_id)
        return np.array(x), np.array(y)

     
    def render(self, dir_name=None):
        '''
        a simple plotter
        '''
        train_length = 1
        plt.ion()
        interactive(True)
        plt.cla()
        self.ax.axis('off')
        # some useful constants
        epsilon = 0.1
        extension_x_of_connections = 2

        min_x = -1 * train_length
        max_x = self.ls1 + self.lc + self.ls2 + 2 * extension_x_of_connections
        max_y = self.nr_agents - 1 + epsilon
        min_y = 0 - epsilon

        nr_conflicts = self.conflict_detector()

        if nr_conflicts > 0:
            color = 'red'
        else:
            color = 'gray'

        # getting the number of passangers
        sizes_lst = []
        for agent_id in range(self.nr_agents):
            sizes_lst.append(self.states[2 * agent_id + 1])

        # getting the destinations
        destinations_y = np.array(range(self.nr_agents))
        destinations_x = np.array(self.destinations) + 2 * extension_x_of_connections

        # drawing the rails

        # the non-overlapping tracks
        for agent_id in range(self.nr_agents):
            self.ax.plot(
                [0, self.ls1 - 1],
                [agent_id, agent_id],
                linestyle='dashed', color='gray')

        # the first connections
        for agent_id in range(self.nr_agents):
            self.ax.plot(
                [self.ls1 - 1, self.ls1 + extension_x_of_connections - 1],
                [agent_id, (self.nr_agents - 1.0) / 2.0],
                linestyle='dashed', color='gray')

        # single track
        self.ax.plot(
            [self.ls1 + extension_x_of_connections - 1, self.ls1 + self.lc + extension_x_of_connections - 1],
            [(self.nr_agents - 1.0) / 2.0, (self.nr_agents - 1.) / 2.0],
            linestyle='dashed',
            color='gray')

        # the second connections
        for agent_id in range(self.nr_agents):
            self.ax.plot(
                [self.ls1 + self.lc + extension_x_of_connections - 1, self.ls1 + self.lc + 2 * extension_x_of_connections - 1],
                [(self.nr_agents - 1.0) / 2.0, agent_id],
                linestyle='dashed', color='gray')

        # the second non-overlapping
        for agent_id in range(self.nr_agents):
            self.ax.plot(
                [self.ls1 + self.lc + 2 * extension_x_of_connections - 1, self.ls1 + self.lc + self.ls2 + 2 * extension_x_of_connections],
                [agent_id, agent_id],
                linestyle='dashed', color='gray')

        # autoscale on unknown axis and known lims on the other
        self.ax.set_autoscaley_on(True)
        self.ax.set_xlim(min_x, max_x)
        self.ax.set_ylim(min_y, max_y * 1.15)

        # getting the coordinates of the trains
        x_data, y_data = self.get_x_y(epsilon=0.02 * (max_y - min_y), extension_x_of_connections=extension_x_of_connections)

        # converting the x and y of the locomotive to trains of length 3
        x_trains = []
        y_trains = []

        for x, y in zip(x_data, y_data):
            for length in range(train_length):
                x_trains.append(x - length)
                y_trains.append(y)
        x_trains = np.array(x_trains)
        y_trains = np.array(y_trains)

        # adding the text, the number of passangers, the timestamp, number of conflicts
        epsilon = 0.05
        for train_id in range(self.nr_agents):
            self.ax.text(x_data[train_id] - 1, y_data[train_id] + epsilon, str(int(sizes_lst[train_id])), fontfamily='sans-serif')

        self.ax.text((max_x - min_x) * 0.5 - 5, 1.15 * max_y, 'time step: ' + str(int(self.timestamp)))
        self.ax.text((max_x - min_x) * 0.5 - 5, 1.1 * max_y, 'conflicts: ' + str(int(nr_conflicts)), color=color)
        self.ax.text((max_x - min_x) * 0.5 - 5, 1.05 * max_y, 'arrived at dest.: ' + str(int(np.sum(self.terminated_lst))), color='blue')

        # drawing the trains (the body, and the loco)
        self.ax.scatter(x_trains - 1, y_trains, color=color, marker='s', alpha=0.6)
        self.ax.scatter(x_data, y_data, color='black', marker='>')

        # drawing the destinations
        self.ax.scatter(destinations_x, destinations_y, color='blue', marker='s')

        plt.draw()
        plt.show()
        plt.pause(0.1)

        if dir_name:

            if not os.path.exists(dir_name):
                os.makedirs(dir_name)

            plt.savefig(
                os.path.join(dir_name, 'fig_' + str(self.timestamp) + '.png'))
