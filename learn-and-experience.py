import numpy as np
from tensorflow.keras.models import load_model
from environment import Environment
from agent import Agent
from rl_utils import Histories
from rl_utils import update_replay_buffer_with_episodes
from rl_utils import testing_performance
from rl_utils import logging_performance
from utilfunctions import get_replay_buffer_size

# -------------------------------- #
# 0 -- initializations
# -------------------------------- #
# 0.0 -- create the environment
# 0.1 -- creating the buffers
# 0.2 -- creating the agents
# 0.2.0 -- creating the list of agents
# 0.2.1 -- saving the Q-target of the untrained agents
# 0.2.2 -- equating the Q-target and Q
#
# -------------------------------- #
# 1 -- filling the replay buffers
# -------------------------------- #

# 1.0 -- d

rounds_data_exploration = 50
continue_an_old_run = False
# 0.0 -- create the environment

ls1 = 3
lc = 3
ls2 = 3
nr_agents = 2
nr_features_per_agent = 2
states = [0., 1., 0., 10.]
assert (len(states) == nr_features_per_agent * nr_agents)
time_cost = 1
destinations = [8., 8.]
assert (len(destinations) == nr_agents)
conflict_cost = 5 #20
nr_actions = 2

env = Environment(ls1, lc, ls2, nr_agents, states, time_cost, destinations, conflict_cost, nr_actions)

# 0.1 -- creating replay buffers for each agent

max_replay_buffer_size = 50_000

replay_buffer_lst = []

for agent_id in range(nr_agents):
    replay_buffer_lst.append(
        Histories(max_replay_buffer_size)
    )


# 0.2.0 -- creating the agents

agent_lst = []
learning_rates = [0.001, 0.0001]
for agent_id in range(nr_agents):
    agent_lst.append(
        Agent(nr_features=nr_features_per_agent,
              nr_actions=env.nr_actions,
              nr_comm_condensed=1,
              gamma=0.98,
              stddev=0.1,
              learning_rate=learning_rates[agent_id],
              agent_id=agent_id))

assert env.nr_actions == agent_lst[0].nr_actions

# 0.2.1 loading the Q and Q-target from an old run
if continue_an_old_run:
    for agent_id in range(nr_agents):
        agent_lst[agent_id].Q_t = load_model('./training-results/Q-target/trained-agents/trained-agent-' + str(agent_id))
        agent_lst[agent_id].update_Q_to_Q_t()

# 0.2.2 -- saving the untrained agents

if continue_an_old_run == False:
    for agent_id in range(nr_agents):
        agent_lst[agent_id].Q_t.save('./training-results/Q-target/not-trained-agent-' + str(agent_id))


replay_buffer_lst = update_replay_buffer_with_episodes(rounds_data_exploration, agent_lst, replay_buffer_lst, env, epsilon=5.0)

# number of updates for Q-target
U = 600
# number of episodes added to the replay buffer per update of Q-target
N = 5
# number of updates epochs of learning for Q per added episode
K = 1

training_log = np.array([])

nr_steps_test = 2

counter = 0

for u in range(U):

    print("update Q-t round :" + str(u))

    if continue_an_old_run:
        epsilon = max(0.05, 0.5 - (0.5 - 0.05) * (u / U))
    else:
        epsilon = max(0.05, 0.5 - (0.5 - 0.05) * (u / U))

    print("...    epsilon in exploring is :" + str(epsilon * 100) + "%,  buffer_size is " + str(get_replay_buffer_size(replay_buffer_lst)))

    for n in range(N):
        print("...    update the replay_buffer :" + str(n))
        replay_buffer_lst = update_replay_buffer_with_episodes(1, agent_lst, replay_buffer_lst, env, epsilon)

        print("...          updating the Q started")

        for k in range(K):
            # import pdb; pdb.set_trace()
            for agent_id in range(nr_agents):
                current_batch = replay_buffer_lst[agent_id].return_a_batch(batchsize=32)
                agent_lst[agent_id].learn(current_batch, env)

        print("...          updating the Q finished")

    print("...          the Q-target update is started.")

    for agent_id in range(nr_agents):
        agent_lst[agent_id].update_Q_t_to_Q()

    print("...          the Q-target is updated.")

    for agent_id in range(nr_agents):
        if (counter % 20) == 0:
            if continue_an_old_run:
                agent_lst[agent_id].Q_t.save('./training-results/Q-target/trained-agents/last-trained-agent-continued-' + str(agent_id))
            else:
                agent_lst[agent_id].Q_t.save('./training-results/Q-target/trained-agents/trained-agent-timestep-' + str(counter) + '-' + str(agent_id))

    print("...          the Q-target network is saved on the disk")

    # testing
    average_performance = testing_performance(nr_steps_test, agent_lst, env, epsilon=0.)

    print("...          the average performance is " + str(average_performance))
    print("...    the test is over")
    training_log = logging_performance(training_log, counter, average_performance, write_to_disk=True)
    counter += 1
