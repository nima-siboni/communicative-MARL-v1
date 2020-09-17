# Multi-agent
import numpy as np
from tensorflow.keras.models import load_model
from environment import Environment
from agent import Agent
from rl_utils import extract_state_and_comm_from_global_state
from utilfunctions import initializer
from utilfunctions import single_shape_adaptor
from random import randint
# The saved model to be read (take the model at the end of the training)
timestep = 580

# if true the trains just move forward
just_forward = False

# if true only random steps are taken
random_steps = True

# make sure that it is either moving forward or taking random steps
assert not random_steps or not just_forward


# This part should be identical to the learn_and_experience.py file
# in a better world I would have wrote this info on the disk (in learn_and_experience) and
# read it from the disk here.

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
conflict_cost = 70 
nr_actions = 2

env = Environment(ls1, lc, ls2, nr_agents, states, time_cost, destinations, conflict_cost, nr_actions)


# 0.2.0 -- creating the agents

agent_lst = []
learning_rates = [0.000005, 0.0001]
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



for agent_id in range(nr_agents):
    agent_lst[agent_id].Q_t = load_model('./training-results/Q-target/trained-agents/trained-agent-timestep-' + str(timestep) + '-' + str(agent_id))
    agent_lst[agent_id].update_Q_to_Q_t()




# lets simulate

global_state_space_size = len(env.states)
nr_features_per_agent = agent_lst[0].nr_features
nr_comm_features_per_agent = agent_lst[0].nr_features
# + agent_lst[0].nr_actions

initial_state_global = env.reset()

for agent_id in range(len(agent_lst)):
    agent_lst[agent_id].terminated = False

state_global, terminated, steps = initializer(initial_state_global)
state_global = single_shape_adaptor(state_global, global_state_space_size)

while not terminated:

    action_id_lst = []
    # choosing actions for each agent
    for agent_id in range(env.nr_agents):
        agent = agent_lst[agent_id]
        state, comm = extract_state_and_comm_from_global_state(state_global, agent_id)
        action_id = agent.action_based_on_Q_target(state, comm, env, epsilon=10)
        if just_forward:
            action_id = 1
        if random_steps:
            action_id = randint(0, 1)
        action_id_lst.append(action_id)

    # STEP : taking a step with all the agent
    new_state_global, reward_lst, terminated_lst, info = env.step(action_id_lst)
    new_state_global = single_shape_adaptor(new_state_global, global_state_space_size)
    if just_forward:
        env.render('./performance-and-animations/animations-as-fast-as-possible')

    if random_steps:
        env.render('./performance-and-animations/animations-random-walk')

    if random_steps == False and just_forward == False:
        env.render('./performance-and-animations/animations')

    terminated = np.array(terminated_lst).all()
    state_global = new_state_global
    steps = steps + 1

print("...    the terminal_state is reached after " + str(steps))
