# communicative-MARL-v1
A multi-agent RL where the agents learn *what* to communicate with each other.

The assumptions:
fully observable environment
it is a markov process
The environment is deterministic

The aim:
to train agents which through communication with the others, collectively:

* avoid **conflicts** and simultaneously
* minimize the total **priority weighted delay**

Here, we aim at solving this problem starting from any state of the system (regardless of all the previously happened delays).


## The MARL approach

The problem to solve is to solve the
* A frequently occuring bottleneck in multi-agent system is the *amount* of the data they can communicate on the timescale which is relevant for the decision making. This is a problem both in cases where the agents are separated in real world or in-silico across different computational nodes (of a distributed system). facilite the amount of the communicated data, where the agents learn what sort of information they should communicate.

* Here we aimed at compeletly decentralized system, with no *super-agent* which decides for on behalve the agents during the execution time . 

Here we have assumed that the communication between the agents are cheap in the simulated environment, but we would like to avoid that as much as possible in the execution phase.

## The environment

The used environment is the [multi-agent-trains-env](https://github.com/nima-siboni/multi-agent-trains-env) which is developed specifically to a RL-friendly simulation environment.

Here, the major modification introduced to this environment concerns the reward engineering. In the original implementation both of the agents recieved a large negative reward as soon as a conflict happened. With a slight modificition, here only the agent which enters into a currently occupied track reccieves the negative reward. 

## Future steps

* Here agents at optimizing the objectives *solely* based on their current state. This means that decisions are made regardless of delays occured prior to the current time, and also **no forcast of future delays**. An interesting extension of the current approach would be to add the delay predict ability to agents. The predicted future delays can be used together with the current state for making better decisions. This can be approached using common AI/non-AI approaches for forcasting sequences of events. 

This makes sense for the train system, as if a technical problem leads to a lets say blockage of a segment, the agent
