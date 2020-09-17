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

One route to MARL is to compose a global agent which determines the actions of all the agents based on the global state of the system. Such a fully centeralized approach is simple to implement and the best affordable approach considering the stability and convergence. Essentially such an approach converts the multi-agent problem to a single agent problem and all the techniques developed for single RL can be used here. One can benefit from convergence (to optimal and sub-optimal solutions) and stability of the single agent algorithms. An example of such a approach is implemented here in [narrow-corridor-ai](https://github.com/nima-siboni/narrow-corridor-ai), where the globally optimal solution is obtained using a tabular value based method.


An example of 

Although the aforementioned approach can be deployed to solve many multi-agent RL problems (for an example , the 


There are a number of draw
to make the convergence and stability of theapproaches where the single agent is much more complicated, i.e. the state 

A common challenge to MARL is thatThe approach taken here can be considered as a semi-independent execution with semi-centeralized learning, as explained in the followings. The practical benefits of this approach is highlighted at the end of this section. 

## Learning

During the learning process, each agent learns independently from the rewards it gets for its actions knowing the state of all other agents. In other words, during the learning the states of other agents are presented to each agent. This is, generally speaking, problematic if the full state of all the agents are passed around:

* The DNN of each agent grows significantly 

as the DNNs for each agent grows rapidly 

are, the information of each agent is 

## Execution 

During the execution the i-th agent makes its decision based on its own state plus some information which is communicated from other agents. In other words, similar to the ..., we establish a comminucation channel between agents. This is similar to ... wh, we do not pass the whole state (observation) of the other agents to this agent, but rather only a small volume of processed/condensed information is passed to it. What is passed to from each agent to the other agents is learned during the training.

## Learning
* 
The problem to solve is to solve the
* A frequently occuring bottleneck in multi-agent system is the *amount* of the data they can communicate on the timescale which is relevant for the decision making. This is a problem both in cases where the agents are separated in real world or in-silico across different computational nodes (of a distributed system). facilite the amount of the communicated data, where the agents learn what sort of information they should communicate.

* Here we aimed at compeletly decentralized system, with no *super-agent* which decides for on behalve the agents during the execution time . 

Here we have assumed that the communication between the agents are cheap in the simulated environment, but we would like to avoid that as much as possible in the execution phase.

here we relax the assumption that during the execution the agent only has access to its own state which is assumed in https://arxiv.org/pdf/1706.02275.pdf
## The environment

The used environment is the [multi-agent-trains-env](https://github.com/nima-siboni/multi-agent-trains-env) which is developed specifically to a RL-friendly simulation environment.

Here, the major modification introduced to this environment concerns the reward engineering. In the original implementation both of the agents recieved a large negative reward as soon as a conflict happened. With a slight modificition, here only the agent which enters into a currently occupied track reccieves the negative reward. 

## Future steps

* Here agents at optimizing the objectives *solely* based on their current state. This means that decisions are made regardless of delays occured prior to the current time, and also **no forcast of future delays**. An interesting extension of the current approach would be to add the delay predict ability to agents. The predicted future delays can be used together with the current state for making better decisions. This can be approached using common AI/non-AI approaches for forcasting sequences of events. 

This makes sense for the train system, as if a technical problem leads to a lets say blockage of a segment, the agent
