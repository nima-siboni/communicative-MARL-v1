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

One route to MARL is to compose a global agent which determines the actions of all the agents based on the global state of the system. Such a fully centeralized approach is simple to implement and the best affordable approach considering the stability and convergence. Essentially such an approach converts the multi-agent problem to a single agent problem and all the techniques developed for single RL can be used, and one can benefit from the convergence (to optimal and sub-optimal solutions) and stability of the single agent algorithms. An example of such an approach is implemented here in [narrow-corridor-ai](https://github.com/nima-siboni/narrow-corridor-ai), where the globally optimal solution is obtained using a tabular value based method.

Although such a centeralized can be used to solve many multi-agent RL problems, in practice this is not always feasible, as explained here: 

* One common challenge is that the super-agent becomes large and unfeasible to train as the number of agents grow (yet another example of curse-of-dimensionality!). 

* Another challenge for this approach occures during the execution phase, namely such a super-agent requires all the information of all the agents to make a decision. This means a large volume of data exchange with the environment for each decision making incident, which might make the usage of RL impossible in settings where the required infrastructure does not exist or the data communication is slower than the timescale relevant for the decision making process. This is a problem both in cases where the agents are separated in real world or in-silico across different computational nodes (of a distributed system). 

These challenges encourage devising new multi-agent algorithms where the learning and the execution phases are less centeralized. A successful example of such methods is the centralized learning decenteralized execution approach proposed in Refs. [[1](https://arxiv.org/pdf/1706.02275.pdf)-[2](https://arxiv.org/pdf/1605.06676.pdf)]. The decenteralized execution, here, means that each agent has only access to its own state (which solve the communication challenge mentioned above). The centeralized earning means that during this phase the agents have full access to each others states (and actions). This is an approch which has proven to be successful to tackle mentioned challenges in many practical instances. The success of this method requires a framework where the agent can decide decide on information which is different than the iformation it had during the training. In the aforementioned Refs., this is achieved via using actor-critic method, where only the critic requires the global information. This is rather a restrictive property of this method that one can only use Actor-Critic, and not other methods like DDQN, which could be more appropriate for the problem at hand. In the following, we turn to a method which does not have this restriction, and also curbs the challenges mentioned above.


The approach we present and implement here can be considered as a semi-independent execution with semi-centeralized learning, as explained in the followings.


## Learning

During the learning process, each agent learns independently from the rewards it gets for its actions knowing the state of all other agents. An essential element here is that during the learning phase, the states of all other agents are presented to each agent. This, if done naively, could lead to the curse of dimensionality problem, as explained above. 

This is essentially similar to the approach taken in Refs. [[1](https://arxiv.org/pdf/1706.02275.pdf)-[2](https://arxiv.org/pdf/1605.06676.pdf)]. In these references, the information of the other agents are used only during training, i.e. for the Actor in 

For the 
* The DNN of each agent grows significantly 

as the DNNs for each agent grows rapidly 

are, the information of each agent is 

## Execution 

During the execution the i-th agent makes its decision based on its own state plus some information which is communicated from other agents. In other words, similar to the ..., we establish a comminucation channel between agents. This is similar to ... wh, we do not pass the whole state (observation) of the other agents to this agent, but rather only a small volume of processed/condensed information is passed to it. What is passed to from each agent to the other agents is learned during the training.


.  In other words, the *amount* of the data they can communicate on the timescale which is relevant for the decision making.
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
