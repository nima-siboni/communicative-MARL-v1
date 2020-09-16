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

## The environment

The used environment is the [multi-agent-trains-env](https://github.com/nima-siboni/multi-agent-trains-env) which is developed specifically to a RL-friendly simulation environment.

Here, the major modification introduced to this environment concerns the reward engineering. In the original implementation both of the agents recieved a large negative reward as soon as a conflict happened. With a slight modificition, here only the agent which enters into a currently occupied track reccieves the negative reward. 
