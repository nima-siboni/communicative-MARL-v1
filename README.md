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

The used environment is the one developed [here](https://github.com/nima-siboni/multi-agent-trains-env).
