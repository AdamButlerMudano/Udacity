# Problem Domain

### Environment

The environment consists of a tennis court on which 2 agents must play a ball over the net as many times as possible without playing the ball out of bounds and without the ball hitting the net or ground. The episosde will terminate if either of these occur.


### Reward and Goal

Rewards of +0.1 will be granted to an agent when it manages to successfully play a the ball over the net. If the ball is played out of bounds or hits the net or ground a reward of -0.01 will be given to that agent.

The problem will be considered complete when the maximum score between the 2 agents averages above +0.5 for the previous 100 episodes.


### State and Action Spaces

Each agent can make actions of size 2, corresponding to up/down and left/right.

The state space has 24 dimensions, with each agent making its own observations at each step. 


# Getting Started

### Installation

The following packages are required to run the solution, these can all be installed from pip:
- unityagents
- numpy
- torch
- matplotlib

The Unity environment used for the simulation can be installed by followed the instructions at Step 2 [here](https://classroom.udacity.com/nanodegrees/nd893/parts/ec710e48-f1c5-4f1c-82de-39955d168eaa/modules/89b85bd0-0add-4548-bce9-3747eb099e60/lessons/3cf5c0c4-e837-4fe6-8071-489dcdb3ab3e/concepts/e85db55c-5f55-4f54-9b2b-d523569d9276).

### Running Simulation

To run the agent training execute  all cells in notebook: **tennis.ipynb**.