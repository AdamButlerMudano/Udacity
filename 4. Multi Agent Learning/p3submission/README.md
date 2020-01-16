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

The relevant Unity environment and code required to run the simulation can be found on the github repository https://github.com/AdamButlerMudano/Udacity. The simulation can be executed by running the Tennis.ipynb notebook within the folder '4. Multi Agent Learning'.

The following packages are required to run the solution, these can all be installed from pip:
- unityagents
- numpy
- torch
- matplotlib



### Running Simulation

To run the agent training execute  all cells in notebook: **tennis.ipynb**.