# Problem Domain

### Environment

The environment consists of an open 2 dimensional space sparsely filled with 'Bananas' with which the agent must learn to navigate.

### Reward and Goal

Rewards of +1 will be granted when the agent navigates to a yellow banana but a reward of -1 will be granted when the agent navigates to a blue banana.

The problem will be considered complete when an average score (over previous 100 episodes) of +13 is achieved.

### State and Action Spaces

There are 4 actions available to the agent corresponding to turn left and right plus move forwards and backwards.

The state space has 37 dimensions that correspond to simulate sensor inputs on our rover agent.


# Getting Started

### Installation

The following packages are required to run the solution, these can all be installed from pip:
- unityagents
- numpy
- torch

The Unity environment used for the simulation can be installed by followed the instructions at Step 2 [here](https://classroom.udacity.com/nanodegrees/nd893/parts/6b0c03a7-6667-4fcf-a9ed-dd41a2f76485/modules/4eeb16ab-5ac5-47bf-974d-12784e9730d7/lessons/69bd42c6-b70e-4866-9764-9bfa8c03cdea/concepts/319dc918-bd2c-4d3b-80a5-063bb5f1905a).

### Running Simulation

To run the agent training execute  all cells in notebook: **navigation.ipynb**.