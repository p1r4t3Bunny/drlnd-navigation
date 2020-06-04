# About this project

This projects contains my solution of the first project in the **[Udacity Deep Reinforcement Learning Nanodegree](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893)** of [Udacity](https://www.udacity.com/).
The goal of this project is to train an agent to navigate (and collect bananas!) in a large, square world. 

# The Environment
A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana. Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around the agent's forward direction. Given this information, the agent has to learn how to best select actions. Four discrete actions are available, corresponding to:

* 0 - move forward.
* 1 - move backward.
* 2 - turn left.
* 3 - turn right.


![Trained Agent](./img/banana.gif)

The environment provided by Udacity is similar to, but not identical to the [Banana Collector environment on the Unity ML-Agents GitHub page](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#banana-collector). To run the code in this project, the specified environment of udacity is needed. To set it up, follow the instructions below.

## Step 1 - Getting started
Install PyTorch, the ML-Agents toolkit, and a few more Python packages according to the [instructions in the DRLND GitHub repository](https://github.com/udacity/deep-reinforcement-learning#dependencies).

## Step 2 - Download the Unity Environment
For this project, you **don't** need to install Unity. Instead, choose the pre-built environmen provided by Udacity matching your operating system:


* Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
* Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
* Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
* Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)

# Instructions

To explore the environment and train the agent, start a jupyter notebook, open TrainNavigationAgent.ipynb and execute the steps. For more information, and an exmaple on how to use the agent, please check instructions inside the notebook.

## Project structure

* `TrainNavigationAgent.ipynb`: The jupyter notebook for executing the training
* `dqn_agent.py` : contains the implementation of the Agent
* `model.py` : contains the PyTorch models of the neural network used by the Agent
* `replay_buffer.py` : The replay buffer implmentation for memory.
* `defaults.py` : Contains the default hyperparameters of the models.


# Results

The trained agent solved the environment in 247 episodes.
For a detailed explanation, please read the [project report](./Report.md)

# Notes
The project uses the code and task description provided in the **[Udacity Deep Reinforcement Learning Nanodegree](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893)**  class as a basis.
