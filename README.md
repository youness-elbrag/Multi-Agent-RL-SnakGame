# Snake Game RL Agent with Deep Q-Learning

## Table of Contents
- [Introduction](#introduction)
- [Deep Q-Learning Algorithm](#deep-q-learning-algorithm)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)

## Introduction
This project aims to develop a reinforcement learning (RL) agent that can learn to play the classic Snake game using the Deep Q-Learning algorithm. The Snake game is a simple yet challenging environment where the agent needs to learn how to navigate and eat food to grow while avoiding collisions with its own tail and the game boundaries. Deep Q-Learning is a popular RL technique that combines Q-Learning with deep neural networks to train agents to make optimal decisions in complex environments.

## Deep Q-Learning Algorithm

Deep Q-Learning (DQL) is a reinforcement learning algorithm that extends the classic Q-Learning approach by using deep neural networks to approximate the Q-function. In DQL, the Q-function, denoted as $\(Q(s, a)\)$, represents the expected cumulative future rewards when taking action \$(a\)$ in state $\(s\)$.

The DQL algorithm involves the following key components:

1. **Q-Network**: A deep neural network that approximates the Q-function. The Q-network takes the state $\(s\)$ as input and outputs Q-values for all possible actions. The Q-values for each action are represented as $\(Q(s, a; \theta)\)$, where $\(\theta\)$ are the network's weights.

   The Q-value is updated using the Bellman equation:
   
   $\[Q(s, a; \theta)$ = $\mathbb{E}_{s'}[r + \gamma \max_{a'} Q(s', a'; \theta^-)|s, a]\]$
   
   where:
   - \$(s'\)$ is the next state.
   - $\(r\)$ is the reward obtained.
   - $\(\gamma\)$ is the discount factor.
   - $\(\theta^-\)$ represents the target Q-network.

2. **Experience Replay**: To improve stability and break correlations in the data, DQL uses experience replay. Experiences $\((s, a, r, s')\)$ are stored in a replay buffer, and mini-batches are randomly sampled for training.

3. **Target Network**: DQL employs a target network with parameters \(\theta^-\) to estimate target Q-values. The target network is a separate network periodically updated with the Q-network's parameters to stabilize training. The target network helps reduce the risk of divergence during training.

4. **Exploration Strategies**: DQL often uses an exploration strategy to balance exploration and exploitation. Two common strategies are:

   - **Epsilon-Greedy Policy**: With probability $\(\epsilon\)$, the agent selects a random action, and with probability $\(1 - \epsilon\)$, it selects the action with the highest Q-value.

   - **Boltzmann Exploration (Softmax Policy)**: Instead of a fixed \$(\epsilon\)$, the Boltzmann exploration strategy uses a temperature parameter $\(\tau\)$. Actions are sampled probabilistically according to the Boltzmann distribution, where the probability of selecting action $\(a\)$ is defined as:
   
     $[P(a|s) = \frac{e^{Q(s, a; \theta) / \tau}}{\sum_{a'} e^{Q(s, a'; \theta) / \tau}}\]$
   
     where $\(\tau\)$ controls the exploration intensity. Higher \(\tau\) values lead to more exploratory behavior, while lower values make the agent more deterministic.

5. **Q-Learning Update Rule**: The Q-values are updated using the Bellman equation, and the network parameters $\(\theta\)$ are adjusted to minimize the temporal difference (TD) error.

DQL aims to find the optimal policy by iteratively improving the Q-function estimates. Over time, the Q-network learns to make better decisions in the environment, ultimately leading to improved performance in the task.

For more details and advanced variations of DQL, refer to the RL literature and relevant research papers.


### Project Structure

snake_env.py: Snake game environment.
dqn.py: Implementation of the Deep Q-Learning algorithm.
train.py: Training script for the RL agent.
play.py: Script to play the Snake game using the trained agent.
model.pth: Saved model weights after training.
README.md: This documentation.

### Installation

```bash
git clone https://github.com/deep-matter/Multi-agnet-RL.git
```

1. Install the required dependencies:

```bash

    pip install -r requirements.txt
```





