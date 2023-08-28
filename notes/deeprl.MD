# Deep Reinforcement Learning

My notes on Deep Reinforcement Learning are @/[deeprl_theory](./drl.MD) and here I roughly go over a typical DRL NN architecture:


<img src="img/dd.png" width=100%> 

Deep Reinforcement Learning (DRL) is a subfield of artificial intelligence that focuses on training agents to make sequential decisions in complex environments using deep neural networks. DRL algorithms can be categorized into various types and classes based on their underlying principles and methodologies. Here are detailed notes on these types and classes:

### +  Value-Based Methods:

+ Q-Learning: Q-learning is one of the earliest reinforcement learning algorithms. In Deep Q-Learning (DQN), deep neural networks are used to approximate the Q-values of state-action pairs. It involves an experience replay buffer to stabilize learning and target networks to improve convergence.  ([qlearning.py](./code/qlearning.py))

+ Double Deep Q-Networks (DDQN): DDQN improves upon DQN by mitigating overestimation bias in Q-values, enhancing stability and learning.  ([ddqn.py](./code/ddqn.py))

+ Prioritized Experience Replay: This technique assigns different priorities to experiences in the replay buffer, allowing the agent to focus on learning from more informative experiences.



### +  Policy-Based Methods:

+ <b>Policy Gradients:</b> Policy gradient methods directly optimize the policy function by maximizing expected cumulative rewards. Algorithms like REINFORCE and Trust Region Policy Optimization (TRPO) fall under this category.

+ <b>Proximal Policy Optimization (PPO):</b> PPO is a popular policy gradient algorithm that addresses some of the limitations of TRPO. It uses a clipped surrogate objective to ensure stable updates.



### +  Actor-Critic Methods:

+  <b>Advantage Actor-Critic (A2C):</b> A2C combines the benefits of both policy gradients and value-based methods by using an actor (policy) network and a critic (value) network. It reduces variance compared to vanilla policy gradients.

+  <b>A3C (Asynchronous Advantage Actor-Critic):</b> A3C employs parallelism to speed up training. Multiple agents explore different trajectories simultaneously, improving sample efficiency.

+ <b>DDPG (Deep Deterministic Policy Gradient):</b> DDPG extends actor-critic methods to continuous action spaces. It uses a deterministic policy and a Q-network to estimate action values.



### +  Model-Based Methods:

+ <b>Monte Carlo Tree Search (MCTS):</b> MCTS is a model-based algorithm used for decision-making in games and planning. It builds a search tree by simulating possible actions and outcomes.

+ <b>Model-Predictive Control (MPC):</b> MPC uses a learned model of the environment to optimize a sequence of actions. It is commonly used in robotics and control tasks.



### +  Exploration Strategies:

+ <b>Epsilon-Greedy:</b> In epsilon-greedy strategies, the agent selects the best-known action with probability (1 - ε) and explores randomly with probability ε.

+ <b>Softmax Exploration:</b> Softmax exploration assigns probabilities to each action based on their estimated values, allowing for a more controlled form of exploration.



### +  Deep Reinforcement Learning for Continuous Control:
+ <b>Deep Deterministic Policy Gradient (DDPG):</b> DDPG is a popular algorithm for continuous action spaces, combining Q-learning and policy gradients.

+ <b>Trust Region Policy Optimization (TRPO):</b> TRPO is a policy optimization method that ensures stable updates by limiting policy changes based on a trust region constraint.

+ <b>SAC (Soft Actor-Critic):</b> SAC is an off-policy algorithm designed for stochastic policies. It incorporates entropy regularization to encourage exploration.



### +  Multi-Agent Reinforcement Learning:

 + <b>MARL (Multi-Agent Reinforcement Learning):</b> MARL deals with scenarios where multiple agents interact in a shared environment. Algorithms like MADDPG and multi-agent PPO are used for such tasks.


### +  Deep Reinforcement Learning for Robotics:

+ <b>Robotic Control:</b> DRL has found applications in robotic control, including tasks like robotic manipulation, locomotion, and navigation.

+ <b>Sim2Real Transfer:</b> Research focuses on transferring DRL policies trained in simulation to real-world robotic platforms.

## Model-based algorithms: 

Model-based algorithm use the transition and reward function to estimate the optimal policy. 
+ They are used in scenarios where we have complete knowledge of the environment and how it reacts to different actions.
+ In Model-based Reinforcement Learning the agent has access to the model of the environment i.e., action required to be performed to go from one state to another, probabilities attached, and corresponding rewards attached.
+ They allow the reinforcement learning agent to plan ahead by thinking ahead.
+ For static/fixed environments,Model-based Reinforcement Learning is more suitable.


## Model-free algorithms
Model-free algorithms find the optimal policy with very limited knowledge of the dynamics of the environment. They do no thave any transition/reward function to judge the best policy.

+ They estimate the optimal policy directly from experience i.e., interaction between agent and environment without having any hint of the reward function.
+ Model-free Reinforcement Learning should be applied in scenarios involving incomplete information of the environment.
+ In real-world, we don't have a fixed environment. Self-driving cars have a dynamic environment with changing traffic conditions, route diversions etc. In such scenarios, Model-free algorithms outperform other techniques

Markov Decision Process (MDP) : [Markov Decision Processes](https://youtu.be/2iF9PRriA7w)

### Bellman Equations : 

+ State is a numerical representation of what an agent observes at a particular point in an environment.

+ Action is the input the agent is giving to the environment based on a policy.

+ Reward is a feedback signal from the environment to the reinforcement learning agent reflecting how the agent has performed in achieving the goal.

Bellman Equations aim to answer these questions: 
The agent is currently in a given state ‘s’. Assuming that we take best possible actions in all subsequent timestamps,what long-term reward the agent can expect?

or What is the value of the state the agent is currently in? Bellman Equations are a class of Reinforcement Learning algorithms that are used particularly for deterministic environments.

$$ V(s) = max_a ( R ( s , a ) + γ V ( s ´ ))  $$


Dynamic Programming :  There are two classes of Dynamic Programming:

1. Value Iteration : The optimal policy (optimal action for a given state) is obtained by choosing the action that maximizes optimal state-value function for the given state.

2. Policy Iteration : This algorithm has two phases in its working:

   Policy Evaluation—It computes the values for the states in the environment using the policy provided by the policy improvement phase.

   Policy Improvement—Looking into the state values provided by the policy evaluation part, it improves the policy so that it can get higher state values.


```python

import torch 
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    """ Actor (Policy) Model."""
    def __init__(self, state_size,action_size, seed, fc1_unit=64,
                 fc2_unit = 64):
        """
        Initialize parameters and build model.
        Params
        =======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_unit (int): Number of nodes in first hidden layer
            fc2_unit (int): Number of nodes in second hidden layer
        """
        super(QNetwork,self).__init__() ## calls __init__ method of nn.Module class
        self.seed = torch.manmual_seed(seed)
        self.fc1= nn.Linear(state_size,fc1_unit)
        seed.fc2 = nn.Linear(fc1_unit,fc2_unit)
        seed.fc3 = nn.Linear(fc2_unit,action_size)
        
    def forward(self,x):
        # x = state
        """
        Build a network that maps state -> action values.
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

```

### Multi-Agent Reinforcement Learning (MARL):


MARL deals with scenarios where multiple agents interact within a shared environment, making decisions and learning simultaneously. These agents may have distinct goals, and their actions can affect each other.
+ Challenges in MARL:

    - <b>Credit Assignment:</b> Determining which agent's actions contributed to the overall reward can be challenging, especially in competitive or cooperative settings.
    - <b>Non-Stationarity:</b> Agents' policies change over time as they learn, leading to non-stationarity in the environment.
    - <b>Curse of Dimensionality:</b> The state and action spaces grow exponentially with the number of agents, making it computationally expensive.
    - <b>Exploration vs. Exploitation:</b> Balancing exploration and exploitation is complex when multiple agents are involved.
+ Types of MARL Algorithms:

    - <b>Independent Learners:</b> Each agent learns independently, treating others as part of the environment. Simple but may lack coordination.
    - <b>Joint Action Learners:</b> Agents coordinate by sharing information or learning a joint policy.
    - <b>Centralized Training with Decentralized Execution (CTDE):</b> Agents train with centralized information but execute actions independently. Enhances coordination.

### Imitation Learning : 

Imitation Learning, also known as Learning from Demonstration (LfD), involves learning a policy by mimicking expert demonstrations rather than trial and error.

#### Key Components:

- <b>Expert Demonstrations:</b> Imitation learning relies on a dataset of demonstrations provided by an expert.
- <b>Imitation Algorithm:</b> Algorithms like Behavioral Cloning and Inverse Reinforcement Learning (IRL) are used to learn a policy that mimics the expert's behavior.
- <b>Policy Evaluation:</b> Imitation learning may require evaluating the learned policy's performance through reinforcement learning or other methods.

### Adversarial Reinforcement Learning (ARL): 

In ARL, RL agents are trained to perform well not only in standard environments but also in the presence of adversarial disturbances. Adversaries can introduce perturbations or modify the environment dynamics to challenge the RL agent.

Exploration and Generalization: Adversarial training can be used to generate diverse and challenging scenarios for RL agents during training, leading to better exploration and improved generalization.

Adversarial Reinforcement Learning (ARL) is an emerging field at the intersection of Reinforcement Learning (RL) and Adversarial Training. It addresses a critical challenge in RL: making RL agents robust to uncertainties, adversarial environments, and unexpected disturbances. ARL leverages adversarial training techniques to enhance the stability and safety of RL agents, making them better suited for real-world applications.

#### Key Components of ARL:

+ <b>Reinforcement Learning (RL)</b>:

    - <b>Agent-Environment Interaction:</b> In RL, an agent interacts with an environment and learns to make a sequence of decisions (actions) to maximize cumulative rewards. The agent's goal is to find an optimal policy that maximizes its expected return.
+ <b>Adversarial Training</b>:

    - <b>Robustness Against Adversarial Inputs:</b> Adversarial training is primarily used to enhance the robustness of machine learning models, particularly against adversarial inputs that are specifically designed to mislead the model.
    - <b>Generation of Adversarial Examples:</b> Adversarial examples are data points that have been subtly perturbed to cause misclassification or erroneous behavior by the model.
    - <b>Adversarial Training Procedure:</b> During adversarial training, the model is exposed to these adversarial examples in addition to the regular training data. The model learns to be more resilient to these perturbations.


### Adversarial Imitation Learning (AIL) :

Adversarial Imitation Learning (AIL) is an innovative approach at the intersection of Imitation Learning and Adversarial Training. It addresses the challenge of learning effective policies by leveraging expert demonstrations and adversarial training techniques. AIL aims to create more robust and generalizable policies by combining the strengths of both paradigms.

Key Component of AIL: Mimicking Expert Behavior: 

Imitation Learning, also known as Learning from Demonstration (LfD), involves learning a policy by mimicking expert demonstrations. The goal is to replicate the behavior exhibited by an expert demonstrator.


resources: [An Introduction to Deep Reinforcement Learning](https://huggingface.co/blog/deep-rl-intro), [A Beginner's Guide to Deep Reinforcement Learning](https://wiki.pathmind.com/deep-reinforcement-learning), [Key concepts in RL - OpenAI](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html), [Key papers in RL - OpenAI](https://spinningup.openai.com/en/latest/spinningup/keypapers.html), @github/[Deep-Reinforcement-Learning-Algorithms-with-PyTorch](https://github.com/p-christ/Deep-Reinforcement-Learning-Algorithms-with-PyTorch), @github/[cleanrl](https://github.com/vwxyzjn/cleanrl), [Implementing Deep Reinforcement Learning Models with Tensorflow + OpenAI Gym](https://lilianweng.github.io/posts/2018-05-05-drl-implementation/), paperswithcode/[rl](https://paperswithcode.com/methods/area/reinforcement-learning), [Python Reinforcement Learning using Gymnasium](https://youtu.be/vufTSJbzKGU), [Reinforcement Learning Course: Intro to Advanced Actor Critic Methods](https://youtu.be/K2qjAixgLqk), [@MachineLearningwithPhil](https://www.youtube.com/@MachineLearningwithPhil/playlists), [Deep Reinforcement Learning in Python Tutorial - A Course on How to Implement Deep Learning Papers](https://youtu.be/GJJc1t0rtSU).