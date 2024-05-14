# Deep Reinforcement Learning

My personal notes on Deep Reinforcement Learning are @/[deeprl_theory](./drl.MD) and here we discuss a typical DRL NN architecture:

<img src="img/dd.png" width=100%> 

Deep Reinforcement Learning (DRL) is a subfield of artificial intelligence that focuses on training agents to make sequential decisions in complex environments using deep neural networks. DRL algorithms can be categorized into various types and classes based on their underlying principles and methodologies. Here are detailed notes on these types and classes:

+ `Markov Decision Processes (MDPs)`: The mathematical framework for modeling sequential decision-making problems, comprising states, actions, transition probabilities, and rewards.
+ `Bellman Equation`: A fundamental equation in dynamic programming that expresses the value of a state in terms of the values of its successor states.
+ `Value Functions`: Value functions quantify the expected return of being in a particular state (V(s)) or taking a specific action in a state (Q(s, a)).
+ `Optimal Value Function`: The optimal value function represents the maximum achievable value for each state or state-action pair in an environment.

### +  Value-Based Methods:

Deep Reinforcement Learning (DRL) has revolutionized the field of artificial intelligence by enabling agents to learn complex behaviors through interaction with their environment. Among the various approaches in DRL, value-based methods stand out for their ability to approximate the value of states or state-action pairs, guiding the agent towards optimal decision-making.

+ `Q-Learning`: Q-Learning is a classic reinforcement learning algorithm that learns the optimal action-value function by iteratively updating Q-values based on the Bellman equation. In Deep Q-Learning (DQN), deep neural networks are used to approximate the Q-values of state-action pairs. It involves an experience replay buffer to stabilize learning and target networks to improve convergence.  ([qlearning.py](./code/qlearning.py))
    - `Temporal Difference (TD) Learning`: Q-Learning employs TD learning to update Q-values by bootstrapping from successor states' estimates.
    - `Exploration vs. Exploitation`: Balancing exploration (trying new actions) and exploitation (choosing actions based on current knowledge) is crucial for Q-Learning's convergence and effectiveness.

+ `Deep Q-Networks (DQN)`: DQN extends Q-Learning to high-dimensional state spaces using deep neural networks to approximate the Q-function.
    - `Experience Replay`: Storing agent experiences in a replay buffer and sampling batches of experiences randomly for training improves data efficiency and stabilizes learning.
    - `Target Networks`: Utilizing separate target networks to compute target Q-values stabilizes training by decoupling target and current network parameters.

+ `Double Deep Q-Networks (DDQN)`: DDQN improves upon DQN by mitigating overestimation bias in Q-values, enhancing stability and learning. Mitigates overestimation bias by decoupling action selection and value estimation.  ([ddqn.py](./code/ddqn.py))
  
Extensions and Enhancements:
+ `Prioritized Experience Replay`: This technique assigns different priorities to experiences in the replay buffer, allowing the agent to focus on learning from more informative experiences.
+ `Dueling DQN`: Separates the estimation of state value and advantage value, improving learning efficiency.
+ `Rainbow DQN`: Combines multiple DQN extensions (e.g., double Q-learning, prioritized experience replay, dueling networks) for enhanced performance.


### +  Policy-Based Methods:
Policy-based methods represent a fundamental approach in Deep Reinforcement Learning (DRL), focusing on directly learning a policy that maps states to actions, rather than estimating value functions. A policy defines the agent's behavior by specifying the probability distribution over actions given states. The theoretical basis for policy optimization, stating that the gradient of the expected return with respect to the policy parameters can be computed analytically.

- `Parametric Policies`: Policies are typically parameterized by a set of learnable parameters, often represented by neural networks.
- `Deterministic vs. Stochastic Policies`: Deterministic policies directly output actions, while stochastic policies output action probabilities, enabling exploration.

+ <b>`Policy Gradients`:</b> Policy gradient methods directly optimize the policy function by maximizing expected cumulative rewards. Algorithms like REINFORCE and Trust Region Policy Optimization (TRPO) fall under this category.
+ <b>`Proximal Policy Optimization (PPO)`:</b> PPO is a popular policy gradient algorithm that addresses some of the limitations of TRPO. It uses a clipped surrogate objective to ensure stable updates. PPO constrains policy updates to ensure they do not deviate too far from the previous policy, enhancing stability and sample efficiency. PPO employs a clipped surrogate objective to prevent large policy updates, mitigating the risk of policy collapse.



### +  Actor-Critic Methods:

Actor-Critic methods represent a powerful class of algorithms in Deep Reinforcement Learning (DRL), combining the strengths of both policy-based and value-based approaches. The Actor-Critic architecture consists of two components:

    + `Actor`: The policy network that learns to select actions based on states.
    + `Critic`: The value function network that evaluates the goodness of actions or state-action pairs.

`Temporal Difference (TD) Learning`: Actor-Critic methods leverage TD learning to update both the actor (policy) and critic (value function) networks iteratively based on temporal difference errors.

+  <b>`Advantage Actor-Critic (A2C)`:</b> A2C combines the benefits of both policy gradients and value-based methods by using an actor (policy) network and a critic (value) network. It reduces variance compared to vanilla policy gradients. A2C uses an advantage function to estimate the advantage of taking a particular action over the average action value, guiding policy updates. A2C employs multiple parallel environments with asynchronous updates to improve sample efficiency and accelerate learning.

+  <b>`A3C (Asynchronous Advantage Actor-Critic)`:</b> A3C employs parallelism to speed up training. Multiple agents explore different trajectories simultaneously, improving sample efficiency. A3C employs asynchronous training across multiple threads or processes, enabling efficient exploration and utilization of computational resources. A3C scales to large-scale environments and distributed settings, making it suitable for training deep neural networks on complex tasks.

+ <b>`DDPG (Deep Deterministic Policy Gradient)`:</b> DDPG extends Actor-Critic methods to problems with continuous action spaces, utilizing a deterministic policy network to output continuous actions. DDPG utilizes experience replay to stabilize learning by storing and sampling experiences from a replay buffer.

Distributed Actor-Critic (DAC): DAC extends A3C to distributed environments, allowing for even greater scalability and parallelization.

### +  Model-Based Methods:

Model-based methods in Deep Reinforcement Learning (DRL) aim to learn a model of the environment dynamics, enabling agents to plan ahead and make informed decisions.

+ <b>`Monte Carlo Tree Search (MCTS)`:</b> MCTS is a model-based algorithm used for decision-making in games and planning. It builds a search tree by simulating possible actions and outcomes.

+ <b>`Model-Predictive Control (MPC)`:</b> MPC uses a learned model of the environment to optimize a sequence of actions. It is commonly used in robotics and control tasks.

Model-Based Deep Reinforcement Learning:

  + `Imagination-Augmented Agents (I2A)`: I2A integrates an internal model (imagination core) with an external policy network, allowing agents to plan and reason about future trajectories.
  + `Model-Based Value Expansion (MBVE)`: MBVE enhances value estimation by leveraging learned environment models to simulate future trajectories and estimate the value of states more accurately.



### +  Exploration Strategies:

+ <b>`Epsilon-Greedy`:</b> In epsilon-greedy strategies, the agent selects the best-known action with probability (1 - ε) and explores randomly with probability ε.

+ <b>`Softmax Exploration`:</b> Softmax exploration assigns probabilities to each action based on their estimated values, allowing for a more controlled form of exploration.



### +  Deep Reinforcement Learning for Continuous Control:
+ <b>`Deep Deterministic Policy Gradient (DDPG)`:</b> DDPG is a popular algorithm for continuous action spaces, combining Q-learning and policy gradients.

+ <b>`Trust Region Policy Optimization (TRPO)`:</b> TRPO is a policy optimization method that ensures stable updates by limiting policy changes based on a trust region constraint.

+ <b>`SAC (Soft Actor-Critic)`:</b> SAC is an off-policy algorithm designed for stochastic policies. It incorporates entropy regularization to encourage exploration.



### +  Multi-Agent Reinforcement Learning:

 + <b>`MARL (Multi-Agent Reinforcement Learning)`:</b> MARL deals with scenarios where multiple agents interact in a shared environment. Algorithms like MADDPG and multi-agent PPO are used for such tasks.


### +  Deep Reinforcement Learning for Robotics:

+ <b>`Robotic Control`:</b> DRL has found applications in robotic control, including tasks like robotic manipulation, locomotion, and navigation.

+ <b>`Sim2Real Transfer`:</b> Research focuses on transferring DRL policies trained in simulation to real-world robotic platforms.

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

## Bellman Equations : 

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