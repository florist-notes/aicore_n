import tensorflow as tf
import numpy as np
import gym

# Define a Double Q-Network
class DoubleQNetwork(tf.keras.Model):
    def __init__(self, num_actions):
        super(DoubleQNetwork, self).__init__()
        self.fc1 = tf.keras.layers.Dense(24, activation='relu')
        self.fc2 = tf.keras.layers.Dense(24, activation='relu')
        self.output_layer = tf.keras.layers.Dense(num_actions, activation='linear')

    def call(self, state):
        x = self.fc1(state)
        x = self.fc2(x)
        return self.output_layer(x)

# Hyperparameters
learning_rate = 0.001
discount_factor = 0.99
num_episodes = 1000
batch_size = 64
target_update_freq = 100

# Create the CartPole-v1 environment
env = gym.make('CartPole-v1')
num_states = env.observation_space.shape[0]
num_actions = env.action_space.n

# Create the DDQN networks
online_network = DoubleQNetwork(num_actions)
target_network = DoubleQNetwork(num_actions)
target_network.set_weights(online_network.get_weights())  # Initialize the target network weights

# Define the loss function and optimizer
loss_fn = tf.keras.losses.Huber()
optimizer = tf.keras.optimizers.Adam(learning_rate)

# Create a replay buffer
replay_buffer = []

# Epsilon-greedy exploration
epsilon_start = 1.0
epsilon_final = 0.01
epsilon_decay = 0.995

def epsilon_greedy_action(state, epsilon):
    if np.random.rand() < epsilon:
        return env.action_space.sample()  # Random action
    else:
        q_values = online_network(state)
        return np.argmax(q_values)

# Training loop
episode_rewards = []

for episode in range(num_episodes):
    state = env.reset()
    episode_reward = 0

    while True:
        epsilon = max(epsilon_final, epsilon_start * epsilon_decay**episode)
        action = epsilon_greedy_action(tf.convert_to_tensor([state], dtype=tf.float32), epsilon)

        next_state, reward, done, _ = env.step(action)

        episode_reward += reward

        replay_buffer.append((state, action, reward, next_state, done))

        if len(replay_buffer) >= batch_size:
            minibatch = np.array(replay_buffer)[np.random.choice(len(replay_buffer), batch_size, replace=False)]
            states, actions, rewards, next_states, dones = zip(*minibatch)

            # Convert to tensors
            states = tf.convert_to_tensor(states, dtype=tf.float32)
            next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
            rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
            actions = tf.convert_to_tensor(actions, dtype=tf.int32)
            dones = tf.convert_to_tensor(dones, dtype=tf.float32)

            # Compute Q-values for the current and next states
            online_q_values = online_network(states)
            target_q_values = target_network(next_states)
            max_actions = tf.argmax(online_q_values, axis=1)
            target_indices = tf.stack([tf.range(batch_size), max_actions], axis=1)
            target_q_values = tf.gather_nd(target_q_values, target_indices)

            target_q_values = rewards + (1.0 - dones) * discount_factor * target_q_values

            with tf.GradientTape() as tape:
                predicted_q_values = tf.reduce_sum(tf.one_hot(actions, num_actions) * online_q_values, axis=1)
                loss = loss_fn(target_q_values, predicted_q_values)

            grads = tape.gradient(loss, online_network.trainable_variables)
            optimizer.apply_gradients(zip(grads, online_network.trainable_variables))

        if done:
            break

    if episode % target_update_freq == 0:
        target_network.set_weights(online_network.get_weights())

    episode_rewards.append(episode_reward)
    print(f"Episode {episode}, Reward: {episode_reward}")

# Close the environment
env.close()

# Calculate and print average rewards
avg_reward = sum(episode_rewards) / len(episode_rewards)
print(f"Average Reward: {avg_reward}")
