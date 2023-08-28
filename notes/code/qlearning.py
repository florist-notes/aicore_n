import numpy as np

# Define the grid world environment
# 'S' represents the start state, 'G' represents the goal state,
# 'H' represents a hole, and 'F' represents free space.
grid = [
    ['S', 'F', 'F', 'F'],
    ['F', 'H', 'F', 'H'],
    ['F', 'F', 'F', 'H'],
    ['H', 'F', 'F', 'G']
]

# Define actions (up, down, left, right)
actions = [(0, -1), (0, 1), (-1, 0), (1, 0)]

# Define Q-learning hyperparameters
learning_rate = 0.1
discount_factor = 0.9
epsilon = 0.1
num_episodes = 1000

# Initialize Q-table with zeros
num_states = len(grid) * len(grid[0])
num_actions = len(actions)
q_table = np.zeros((num_states, num_actions))

# Map from state coordinates to state index
state_to_index = {(i, j): i * len(grid[0]) + j for i in range(len(grid)) for j in range(len(grid[0]))}

# Convert grid world to a list of states (excluding holes and the goal)
states = [(i, j) for i in range(len(grid)) for j in range(len(grid[0])) if grid[i][j] not in ('H', 'G')]

# Q-learning algorithm
for episode in range(num_episodes):
    state = states[np.random.randint(0, len(states))]  # Start from a random state
    done = False

    while not done:
        # Epsilon-greedy policy
        if np.random.rand() < epsilon:
            action = np.random.randint(0, num_actions)
        else:
            state_index = state_to_index[state]
            action = np.argmax(q_table[state_index, :])

        # Take the selected action
        next_state = (state[0] + actions[action][0], state[1] + actions[action][1])

        # Check for valid next state (within the grid)
        if 0 <= next_state[0] < len(grid) and 0 <= next_state[1] < len(grid[0]):
            if grid[next_state[0]][next_state[1]] == 'G':
                reward = 1  # Goal reached
                done = True
            elif grid[next_state[0]][next_state[1]] == 'H':
                reward = -1  # Hole encountered
                done = True
            else:
                reward = 0  # Free space

            # Q-learning update rule
            state_index = state_to_index[state]
            next_state_index = state_to_index[next_state]
            q_table[state_index, action] = (1 - learning_rate) * q_table[state_index, action] + \
                learning_rate * (reward + discount_factor * np.max(q_table[next_state_index, :]))

            state = next_state

# Print the learned Q-table
print("Learned Q-table:")
print(q_table)

# Optimal policy (finding the best actions for each state)
optimal_policy = np.argmax(q_table, axis=1)
optimal_policy = [actions[action] if grid[state[0]][state[1]] != 'G' else 'G' for state, action in zip(states, optimal_policy)]
optimal_policy = np.array(optimal_policy).reshape(len(grid), len(grid[0]))

print("Optimal Policy:")
print(optimal_policy)
