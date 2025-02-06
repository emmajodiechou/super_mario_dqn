from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import random
import torch.nn.functional as F

env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT)
done = True

from os.path import join
from shutil import copyfile, move
import torch
from torch import save
from torch.optim import Adam
import random as rand


import numpy as np
from collections import deque
from gym import make, ObservationWrapper, wrappers, Wrapper
from gym.spaces import Box
from nes_py.wrappers import JoypadSpace
import numpy as np
import random
import torch
import torch.nn as nn
from random import sample


from torch import FloatTensor, LongTensor
from torch.autograd import Variable

# for step in range(2000):
#     if done:
#         observation= env.reset()
#     #observation, reward, terminated, truncated, info  = env.step(env.action_space.sample())
#     action =random.choice(range(env.action_space.n)) 
#     (state, reward, done, info) = env.step(action)
#     env.render()
#     terminated = done  # If terminated/truncated distinction is required
#     truncated = False  # Or some logic based on the environment



#-----------build an agent can choose best action-------------


# print(state.shape)
# print(type(state))
# import numpy as np
# import torch.nn.functional as F


# state_shape=(240,256,3)
class PrioritizedBuffer:
    def __init__(self, capacity, alpha=0.6):
        self._alpha = alpha
        self._capacity = capacity
        self._buffer = []
        self._position = 0
        self._priorities = np.zeros((capacity,), dtype=np.float32)

    def push(self, state, action, reward, next_state, done):
        state = np.expand_dims(state, 0)
        
        next_state = np.expand_dims(next_state, 0)
        
        max_prio = self._priorities.max() if self._buffer else 1.0

        batch = (state, action, reward, next_state, done)
        if len(self._buffer) < self._capacity:
            self._buffer.append(batch)
        else:
            self._buffer[self._position] = batch

        self._priorities[self._position] = max_prio
        self._position = (self._position + 1) % self._capacity

    def sample(self, batch_size, beta=0.4):
        if len(self._buffer) == self._capacity:
            prios = self._priorities
        else:
            prios = self._priorities[:self._position]

        probs = prios ** self._alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self._buffer), batch_size, p=probs)
        samples = [self._buffer[idx] for idx in indices]
        # print('samples ',samples[0][0].shape)

        # print('samples ',samples[0][1])
        # print('samples ',samples[0][2])
        # print('samples ',samples[0][3])
        # print('samples ',samples[0][4])
        total = len(self._buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)

        batch = list(zip(*samples))
        states = np.concatenate(batch[0])
        #print('states',states.shape)
        actions = batch[1]
        #print('actions',len(actions))
        rewards = batch[2]
        #print('rewards',len(rewards))
        next_states = np.concatenate(batch[3])
        #print('next_states',next_states.shape)
        dones = batch[4]
        return states, actions, rewards, next_states, dones, indices, weights
    

class Agentnet(nn.Module):
    def __init__(self):
        super(Agentnet, self).__init__()
        self._input_shape = (3, 240, 256)  # Corrected input shape (channels, height, width)
        self._num_actions = 7  # Output actions from 0 to 7 (8 actions in total)
        
        # Define convolutional feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(self._input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        # Define fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(self.feature_size, 512),
            nn.ReLU(),
            nn.Linear(512, self._num_actions)  # Output size matches the number of actions
        )

    def forward(self, x):
        # the input shape need to be (batch_size, 3, 240, 256)
        # Pass input through the feature extractor and flatten
        x = self.features(x).view(x.size(0), -1)
        # Pass through fully connected layers
        return self.fc(x)

    @property
    def feature_size(self):
        # Compute the size of the flattened features
        x = self.features(torch.zeros(1, *self._input_shape))  # Shape: (1, channels, height, width)
        return x.view(1, -1).size(1)

class Agent:
    def __init__(self, state_shape=(3, 240, 256), num_actions=7, buffer_capacity=2000, batch_size=32, gamma=0.99):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.state_shape = state_shape
        self.num_actions = num_actions
        self.batch_size = batch_size
        self.gamma = gamma  # Discount factor

        # Only one network (policy network)
        self.policy_net = Agentnet().to(self.device)

        # Optimizer
        self.optimizer = Adam(self.policy_net.parameters(), lr=1e-4)

        # Replay buffer
        self.replay_buffer = PrioritizedBuffer(buffer_capacity)

        # Epsilon-greedy parameters
        self.epsilon_start = 1.0
        self.epsilon_end = 0.1
        self.epsilon_decay = 500  # Decay over steps
        self.steps_done = 0

    def choose_action(self, state, train=True):
        """Select an action using epsilon-greedy policy."""
        self.steps_done += 1
        epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
            np.exp(-1.0 * self.steps_done / self.epsilon_decay)

        if train and random.random() < epsilon:
            # Random action
            return random.randint(0, self.num_actions - 1)
        else:
            #Greedy action
            state = np.transpose(state, (2, 0, 1)).copy()

            state = torch.tensor(state, dtype=torch.float32).to(self.device).unsqueeze(0)
            with torch.no_grad():
                q_values = self.policy_net(state)
            return q_values.argmax(dim=1).item()

    def cache_memory(self, state, action, reward, next_state, done):
        """Store experiences in replay buffer."""
        state = np.transpose(state, (2, 0, 1)).copy()
        next_state = np.transpose(next_state, (2, 0, 1)).copy()
        self.replay_buffer.push(state, action, reward, next_state, done)

    def learn(self):
        """Perform one step of optimization on the policy network."""
        if len(self.replay_buffer._buffer) < self.batch_size:
            return  # Not enough samples

        # Sample a batch of experiences
        states, actions, rewards, next_states, dones, indices, weights = self.replay_buffer.sample(self.batch_size)
        #-----------print_shape)
        # Convert to tensors
        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)
        weights = torch.tensor(weights, dtype=torch.float32).to(self.device)

        # Compute current Q-values
        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Compute target Q-values using the same network (no target network)
        with torch.no_grad():
            next_q_values = self.policy_net(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        # Compute loss (Huber loss)
        loss = (weights * F.smooth_l1_loss(q_values, target_q_values, reduction='none')).mean()

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update priorities in the buffer
        errors = torch.abs(q_values - target_q_values).cpu().detach().numpy()
        for i, idx in enumerate(indices):
            self.replay_buffer._priorities[idx] = errors[i] + 1e-6  # Add small constant to avoid 0 priorities

    def save_policy(self, filepath="policy_net.pth"):
        """Save the policy network's state to a file."""
        torch.save(self.policy_net.state_dict(), filepath)
        print(f"Policy network saved to {filepath}")







agent=Agent()



# # Define the number of frames to skip
frame_skip = 4  # Number of frames to skip
total_reward = 0
# # Training loop
print('training devices:', torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
for i in range(20000):


    action = agent.choose_action(state, train=True)  # Choose action

#     # Initialize variables to accumulate reward and handle frame skipping
    
    for _ in range(frame_skip):
#         # Perform the chosen action
        (next_state, reward, done, info) = env.step(action)

#         # Accumulate the reward
        total_reward += reward
        if done:
            break

#         # If the episode ends during the frame skip, break
#         if done:
#             break

#     # Process the final `next_state` after skipping frames
#     next_state = np.transpose(next_state, (2, 0, 1))  # Convert to (channels, height, width)
#     next_state = torch.tensor(next_state / 255.0, dtype=torch.float32).to(agent.device)

#     # Store the experience in the replay buffer
    agent.cache_memory(state, action, reward, next_state, done)
    
#     # Perform learning
    agent.learn()

    state=next_state

    if i%200==0:
        agent.save_policy('policy_net.pth')
        print("epoch=",i," reward=",reward,' total reward=',total_reward)

env.close()
#     # Update state and episode reward
#     state = next_state
#     episode_reward += total_reward

#     # Print episode progress
#     print(
#         f"Episode Reward: {episode_reward} - "
#         f"Steps Done: {agent.steps_done}"
#     )

#     # Check for termination
#     if info.get("flag_get", False) or done:
#         break






# # import gym
# # import nes_py

# # print(f"Gym version: {gym.__version__}")
# # print(f"nes_py version: {nes_py.__version__}")

# conda activate Mario
# https://pypi.org/project/gym/0.21.0/#files

# conda create -n mario_env python=3.8 -y
# conda activate mario_env

# '''
