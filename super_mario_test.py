from super_mario_dqn import Agent
import torch
import numpy as np
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import cv2

# Load environment
env = gym_super_mario_bros.make('SuperMarioBros-v3')
env = JoypadSpace(env, SIMPLE_MOVEMENT)

# Initialize agent
agent = Agent()  

# Load trained policy
agent.policy_net.load_state_dict(torch.load('policy_net.pth', map_location=agent.device))
agent.policy_net.eval()  # Set model to evaluation mode

# Reset environment
state = env.reset()
done = False
total_reward = 0

# Play for a fixed number of episodes
for episode in range(5):  # Run 5 test episodes
    print(f"🎮 Starting Test Episode {episode + 1}")
    state = env.reset()
    total_reward = 0
    done = False

    while not done:
        env.render()  # Show Mario playing the game
        
        # Preprocess state (convert to (C, H, W) and normalize)
        state = np.transpose(state, (2, 0, 1))  # Convert (H, W, C) → (C, H, W)
        state = torch.tensor(state / 255.0, dtype=torch.float32).to(agent.device).unsqueeze(0)

        # Choose best action (Greedy policy)
        with torch.no_grad():
            action = agent.policy_net(state).argmax(dim=1).item()

        # Perform action
        next_state, reward, done, info = env.step(action)

        # Update state
        state = next_state
        total_reward += reward

    print(f"🏆 Episode {episode + 1} finished with Total Reward: {total_reward}")

env.close()
print("✅ Testing Complete!")
