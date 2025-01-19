from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import random
env = gym_super_mario_bros.make('SuperMarioBros-v3')
env = JoypadSpace(env, SIMPLE_MOVEMENT)

done = True
print(env.action_space)
for step in range(2000):
    if done:
        observation= env.reset()
    #observation, reward, terminated, truncated, info  = env.step(env.action_space.sample())
    action = random.choice(range(env.action_space.n))
    (state, reward, done, info) = env.step(action)
    env.render()
    terminated = done  # If terminated/truncated distinction is required
    truncated = False  # Or some logic based on the environment

env.close()