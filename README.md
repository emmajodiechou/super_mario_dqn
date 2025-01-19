# super_mario_dqn
we use dqn with prioritized memory to train gym mario system

env is the object in gym supmario v3
action -> interger in 0~6
env.step(action) : env do action and get new state and new reward



# DQN theory:

![image](https://github.com/user-attachments/assets/af898ccd-8458-462c-8264-fc202c4bf2d3)

Reinforce learning component : agent, environment state,reward , we want to choose best action to make total reward maximum


DQN use a network to predict each action q value and choose the action which has max q value.In the beginnning prefer to use random to choose action to explore environment more

The network input is (240*256*3) matix , output is 1*7 matrix

Ｎetwork update formula:  s,a,s',r (s' is t+1 state, r is t+1 reward)

![image](https://github.com/user-attachments/assets/5e6c36b5-2007-44d1-a1ed-ded6442b3201)

![image](https://github.com/user-attachments/assets/de26703a-2988-4564-84ea-0a013d95194f)

![image](https://github.com/user-attachments/assets/db5aae8b-3c57-44ad-8b6d-fc069f3c73f1)

![image](https://github.com/user-attachments/assets/d7eba530-a83b-40c1-a447-6ac66c11cb9f)


# Traning work flow:

![image](https://github.com/user-attachments/assets/45c1facc-d4ef-49c5-9886-7cb025ce1b73)

agent.choose_action(state)->agent.cache_memory(s,a,s',r)-> agent.learn()

# Need defined class:

![image](https://github.com/user-attachments/assets/04f95ef3-ecd8-4483-882d-5d5acdcaf9f4)

