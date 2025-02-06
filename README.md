# What is reinforce learning
*  Component and goal of reinforce learning
![image](https://hackmd.io/_uploads/ryr-kS2z1x.png)

![image](https://hackmd.io/_uploads/H1sN1Bhfye.png)

* observation (state)/action/reward

input(state)->output (action) : goal ->maximum reward

reinforced learning uses the state of the current enviornment to calculate the maximum reward and decide its next action

### Method of reinforce learning

![image](https://hackmd.io/_uploads/Hy6I1SnMJe.png)
![image](https://hackmd.io/_uploads/By_pyr2fkg.png)

* what is policy based: 
    directly learn s-->a
    continous state:Deterministic Policy Gradient

* what is value based will choose the action that has maximum value:
    *  only consider state 
        only evaluate now state and new state has what expected reward

        Descrete : Monte-carlo/**TD error** 

    * consider state/action :  

        * input: state /action->output expected reward
        Discrete: **Q learning**  /SARSA
        Continuous:  **DQN(Deep nearal networ)**
        
:::info 
-> most method is combined value/ policy : soft actor-critic, DDPG
->action will choose has maximum expected reward from critic, critic will learn more correctly expected reward from receive data
:::

s(t) ,choose best a(t) ->r(t+1)

cho

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

# Priortized buffer




## Whar is Experience Replay Buffer:

The buffer stores past experiences (state, action, reward, next state) collected by the agent.
Instead of training on the latest experiences, it randomly samples past experiences to break correlation and stabilize learning.

## What is Prioritization in Sampling:

Unlike uniform sampling, PER gives priority to more important experiences.
Importance is usually measured by Temporal-Difference (TD) error, which shows how surprising or incorrect the agent's predictions were.

## How It Works:

Calculate TD error: Each experience is assigned a priority based on the TD error.
Sample based on priority: Experiences with higher TD error have a higher probability of being replayed.
Update priorities: After the experience is used for training, the priority is updated.

the sample formula as following:
![image](https://hackmd.io/_uploads/SyCCVbdw1g.png)
