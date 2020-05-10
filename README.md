# OpenAI Gym's Taxi-v3 Task
## Q-Learning solution for Taxi Problem from OpenAI Gym
We are using OpenAI Gym's Taxi-v3 environment to design an algorithm to teach a taxi agent to navigate a small gridworld. <br>
The goal is to adapt all that you've learned in the previous lessons to solve a new environment!
<img src=https://video.udacity-data.com/topher/2018/April/5ad260ed_screen-shot-2018-04-14-at-3.13.15-pm/screen-shot-2018-04-14-at-3.13.15-pm.png width="400" height="400">
<br>
**States**: There are 500 possible states, corresponding to 25 possible grid locations, 5 locations for the passenger, and 4 destinations. <br>
**Actions** : There are 6 possible actions, corresponding to moving North, East, South, or West, picking up the passenger, and dropping off the passenger.

#### The workspace contains three files:
1. agent.py: Developed reinforcement learning agent here.
2. monitor.py: The interact function tests how well your agent learns from interaction with the environment.
3. main.py: Run this file in the terminal to check the performance of your agent.

Solving : OpenAI Gym defines "solving" this task as getting average return of 9.7 over 100 consecutive trials.


### Reference
https://arxiv.org/pdf/cs/9905014.pdf
