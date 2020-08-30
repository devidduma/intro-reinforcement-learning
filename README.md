Reinforcement Learning algorithms as presented on the [CS234 Course](http://web.stanford.edu/class/cs234/CS234Win2019/index.html) by Prof. Emma Brunskill. ([Lecture videos](https://www.youtube.com/playlist?list=PLoROMvodv4rOSOPzutgyCTapiGlY2Nd8u))
Forked from rlcode/reinforcement-learning

This repo provides easy-to-read code examples and will cover every topic and algorithm presented on the CS234 course. One file for each algorithm.

Currently under development.

## Dependencies
1. Python
2. Tensorflow
3. Keras
4. numpy
5. pandas
6. matplot
7. pillow
8. Skimage
9. h5py

### Install Requirements
```
pip install -r requirements.txt
```

## Table of Contents

**Grid World** - Mastering the basics of reinforcement learning in the simplified world called "Grid World"

- [Policy Iteration](./1-grid-world/1-policy-iteration)
- [Value Iteration](./1-grid-world/2-value-iteration)
- [Monte Carlo](./1-grid-world/3-monte-carlo)
    - [First Visit MC](./1-grid-world/3-monte-carlo/first_visit_mc_agent.py)
    - [Every Visit MC](./1-grid-world/3-monte-carlo/every_visit_mc_agent.py)
    - [Incremental MC](./1-grid-world/3-monte-carlo/incremental_mc_agent.py)
- [Temporal Difference](./1-grid-world/4-temporal-difference)
- [Monte Carlo Q Evaluation](./1-grid-world/5-monte-carlo-q-evaluation)
    - [First Visit MCQE](./1-grid-world/5-monte-carlo-q-evaluation/first_visit_mc_q_eval_agent.py)
    - [Every Visit MCQE](./1-grid-world/5-monte-carlo-q-evaluation/every_visit_mc_q_eval_agent.py)
    - [Incremental MCQE](./1-grid-world/5-monte-carlo-q-evaluation/incremental_mc_q_eval_agent.py)
- [SARSA](1-grid-world/6-sarsa)
- [Q-Learning](1-grid-world/7-q-learning)
- [Double Q-Learning](1-grid-world/8-double-q-learning)
- [Deep MC Q Evaluation (Batch)](1-grid-world/8-deep-monte-carlo-q-evaluation)
- [Deep SARSA](1-grid-world/8-deep-sarsa)
- [REINFORCE](1-grid-world/9-reinforce)

**CartPole** - Applying deep reinforcement learning on basic Cartpole game.

- [Deep Q Network](./2-cartpole/1-dqn)
- [Double Deep Q Network](./2-cartpole/2-double-dqn)
- [Policy Gradient](./2-cartpole/3-reinforce)
- [Actor Critic (A2C)](./2-cartpole/4-actor-critic)
- [Asynchronous Advantage Actor Critic (A3C)](./2-cartpole/5-a3c)

**Atari** - Mastering Atari games with Deep Reinforcement Learning

- **Breakout** - [DQN](./3-atari/1-breakout/breakout_dqn.py), [DDQN](./3-atari/1-breakout/breakout_ddqn.py) [Dueling DDQN](./3-atari/1-breakout/breakout_ddqn.py) [A3C](./3-atari/1-breakout/breakout_a3c.py)
- **Pong** - [Policy Gradient](./3-atari/2-pong/pong_reinforce.py)

**OpenAI GYM** - [WIP]

- Mountain Car - [DQN](./4-gym/1-mountaincar)
