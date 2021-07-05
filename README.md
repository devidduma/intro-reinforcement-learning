## Bachelor Project

### Introduction to Reinforcement Learning

This repository is based on [rlcode/reinforcement-learning](https://github.com/rlcode/reinforcement-learning).
They did an amazing job in providing simple and clean examples of reinforcement learning algorithms.
It helped me a lot in applying theory to practice.

This repository contains 12 reinforcement learning topics with a total of 19 different algorithm implementations.

Forked work:
 - *8 implementations* have been forked from the parent repository. I did some refractoring and hyperparameter tuning.

Work done by me:
 - **4 new topics** and **10 implementations**, including Monte Carlo variations.
 - **1 special** parallel programming implementation of deep Monte Carlo on policy Q evaluation.
 - **12 Jupyter Notebook** documentations for each topic, in order to combine theory with code.

### Install Requirements
```
pip install -r requirements.txt
```

### Table of Contents

- [Policy Iteration](1-policy-iteration/policy_iteration.ipynb)
- [Value Iteration](2-value-iteration/value_iteration.ipynb)
- [Monte Carlo](3-monte-carlo/mc_agent.ipynb)
- [Temporal Difference](4-temporal-difference/td_agent.ipynb)
- [Monte Carlo Q-Evaluation](5-monte-carlo-q-evaluation/mc_q_eval_agent.ipynb)
- [SARSA](6-sarsa/sarsa_agent.ipynb)
- [Q-Learning](7-q-learning/q_learning_agent.ipynb)
- [Double Q-Learning](8-double-q-learning/double_q_learning_agent.ipynb)
- [Deep Monte Carlo Q-Evaluation](9-deep-monte-carlo-q-evaluation/deep_mc_q_eval_agent.ipynb)
- [Deep SARSA](10-deep-sarsa/deep_sarsa_agent.ipynb)
- [Deep Q-Network](11-dqn/cartpole_dqn.ipynb)
- [Double Deep Q-Network](12-double-dqn/cartpole_ddqn.ipynb)