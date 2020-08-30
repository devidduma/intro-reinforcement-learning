import numpy as np
import random
from environment import Env
from collections import defaultdict

class QLearningAgent:
    def __init__(self, actions):
        # actions = [0, 1, 2, 3]
        self.actions = actions
        self.learning_rate = 0.01
        self.discount_factor = 0.9
        self.epsilon = 0.1
        self.qA_table = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])
        self.qB_table = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])

    # update q function with sample <s, a, r, s'>
    def learn(self, state, action, reward, next_state):
        # choose which table will be updated randomly
        if np.random.rand() < 0.5:
            q_table = self.qA_table
        else:
            q_table = self.qB_table

        current_q = q_table[state][action]
        # using Bellman Optimality Equation to update q function
        QL_Target = reward + self.discount_factor * max(q_table[next_state])
        QL_Error = QL_Target - current_q
        q_table[state][action] = current_q + self.learning_rate * QL_Error

    # get action for the state according to the q function table
    # agent pick action of epsilon-greedy policy
    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            # take random action
            action = np.random.choice(self.actions)
        else:
            # take action according to the q function tables
            state_action_A = self.qA_table[state]
            state_action_B = self.qB_table[state]
            state_action_ABsum = [sum(x) for x in zip(state_action_A, state_action_B)]
            action = self.arg_max(state_action_ABsum)
        return action

    @staticmethod
    def arg_max(state_action):
        max_index_list = []
        max_value = state_action[0]
        for index, value in enumerate(state_action):
            if value > max_value:
                max_index_list.clear()
                max_value = value
                max_index_list.append(index)
            elif value == max_value:
                max_index_list.append(index)
        return random.choice(max_index_list)

if __name__ == "__main__":
    env = Env()
    agent = QLearningAgent(actions=list(range(env.n_actions)))

    for episode in range(1000):
        state = env.reset()

        while True:
            env.render()

            # take action and proceed one step in the environment
            action = agent.get_action(str(state))
            next_state, reward, done = env.step(action)

            # with sample <s,a,r,s'>, agent learns new q function
            agent.learn(str(state), action, reward, str(next_state))

            state = next_state
            env.print_value_all(agent.qA_table, agent.qB_table)

            # if episode ends, then break
            if done:
                print("episode : ", episode)
                break
