import numpy as np
import random
from collections import defaultdict
from environment import Env


class VisitStateAction:
    def __init__(self, total_G = 0, N = 0, Q = 0):
        self.total_G = total_G
        self.N = N
        self.Q = Q

# Monte Carlo Agent which learns every episodes from the sample
class MCAgent:
    def __init__(self, actions):
        self.width = 5
        self.height = 5
        self.actions = actions
        self.discount_factor = 0.9
        self.decaying_epsilon_counter = 1
        self.decaying_epsilon_mul_factor = 0.2
        self.samples = []
        self.q_value_table = defaultdict(VisitStateAction)

    # append sample to memory(state, reward, done)
    def save_sample(self, state, action, reward, done):
        self.samples.append([state, action, reward, done])

    # for every episode, agent updates q function of visited state action pairs
    def update(self):
        # state action name and G for each state as appeared in the episode
        all_states = []
        G = 0
        for reward in reversed(self.samples):
            # reward[0] state info, reward[1] action info
            state_action_name = str([reward[0], reward[1]])
            G = reward[2] + self.discount_factor * G
            all_states.append([state_action_name, G])
        all_states.reverse()

        return all_states

    # update visited states for first visit or every visit MC
    def update_global_q_value_table(self, state_action_name, G_t):
        updated = False
        if state_action_name in self.q_value_table:
            state_action = self.q_value_table[state_action_name]
            state_action.total_G = state_action.total_G + G_t
            state_action.N = state_action.N + 1
            state_action.Q = state_action.total_G / state_action.N
            updated = True
        if not updated:
            self.q_value_table[state_action_name] = VisitStateAction(total_G=G_t, N=1, Q=G_t)


    # get action for the state according to the q function table
    # agent pick action of epsilon-greedy policy
    def get_action(self, state):
        epsilon = 1 / (self.decaying_epsilon_counter * self.decaying_epsilon_mul_factor)
        if np.random.rand() < epsilon:
            # take random action
            action = np.random.choice(self.actions)
        else:
            # take action according to the q function table
            q_values = self.possible_Q_values(state)
            action = self.arg_max(q_values)
        return int(action)

    # compute arg_max if multiple candidates exit, pick one randomly
    @staticmethod
    def arg_max(next_state):
        max_index_list = []
        max_value = next_state[0]
        for index, value in enumerate(next_state):
            if value > max_value:
                max_index_list.clear()
                max_value = value
                max_index_list.append(index)
            elif value == max_value:
                max_index_list.append(index)
        return random.choice(max_index_list)

    # get the possible next states
    def possible_Q_values(self, state):

        Q_values = [self.q_value_table[str([state, x])].Q for x in range(4)]

        return Q_values
