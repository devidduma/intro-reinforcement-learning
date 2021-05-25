import numpy as np
import random
from collections import defaultdict
from environment import Env


class VisitState:
    def __init__(self, total_G=0, N=0, V=0):
        self.total_G = total_G
        self.N = N
        self.V = V


# Monte Carlo Agent which learns every episodes from the sample
class MCAgent:
    def __init__(self, actions):
        self.width = 5
        self.height = 5
        self.actions = actions
        self.discount_factor = 0.9
        self.decaying_epsilon_counter = 1
        self.decaying_epsilon_mul_factor = 0.2
        self.epsilon = None
        self.samples = []
        self.value_table = defaultdict(VisitState)

    # append sample to memory(state, reward, done)
    def save_sample(self, state, reward, done):
        self.samples.append([state, reward, done])

    # for each episode, calculate discounted returns and return info
    def preprocess_visited_states(self):
        # state name and G for each state as appeared in the episode
        all_states = []
        G = 0
        for reward in reversed(self.samples):
            state_name = str(reward[0])
            G = reward[1] + self.discount_factor * G
            all_states.append([state_name, G])
        all_states.reverse()

        self.decaying_epsilon_counter = self.decaying_epsilon_counter + 1

        return all_states

    # to be defined in children classes
    def mc(self):
        pass

    # update visited states for first visit or every visit MC
    def update_global_value_table(self, state_name, G_t):
        pass

    # get action for the state according to the v function table
    # agent pick action of epsilon-greedy policy
    def get_action(self, state):
        self.epsilon = 1 / (self.decaying_epsilon_counter * self.decaying_epsilon_mul_factor)
        if np.random.rand() < self.epsilon:
            # take random action
            action = np.random.choice(self.actions)
        else:
            # take action according to the v function table
            next_state = self.possible_next_state(state)
            action = self.arg_max(next_state)
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
    def possible_next_state(self, state):
        col, row = state
        next_state = [0.0] * 4

        if row != 0:
            next_state[0] = self.value_table[str([col, row - 1])].V
        else:
            next_state[0] = self.value_table[str(state)].V
        if row != self.height - 1:
            next_state[1] = self.value_table[str([col, row + 1])].V
        else:
            next_state[1] = self.value_table[str(state)].V
        if col != 0:
            next_state[2] = self.value_table[str([col - 1, row])].V
        else:
            next_state[2] = self.value_table[str(state)].V
        if col != self.width - 1:
            next_state[3] = self.value_table[str([col + 1, row])].V
        else:
            next_state[3] = self.value_table[str(state)].V

        return next_state

    # to be called in a main loop
    def mainloop(self, env, verbose=False):
        for episode in range(1000):
            state = env.reset()
            action = self.get_action(state)

            while True:
                env.render()

                # forward to next state. reward is number and done is boolean
                next_state, reward, done = env.step(action)
                self.save_sample(next_state, reward, done)

                # get next action
                action = self.get_action(next_state)

                # at the end of each episode, update the v function table
                if done:
                    if(verbose):
                        print("episode : ", episode, "\t[3, 2]: ", round(self.value_table["[3, 2]"].V, 2),
                              "\t[2, 3]:", round(self.value_table["[2, 3]"].V, 2),
                              "\t[2, 2]:", round(self.value_table["[2, 2]"].V, 2),
                              "\t\tepsilon: ", round(self.epsilon, 2))
                    self.mc()
                    self.samples.clear()
                    break
