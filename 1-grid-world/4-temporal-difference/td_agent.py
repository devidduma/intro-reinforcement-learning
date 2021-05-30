import numpy as np
import random
from collections import defaultdict
from environment import Env


class Tuple:
    def __init__(self, state, action, reward, next_state, next_action, done):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.next_action = next_action
        self.done = done

# Temporal Difference Agent which learns from each tuple during an episode
# render sleep time updated to 0.01
class TDAgent:
    def __init__(self, actions):
        self.width = 5
        self.height = 5
        self.actions = actions
        self.discount_factor = 1
        self.decaying_epsilon_counter = 1
        self.decaying_epsilon_mul_factor = 0.2
        self.epsilon = None
        self.tuple = None
        self.learning_rate = 1
        self.value_table = defaultdict(float)

    # append sample to memory(state, reward, done)
    def save_tuple(self, tuple):
        self.tuple = tuple

    # for every tuple, agent updates v function of visited states
    def update(self):
        state_name = str(self.tuple.state)
        next_state_name = str(self.tuple.next_state)

        V = self.value_table[state_name]
        next_V = self.value_table[next_state_name]
        reward = self.tuple.reward

        TD_Target = reward + self.discount_factor * next_V
        TD_Error = TD_Target - V
        V = V + self.learning_rate * TD_Error

        self.value_table[state_name] = V

        if self.tuple.done:
            self.value_table[next_state_name] = reward

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
            next_state[0] = self.value_table[str([col, row - 1])]
        else:
            next_state[0] = self.value_table[str(state)]
        if row != self.height - 1:
            next_state[1] = self.value_table[str([col, row + 1])]
        else:
            next_state[1] = self.value_table[str(state)]
        if col != 0:
            next_state[2] = self.value_table[str([col - 1, row])]
        else:
            next_state[2] = self.value_table[str(state)]
        if col != self.width - 1:
            next_state[3] = self.value_table[str([col + 1, row])]
        else:
            next_state[3] = self.value_table[str(state)]

        return next_state


# main loop
if __name__ == "__main__":
    env = Env()
    agent = TDAgent(actions=list(range(env.n_actions)))

    for episode in range(1000):
        state = env.reset()
        action = agent.get_action(state)
        reward = 0

        while True:
            env.render()

            # forward to next state. reward is number and done is boolean
            next_state, next_reward, done = env.step(action)
            # get next action
            next_action = agent.get_action(next_state)

            # save only tuple
            agent.save_tuple(Tuple(state, action, reward, next_state, next_action, False))
            # update v values immediately
            agent.update()
            # clear tuple
            agent.tuple = None

            state = next_state
            action = next_action
            reward = next_reward

            # at the end of each episode, print episode info
            if done:
                # ---- Terminal State
                # save only tuple
                agent.save_tuple(Tuple(state, action, reward, state, action, True))
                # update v values immediately
                agent.update()
                # clear tuple
                agent.tuple = None
                # ----

                agent.decaying_epsilon_counter = agent.decaying_epsilon_counter + 1
                # decaying learning rate
                agent.learning_rate = 1 / (episode + 2)

                print("episode : ", episode, "\t[3, 2]: ", round(agent.value_table["[3, 2]"], 2),
                      " [2, 3]:", round(agent.value_table["[2, 3]"], 2), " [2, 2]:", round(agent.value_table["[2, 2]"], 2),
                      "\tepsilon: ", round(agent.epsilon, 2))
                break
