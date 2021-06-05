import numpy as np
import random
from collections import defaultdict
from environment import Env


# SARSA agent learns every time step from the sample <s, a, r, s', a'>
# render sleep time updated to 0.005
class SARSAgent:
    def __init__(self, actions):
        self.actions = actions
        self.learning_rate = 1
        self.discount_factor = 0.9
        self.decaying_epsilon_counter = 1
        self.decaying_epsilon_mul_factor = 0.1
        self.epsilon = None
        self.q_table = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])

    # with sample <s, a, r, s', a'>, learns new q function
    def learn(self, state, action, reward, next_state, next_action):
        current_q = self.q_table[state][action]
        next_state_q = self.q_table[next_state][next_action]
        SRS_Target = reward + self.discount_factor * next_state_q
        SRS_Error = SRS_Target - current_q
        new_q = current_q + self.learning_rate * SRS_Error
        self.q_table[state][action] = new_q

    # get action for the state according to the q function table
    # agent pick action of epsilon-greedy policy
    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            # take random action
            action = np.random.choice(self.actions)
        else:
            # take action according to the q function table
            state_action = self.q_table[state]
            action = self.arg_max(state_action)
        return action

    # epsilon-greedy policy
    def update_epsilon(self):
        self.epsilon = 1 / (self.decaying_epsilon_counter * self.decaying_epsilon_mul_factor)

    # decaying learning rate satisfying Robbins-Munro sequence
    def update_learning_rate(self):
        self.learning_rate = 1 / (self.decaying_epsilon_counter * self.decaying_epsilon_mul_factor)
        if self.learning_rate > 1:
            self.learning_rate = 1

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

    def mainloop(self, env, verbose = False):
        for episode in range(1000):
            # reset environment and initialize state
            state = env.reset()

            # update epsilon and get action of state from agent
            self.update_epsilon()
            action = self.get_action(str(state))

            while True:
                env.render()

                # take action and proceed one step in the environment
                next_state, reward, done = env.step(action)

                # update epsilon and get next action
                self.update_epsilon()
                next_action = self.get_action(str(next_state))

                # with sample <s,a,r,s',a'>, agent learns new q function
                self.learn(str(state), action, reward, str(next_state), next_action)

                state = next_state
                action = next_action

                # print q function of all states at screen
                env.print_value_all(self.q_table)

                # if episode ends, then break
                if done:
                    self.decaying_epsilon_counter = self.decaying_epsilon_counter + 1
                    # decaying learning rate satisfying Robbins-Munro sequence
                    self.update_learning_rate()

                    if verbose:
                        print("episode: ", episode,
                              "\tepsilon: ", round(self.epsilon, 2),
                              "\tlearning rate: ", round(self.learning_rate, 2)
                              )
                    break


if __name__ == "__main__":
    env = Env()
    agent = SARSAgent(actions=list(range(env.n_actions)))
    agent.mainloop(env, verbose=True)
