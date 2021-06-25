import concurrent.futures
import copy
from concurrent.futures import ThreadPoolExecutor, ALL_COMPLETED

import gym
import pylab
import random
import numpy as np
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import Sequential

EPISODES = 1000


# this is Deep MC Q Evaluation Agent for the GridWorld
# Utilize Neural Network as q function approximator
class DeepMCQEvalAgent:
    def __init__(self):
        self.load_model = False
        # actions which agent can do
        self.action_space = [0, 1]
        # get size of state and action
        self.action_size = len(self.action_space)
        self.state_size = 4
        self.discount_factor = 0.99
        self.learning_rate = 0.01
        self.learning_rate_decay = 0.01

        self.epsilon = 1.  # exploration
        self.epsilon_decay = 0.999997
        self.epsilon_min = 0.01
        self.model = self.build_model()

        self.samples = []
        self.global_step = 0
        self.total_score = 0

        if self.load_model:
            self.epsilon = 0.05
            self.model.load_weights('./save_model/deep_mc_q_eval.h5')

    # approximate Q function using Neural Network
    # state is input and Q Value of each action is output of network
    def build_model(self):
        model = Sequential()
        model.add(Dense(30, input_dim=self.state_size, activation='relu'))
        model.add(Dense(60, activation='relu'))
        model.add(Dense(60, activation='relu'))
        model.add(Dense(30, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.summary()
        model.compile(loss='categorical_crossentropy',
                      optimizer=Adam(lr=self.learning_rate, decay=self.learning_rate_decay))
        return model

    # get action from model using epsilon-greedy policy
    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            # The agent acts randomly
            return random.randrange(self.action_size)
        else:
            # Predict the reward value based on the given state
            state = np.array(state, dtype=np.float32).reshape(-1, self.state_size)
            q_values = self.model.predict(state)
            return np.argmax(q_values[0])

    def train_model(self, visit_state_batch, action_batch, G_t_batch):
        target_batch = self.model.predict(visit_state_batch)
        # update target with observed G_t
        for target, action, G_t in zip(target_batch, action_batch, G_t_batch):
            target[action] = G_t

        # make batches with target G_t (returns)
        # and do the model fit!
        self.model.fit(visit_state_batch, target_batch, epochs=2, verbose=0, batch_size=32)

    # for every episode, calculate return of visited states
    def calculate_returns(self):
        # state name and G for each state as appeared in the episode
        all_states = []
        G = 0
        for reward in reversed(self.samples):
            G = reward[1] + self.discount_factor * G
            state_info = reward[0]
            action = reward[1]
            done = reward[3]
            all_states.append([state_info, action, G, done])

            # reset G if done
            if done:
                G = 0

        all_states.reverse()

        return all_states

    def first_or_every_visit_mc(self, first_visit=False):
        all_states = self.calculate_returns()
        visit_state_batch = []
        action_batch = []
        G_t_batch = []

        visit_state = []
        for state in all_states:
            # extract info from tuple
            state_info = state[0]
            action = state[1]
            G_t = state[2]
            done = state[3]

            # calculate according to EV or FV
            if not first_visit or str(state_info) not in visit_state:
                visit_state.append(str(state_info))

                visit_state_batch.append(state_info)
                action_batch.append(action)
                G_t_batch.append(G_t)

            # clear if first visit
            if first_visit:
                if done:
                    visit_state.clear()

        visit_state_batch = np.array(visit_state_batch, dtype=np.float32).reshape(-1, self.state_size)

        # print(np.shape(visit_state_batch))
        # print(np.shape(action_batch))
        # print(np.shape(G_t_batch))
        self.train_model(visit_state_batch, action_batch, G_t_batch)

    def mainloop(self, agents_count, first_visit=False):
        scores, episodes = [], []

        executor = ThreadPoolExecutor(agents_count)
        for e in range(EPISODES):
            env = []
            # create envs
            for i in range(agents_count):
                env.append(gym.make("CartPole-v1"))

            self.total_score = 0
            futures_list = []
            # execute agent in env
            for i in range(agents_count):
                futures_list.append(executor.submit(self.sample_episode, e, env[i]))

            # wait for futures to stop
            concurrent.futures.wait(futures_list, timeout=None, return_when=ALL_COMPLETED)

            # avg score calculation
            avg_score = self.total_score / agents_count

            self.first_or_every_visit_mc(first_visit=first_visit)
            self.samples.clear()

            scores.append(avg_score)
            episodes.append(e)
            pylab.plot(episodes, scores, 'b')
            pylab.savefig("./save_graph/deep_mc_q_eval2.png")

            self.print_info(e, avg_score)

    def sample_episode(self, e, env):
        done = False
        state = env.reset()
        state = np.reshape(state, [self.state_size])

        sample = []

        while not done:
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

            # fresh env
            self.global_step += 1

            # get action for the current state and go one step in environment
            action = self.get_action(state)
            next_state, reward, done, info = env.step(action)
            next_state = np.reshape(next_state, [self.state_size])

            # save tuple to episode
            sample.append([state, action, reward, False])

            state = next_state
            # every time step we do training
            self.total_score += reward

            state = copy.deepcopy(next_state)

            if done:
                # last tuple
                action = agent.get_action(state)
                sample.append([state, action, 1, True])

                # save to main memory
                self.samples.extend(sample)

                return

    def print_info(self, e, score):
        print("episode:", e,
              "\tscore:", score,
              "\tglobal_step:", self.global_step,
              "\tepsilon:", agent.epsilon,
              "\tlearning_rate_decay:", agent.learning_rate_decay
              )


if __name__ == "__main__":
    agent = DeepMCQEvalAgent()
    agent.mainloop(agents_count=60, first_visit=False)
