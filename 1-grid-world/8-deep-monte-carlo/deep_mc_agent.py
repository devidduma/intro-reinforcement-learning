import copy
import pylab
import random
import numpy as np
from environment import Env
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import Sequential
import tensorflow as tf

EPISODES = 1000


# this is DeepMC Agent for the GridWorld
# Utilize Neural Network as v function approximator
class DeepMCAgent:
    def __init__(self):
        self.load_model = False
        # actions which agent can do
        self.action_space = [0, 1, 2, 3, 4]
        # get size of state and action
        self.action_size = len(self.action_space)
        self.state_size = 15
        self.discount_factor = 0.99
        self.learning_rate = 0.001

        self.epsilon = 1.  # exploration
        self.epsilon_decay = .96
        self.epsilon_min = 0.01
        self.model = self.build_model()

        self.samples = []

        if self.load_model:
            self.epsilon = 0.05
            self.model.load_weights('./save_model/deep_mc_trained.h5')

    # append sample to memory(state, reward, done)
    def save_sample(self, state, action, reward, done):
        self.samples.append([state, action, reward, done])

    # approximate Q function using Neural Network
    # state is input and Q Value of each action is output of network
    def build_model(self):
        model = Sequential()
        model.add(Dense(30, input_dim=self.state_size, activation='relu'))
        model.add(Dense(30, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.summary()
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    # get action from model using epsilon-greedy policy
    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            # The agent acts randomly
            return random.randrange(self.action_size)
        else:
            # Predict the reward value based on the given state
            state = np.float32(state)
            q_values = self.model.predict(state)
            return np.argmax(q_values[0])


    def train_model(self, visited_state, action, G_t):
        
        target = self.model.predict(visited_state)[0]
        # update target with observed G_t
        target[action] = G_t

        target = np.reshape(target, [1, 5])

        # make batches with target G_t (returns)
        # and do the model fit!
        self.model.fit(visited_state, target, epochs=1, verbose=0)

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
        all_states.reverse()

        return all_states

    def first_visit_mc(self):
        all_states = self.calculate_returns()
        all_states_filtered = []
        all_Gt_filtered = []

        visit_state = []
        for state in all_states:
            state_info = state[0]
            action = state[1]
            G_t = state[2]
            done = state[3]
            if str(state_info) not in visit_state:
                visit_state.append(str(state_info))
                #all_states_filtered = np.concatenate((all_states_filtered, state[0]), axis=0)
                #all_Gt_filtered.append(state[1])
                if not done:
                    self.train_model(state_info, action, G_t)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

if __name__ == "__main__":
    env = Env()
    agent = DeepMCAgent()

    global_step = 0
    scores, episodes = [], []

    for e in range(EPISODES):
        done = False
        score = 0
        state = env.reset()
        state = np.reshape(state, [1, 15])

        while not done:
            # fresh env
            global_step += 1

            # get action for the current state and go one step in environment
            action = agent.get_action(state)
            next_state, reward, done = env.step(action)
            next_state = np.reshape(next_state, [1, 15])

            # save tuple to episode
            agent.save_sample(state, action, reward, False)

            state = next_state
            # every time step we do training
            score += reward

            state = copy.deepcopy(next_state)

            if done:
                scores.append(score)
                episodes.append(e)
                pylab.plot(episodes, scores, 'b')
                pylab.savefig("./save_graph/deep_mc_.png")

                # we don't need last tuple
                #agent.save_sample(state, None, 0, True)

                agent.first_visit_mc()
                agent.samples.clear()

                print("episode:", e, "  score:", score, "global_step",
                      global_step, "  epsilon:", agent.epsilon)

        if e % 100 == 0:
            agent.model.save_weights("./save_model/deep_mc.h5")
