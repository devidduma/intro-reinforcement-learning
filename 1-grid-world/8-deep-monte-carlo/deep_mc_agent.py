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
    def save_sample(self, state, reward, done):
        self.samples.append([state, reward, done])

    # approximate Q function using Neural Network
    # state is input and Q Value of each action is output of network
    def build_model(self):
        model = Sequential()
        model.add(Dense(30, input_dim=self.state_size, activation='relu'))
        model.add(Dense(30, activation='relu'))
        model.add(Dense(1, activation='linear'))
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
            v_values = self.model.predict(state)
            return np.argmax(v_values[0])

    def train_model(self, visited_states_batch, G_batch):
        visited_states_batch = np.array(visited_states_batch, dtype=np.float32).reshape([-1, 15])
        #visited_states_batch = tf.convert_to_tensor(visited_states_batch)
        G_batch = np.array(G_batch, dtype=np.float32).reshape(-1, 1)
        #G_batch = tf.convert_to_tensor(G_batch)
        print(visited_states_batch)
        print(G_batch)
        print(np.shape(visited_states_batch))
        print(np.shape(G_batch))

        # make batches with target G_t (returns)
        # and do the model fit!
        self.model.fit(visited_states_batch, G_batch, epochs=1, verbose=0, batch_size=4)

    # for every episode, calculate return of visited states
    def calculate_returns(self):
        # state name and G for each state as appeared in the episode
        all_states = []
        G = 0
        for reward in reversed(self.samples):
            G = reward[1] + self.discount_factor * G
            all_states.append([reward[0], G])
        all_states.reverse()

        return all_states

    def first_visit_mc(self):
        all_states = self.calculate_returns()
        all_states_filtered = []
        all_Gt_filtered = []

        visit_state = []
        for state in all_states:
            if str(state[0]) not in visit_state:
                visit_state.append(str(state[0]))
                all_states_filtered = np.concatenate((all_states_filtered, state[0]), axis=0)
                all_Gt_filtered.append(state[1])


        #print(all_states_filtered)
        #print(all_Gt_filtered)
        self.train_model(all_states_filtered, all_Gt_filtered)

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
        state = np.reshape(state, [15])

        while not done:
            # fresh env
            global_step += 1

            # get action for the current state and go one step in environment
            action = agent.get_action(state)
            next_state, reward, done = env.step(action)
            next_state = np.reshape(next_state, [15])

            # save tuple to episode
            agent.save_sample(next_state, reward, done)

            state = next_state
            # every time step we do training
            score += reward

            state = copy.deepcopy(next_state)

            if done:
                scores.append(score)
                episodes.append(e)
                pylab.plot(episodes, scores, 'b')
                pylab.savefig("./save_graph/deep_mc_.png")

                agent.first_visit_mc()
                agent.samples.clear()

                print("episode:", e, "  score:", score, "global_step",
                      global_step, "  epsilon:", agent.epsilon)

        if e % 100 == 0:
            agent.model.save_weights("./save_model/deep_mc.h5")
