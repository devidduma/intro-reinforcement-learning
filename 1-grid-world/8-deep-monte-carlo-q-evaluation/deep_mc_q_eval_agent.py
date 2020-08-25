import copy
import pylab
import random
import numpy as np
from environment import Env
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
        self.action_space = [0, 1, 2, 3, 4]
        # get size of state and action
        self.action_size = len(self.action_space)
        self.state_size = 15
        self.discount_factor = 0.99
        self.learning_rate = 0.001

        self.epsilon = 1.  # exploration
        self.epsilon_decay = .999
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
        self.model.fit(visit_state_batch, target_batch, epochs=4, verbose=0, batch_size=4)

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

    def first_or_every_visit_mc(self, first_visit=True):
        all_states = self.calculate_returns()
        visit_state_batch = []
        action_batch = []
        G_t_batch = []

        visit_state = []
        for state in all_states:
            state_info = state[0]
            action = state[1]
            G_t = state[2]
            done = state[3]
            if not first_visit or str(state_info) not in visit_state:
                visit_state.append(str(state_info))

                visit_state_batch.append(state_info)
                action_batch.append(action)
                G_t_batch.append(G_t)

        visit_state_batch = np.array(visit_state_batch, dtype=np.float32).reshape(-1, self.state_size)

        #print(np.shape(visit_state_batch))
        #print(np.shape(action_batch))
        #print(np.shape(G_t_batch))
        self.train_model(visit_state_batch, action_batch, G_t_batch)


if __name__ == "__main__":
    env = Env()
    agent = DeepMCQEvalAgent()

    global_step = 0
    scores, episodes = [], []

    for e in range(EPISODES):
        done = False
        score = 0
        state = env.reset()
        state = np.reshape(state, [15])

        while not done:
            if agent.epsilon > agent.epsilon_min:
                agent.epsilon *= agent.epsilon_decay

            # fresh env
            global_step += 1

            # get action for the current state and go one step in environment
            action = agent.get_action(state)
            next_state, reward, done = env.step(action)
            next_state = np.reshape(next_state, [15])

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
                pylab.savefig("./save_graph/deep_mc_q_eval_.png")

                # we don't need last tuple
                # agent.save_sample(state, action, 0, True)

                agent.first_or_every_visit_mc(first_visit=False)
                agent.samples.clear()

                print("episode:", e, "  score:", score, "global_step",
                      global_step, "  epsilon:", agent.epsilon)

        if e % 100 == 0:
            agent.model.save_weights("./save_model/deep_mc_q_eval.h5")
