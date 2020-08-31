from mc_agent import MCAgent
from environment import Env


class EVMCAgent(MCAgent):
    def __init__(self, actions):
        super(EVMCAgent, self).__init__(actions)

    # for every episode, agent updates v function of visited states
    def update(self):
        all_states = super(EVMCAgent, self).update()
        self.every_visit_mc(all_states)

    def every_visit_mc(self, all_states):
        for state in all_states:
                self.update_global_value_table(state[0], state[1])


# main loop
if __name__ == "__main__":
    env = Env()
    agent = EVMCAgent(actions=list(range(env.n_actions)))

    for episode in range(1000):
        state = env.reset()
        action = agent.get_action(state)

        while True:
            env.render()

            # forward to next state. reward is number and done is boolean
            next_state, reward, done = env.step(action)
            agent.save_sample(next_state, reward, done)

            # get next action
            action = agent.get_action(next_state)

            # at the end of each episode, update the v function table
            if done:
                print("episode : ", episode, "\t[3, 2]: ", round(agent.value_table["[3, 2]"].V, 2),
                      " [2, 3]:", round(agent.value_table["[2, 3]"].V, 2), " [2, 2]:", round(agent.value_table["[2, 2]"].V, 2),
                      "\tepsilon: ", round(agent.epsilon, 2))
                agent.update()
                agent.samples.clear()
                break
