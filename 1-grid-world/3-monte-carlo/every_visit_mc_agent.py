from mc_agent import MCAgent
from environment import Env


class EVMCAgent(MCAgent):
    def __init__(self, actions):
        super(EVMCAgent, self).__init__(actions)

    # for every episode, agent updates q function of visited states
    def update(self):
        all_states = super(EVMCAgent, self).update()
        # use either first visit, every visit or incremental MC
        self.every_visit_mc(all_states)

    def every_visit_mc(self, all_states):
        for state in all_states:
                self.update_global_visit_state(state[0], state[1])
        for state in self.visit_state:
            self.value_table[state.name] = state.V


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
                print("episode : ", episode)
                agent.update()
                agent.samples.clear()
                break
