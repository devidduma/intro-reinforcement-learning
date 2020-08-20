from mc_agent import MCAgent, VisitState
from environment import Env


class IMCAgent(MCAgent):
    def __init__(self, actions):
        super(IMCAgent, self).__init__(actions)

    # for every episode, agent updates v function of visited states
    def update(self):
        all_states = super(IMCAgent, self).update()
        self.incremental_mc(all_states)

    def incremental_mc(self, all_states):
        for state in all_states:
            self.update_global_visit_state(state[0], state[1])

    # redefined update visited states for incremental MC
    def update_global_visit_state(self, state_name, G_t):
        updated = False
        if state_name in self.value_table:
            state = self.value_table[state_name]
            state.N = state.N + 1
            learning_rate = 0.5 * 1 / state.N
            state.V = state.V + learning_rate * (G_t - state.V)
            updated = True
        if not updated:
            self.value_table[state_name] = VisitState(total_G=G_t, N=1, V=G_t)


# main loop
if __name__ == "__main__":
    env = Env()
    agent = IMCAgent(actions=list(range(env.n_actions)))

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
