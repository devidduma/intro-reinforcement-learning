from mc_agent import MCAgent, VisitState
from environment import Env


class EVMCAgent(MCAgent):
    def __init__(self, actions):
        super(EVMCAgent, self).__init__(actions)

    # for every episode, update V values of visited states
    def mc(self):
        all_states = super(EVMCAgent, self).preprocess_visited_states()
        for state in all_states:
            self.update_global_value_table(state[0], state[1])

    # update V values of visited states for first visit or every visit MC
    def update_global_value_table(self, state_name, G_t):
        updated = False
        if state_name in self.value_table:
            state = self.value_table[state_name]
            state.total_G = state.total_G + G_t
            state.N = state.N + 1
            state.V = state.total_G / state.N
            updated = True
        if not updated:
            self.value_table[state_name] = VisitState(total_G=G_t, N=1, V=G_t)


# main loop
if __name__ == "__main__":
    env = Env()
    agent = EVMCAgent(actions=list(range(env.n_actions)))
    agent.mainloop(env)
