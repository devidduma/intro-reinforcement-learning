from mc_agent import MCAgent, VisitState
from environment import Env


class IMCAgent(MCAgent):
    def __init__(self, actions):
        super(IMCAgent, self).__init__(actions)

    # for every episode, update V values of visited states
    def mc(self):
        all_states = super(IMCAgent, self).preprocess_visited_states()
        for state in all_states:
            self.update_global_visit_state(state[0], state[1])

    # redefined V value update of visited states for incremental MC
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
    agent.mainloop(env)
