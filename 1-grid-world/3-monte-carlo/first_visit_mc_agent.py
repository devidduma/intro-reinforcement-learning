from mc_agent import MCAgent, VisitState
from environment import Env


class FVMCAgent(MCAgent):
    def __init__(self, actions):
        super(FVMCAgent, self).__init__(actions)

    # for every episode, update V values of visited states
    def mc(self):
        all_states = super(FVMCAgent, self).preprocess_visited_states()
        visit_state = []
        for state in all_states:
            if state[0] not in visit_state:
                visit_state.append(state[0])
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
    agent = FVMCAgent(actions=list(range(env.n_actions)))
    agent.mainloop(env)
