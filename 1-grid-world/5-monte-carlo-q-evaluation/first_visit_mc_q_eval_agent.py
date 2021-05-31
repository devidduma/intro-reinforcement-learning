from mc_q_eval_agent import MCAgent, VisitStateAction
from environment import Env


class FVMCAgent(MCAgent):
    def __init__(self, actions):
        super(FVMCAgent, self).__init__(actions)

    # for every episode, agent updates q function of visited state action pairs
    def mc(self):
        all_state_actions = super(FVMCAgent, self).preprocess_visited_states()
        visit_state_action = []
        for state_action in all_state_actions:
            if state_action[0] not in visit_state_action:
                visit_state_action.append(state_action[0])
                self.update_global_q_value_table(state_action[0], state_action[1])

    # update visited states for first visit or every visit MC
    def update_global_q_value_table(self, state_action_name, G_t):
        updated = False
        if state_action_name in self.q_value_table:
            state_action = self.q_value_table[state_action_name]
            state_action.total_G = state_action.total_G + G_t
            state_action.N = state_action.N + 1
            state_action.Q = state_action.total_G / state_action.N
            updated = True
        if not updated:
            self.q_value_table[state_action_name] = VisitStateAction(total_G=G_t, N=1, Q=G_t)


# main loop
if __name__ == "__main__":
    env = Env()
    agent = FVMCAgent(actions=list(range(env.n_actions)))
    agent.mainloop(env, verbose=False)