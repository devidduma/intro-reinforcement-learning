from mc_agent import MCAgent
from environment import Env

class FVMCAgent (MCAgent):
    def __init__(self, actions):
        super(FVMCAgent, self).__init__(actions)

    # for every episode, agent updates v function of visited states
    def update(self):
        all_states = super(FVMCAgent, self).update()
        self.first_visit_mc(all_states)

    def first_visit_mc(self, all_states):
        visit_state = []
        for state in all_states:
            if state[0] not in visit_state:
                visit_state.append(state[0])
                self.update_global_value_table(state[0], state[1])


# main loop
if __name__ == "__main__":
    env = Env()
    agent = FVMCAgent(actions=list(range(env.n_actions)))

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
