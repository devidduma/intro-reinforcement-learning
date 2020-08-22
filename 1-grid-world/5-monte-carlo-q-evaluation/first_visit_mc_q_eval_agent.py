from mc_q_eval_agent import MCAgent
from environment import Env

class FVMCAgent (MCAgent):
    def __init__(self, actions):
        super(FVMCAgent, self).__init__(actions)

    # for every episode, agent updates q function of visited state action pairs
    def update(self):
        all_state_actions = super(FVMCAgent, self).update()
        self.first_visit_mc(all_state_actions)

    def first_visit_mc(self, all_state_actions):
        visit_state_action = []
        for state_action in all_state_actions:
            if state_action[0] not in visit_state_action:
                visit_state_action.append(state_action[0])
                self.update_global_q_value_table(state_action[0], state_action[1])


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

            agent.save_sample(state, action, reward, done)

            # update state
            state = next_state
            # get next action
            action = agent.get_action(next_state)


            # at the end of each episode, update the q function table
            if done:
                print("episode : ", episode)
                agent.update()
                agent.samples.clear()

                """
                for state_action in agent.q_value_table:
                    print("SA: ",state_action, " N: ", agent.q_value_table[state_action].N, " Q: ", agent.q_value_table[state_action].Q)
                """
                break
