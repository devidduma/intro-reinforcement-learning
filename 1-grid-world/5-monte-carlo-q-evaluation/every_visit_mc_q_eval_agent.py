from mc_q_eval_agent import MCAgent
from environment import Env


class EVMCAgent(MCAgent):
    def __init__(self, actions):
        super(EVMCAgent, self).__init__(actions)

    # for every episode, agent updates q function of visited state action pairs
    def update(self):
        all_state_actions = super(EVMCAgent, self).update()
        self.every_visit_mc(all_state_actions)

    def every_visit_mc(self, all_state_actions):
        for state_action in all_state_actions:
                self.update_global_q_value_table(state_action[0], state_action[1])




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

                agent.decaying_epsilon_counter = agent.decaying_epsilon_counter + 1

                """
                for state_action in agent.q_value_table:
                    print("SA: ",state_action, " N: ", agent.q_value_table[state_action].N, " Q: ", agent.q_value_table[state_action].Q)
                """
                break
