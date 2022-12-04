import numpy as np


class Agent():
    '''
    Monte Carlo (MC) prediction algorithm (Monte Carlo -> end of episode update)
    - Use experience to estimate value of policy
    - Iterate estimation and improvement
    - Found value of policy with first visit MC prediction
    '''
    def __init__(self, gamma=0.99):
        self.V = {}
        self.sum_space = [i for i in range(4,22)] # 22 is exclusive
        self.dealer_show_card_space = [i+1 for i in range(10)] # Can be everything from 1 to 10. 1 is ace, 10 is 10 or face card
        self.ace_space = [False, True] # Does the agent has a usable ace?
        self.action_space = [0,1] # 0 is stick (don't give me another card), 1 is hit (give me another card)

        self.state_space = []
        self.returns = {}
        self.states_visited = {} # first visit or not
        self.memory = []
        self.gamma = gamma # discount factor

        self.init_vals()

    def init_vals(self):
        for total in self.sum_space:
            for card in self.dealer_show_card_space:
                for ace in self.ace_space:
                    self.V[(total, card, ace)] = 0
                    self.returns[(total, card, ace)] = []
                    self.states_visited[(total, card, ace)] = 0
                    self.state_space.append((total, card, ace))

    def policy(self, state):
        total, _, _ = state
        action = 0 if total >= 20 else 1
        return action

    def update_V(self):
        for idt, (state, _) in enumerate(self.memory):
            G = 0 # total_return
            if self.states_visited[state] == 0:
                self.states_visited[state] += 1
                discount = 1
                for t, (_, reward) in enumerate(self.memory[idt:]):
                    G += reward * discount
                    discount *= self.gamma
                    self.returns[state].append(G)

        for state, _ in self.memory:
            self.V[state] = np.mean(self.returns[state])

        for state in self.state_space:
            self.states_visited[state] = 0

        self.memory = []