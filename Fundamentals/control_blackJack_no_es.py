import numpy as np


class Agent():
    def __init__(self, epsilon=0.1, gamma=0.99):
        self.Q = {}
        self.sum_space = [i for i in range(4, 22)]  # 22 is exclusive
        self.dealer_show_card_space = [i + 1 for i in range(10)]  # Can be everything from 1 to 10. 1 is ace, 10 is 10 or face card
        self.ace_space = [False, True]  # Does the agent has a usable ace?
        self.action_space = [0, 1]  # 0 is stick (don't give me another card), 1 is hit (give me another card)

        self.state_space = []
        self.memory = []
        self.pairs_visited = {}
        self.returns = {}

        self.gamma = gamma
        self.epsilon = epsilon

        self.init_vals()
        self.init_policy()

    def init_vals(self):
        for total in self.sum_space:
            for card in self.dealer_show_card_space:
                for ace in self.ace_space:
                    state = (total, card, ace)
                    self.state_space.append(state)
                    for action in self.action_space:
                        self.Q[(state, action)] = 0
                        self.returns[(state, action)] = []
                        self.pairs_visited[(state, action)] = 0

    def init_policy(self):
        policy = {} # temorary policy
        n_actions = len(self.action_space)

        for state in self.state_space:
            policy[state] = np.ones(n_actions) / n_actions
        self.policy = policy

    def choose_action(self, state):
        action = np.random.choice(self.action_space, p=self.policy[state])
        return action

    def update_Q(self):
        # calculate discounted future returns for each first visit to each state
        for idt, (state, action, _) in enumerate(self.memory):
            G = 0
            discount = 1
            if self.pairs_visited[(state, action)] == 0:
                self.pairs_visited[(state, action)] += 1
                for t, (_, _, reward) in enumerate(self.memory[idt:]):
                    G += reward * discount
                    discount *= self.gamma
                    self.returns[(state, action)].append(G)

        # Calculate the mean for the returns and update the policy for that state
        for state, action, _ in self.memory:
            self.Q[(state, action)] = np.mean(self.returns[(state, action)])
            self.update_policy(state)

        # Reset the memory and the pairs visited
        for state_action in self.pairs_visited.keys():
            self.pairs_visited[state_action] = 0
        
        self.memory = []

    def update_policy(self, state):
        actions = [self.Q[(state, a)] for a in self.action_space]
        a_max = np.argmax(actions)
        n_actions = len(self.action_space)
        probs = []
        for action in self.action_space:
            prob = 1 - self.epsilon + self.epsilon / n_actions if action == a_max else \
                self.epsilon / n_actions 
            probs.append(prob)
        self.policy[state] = probs

        