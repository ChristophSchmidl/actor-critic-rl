import gym
from prediction_blackJack import Agent


if __name__ == '__main__':
    env = gym.make('Blackjack-v1')
    agent = Agent()
    n_epsiodes = 500000

    for i in range(n_epsiodes):
        if i % 50000 == 0:
            print('Episode: {}'.format(i))
        observation, obs_info = env.reset()
        terminated = False
        while not terminated:
            action = agent.policy(observation)
            next_observation, reward, terminated, truncated, info = env.step(action)
            agent.memory.append((observation, reward))
            observation = next_observation
        agent.update_V()
    print(agent.V[(21, 3, True)]) # Test state where the agent is almost certain to win
    print(agent.V[(4, 1, False)]) # Test state where the agent is almost certain to lose