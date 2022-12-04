import gym
import numpy as np
from actor_critic_torch import Agent
from utils import plot_learning_curve


if __name__ == '__main__':
    env = gym.make("LunarLander-v2")
    agent = Agent(gamma=0.99, lr=5e-6, input_dims=[8], n_actions=4,
                    fc1_dims=2048, fc2_dims=1536)
    n_games = 3000

    fname = 'ACTOR_CRITIC_' + 'lunar_lander_' + str(agent.fc1_dims) + \
            '_fc1_dims_' + str(agent.fc2_dims) + '_fc2_dims_lr' + str(agent.lr) + \
                '_' + str(n_games) + 'games'
    figure_file = 'plots/' + fname + '.png'

    scores = []
    for i in range(n_games):
        terminated = False
        observation, obs_info = env.reset()
        score = 0
        while not terminated:
            action = agent.choose_action(observation)
            next_observation, reward, terminated, truncated, info = env.step(action)
            score += reward
            agent.learn(observation, reward, next_observation, terminated)
            observation = next_observation
        scores.append(score)

        avg_score = np.mean(scores[-100:])
        print(f"Episode {i} score {score:.1f} average score {avg_score:.1f}")

    x = [i+1 for i in range(n_games)]
    plot_learning_curve(x, scores, figure_file)