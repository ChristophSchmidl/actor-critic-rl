import gym
import matplotlib.pyplot as plt
import numpy as np
from reinforce_torch import PolicyGradaientAgent


def plot_learning_curve(scores, x, figure_file):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
    plt.plot(x, running_avg)
    plt.title('Running average of previous 100 scores')
    plt.savefig(figure_file)

if __name__ == '__main__':
    env = gym.make("LunarLander-v2")
    n_games = 3000
    agent = PolicyGradaientAgent(gamma=0.99, lr=0.0005, input_dims=[8], n_actions=4)

    fname = 'REINFORCE_' + 'lunar_lander_lr' + str(agent.lr) + '_' \
        + str(n_games) + 'games'
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
            agent.store_rewards(reward)
            observation = next_observation
        agent.learn()
        scores.append(score)

        avg_score = np.mean(scores[-100:])
        print(f"Episode {i} score {score:.2f} average score {avg_score:.2f}")

    x = [i+1 for i in range(len(scores))]
    plot_learning_curve(scores, x, figure_file)