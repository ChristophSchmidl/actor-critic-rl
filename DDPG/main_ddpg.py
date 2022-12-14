import gym
import numpy as np
from ddpg_torch import Agent
from utils import plot_learning_curve


if __name__ == '__main__':
    env = gym.make("LunarLanderContinuous-v2")
    agent = Agent(alpha=0.0001, beta=0.001, input_dims=env.observation_space.shape,
                    tau=0.001, batch_size=64, fc1_dims=400, fc2_dims=300,
                    n_actions=env.action_space.shape[0])
    n_games = 1000
    filename = f"LunarLander_alpha_{agent.alpha}_beta_{agent.beta}_{n_games}_games"
    figure_file = f"plots/{filename}.png"

    best_score = env.reward_range[0]
    score_history = []

    for i in range(n_games):
        observation, obs_info = env.reset()
        done = False
        score = 0
        agent.noise.reset()
        while not done:
            action = agent.choose_action(observation)
            next_observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            agent.remember(observation, action, reward, next_observation, done)
            agent.learn()
            score += reward
            observation = next_observation
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score < best_score:
            best_score = avg_score
            agent.save_models()
        
        print(f"Epsisode {i} score {score:.1f} average score {avg_score:.1f}")
    x = [i+1 for i in range(n_games)]
    plot_learning_curve(x, score_history, figure_file)
    