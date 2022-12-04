import gym


if __name__ == '__main__':
    env = gym.make("LunarLander-v2", render_mode="human")
    n_games = 100

    for i in range(n_games):
        observation, obs_info = env.reset()
        terminated = False
        score = 0
        while not terminated:
            action = env.action_space.sample()
            next_observation, reward, terminated, truncated, info = env.step(action)
            score += reward
        print(f"Episode {i} score {score:.1f}")
    env.close()