import numpy as np
import gym
from time import sleep

def simple_policy(state):
    action = 0 if state < 5 else 1
    return action

if __name__ == '__main__':
    '''
    - We can now focus on the control problem
    - Digitizing to handle continuous state spaces
    '''
    visualize_env = False # Set to True to see the environment (very slow for training)
    render_mode = "human" if visualize_env  else None

    env = gym.make("CartPole-v1", render_mode=render_mode)
    alpha = 0.1
    gamma = 0.99
    episodes = 5000
    

    states = np.linspace(-0.2094, 0.2094, 10)
    V = {}
    for state in range(len(states)+1):
        V[state] = 0

    for i in range(episodes):
        observation, obs_info = env.reset()
        terminated = False

        while not terminated:
            if visualize_env:
                env.render()
                sleep(0.03)
            # Syntax : np.digitize(Array, Bin, Right)
            # Return : Return an array of indices of the bins.    
            state = int(np.digitize(observation[2], states))
            action = simple_policy(state)
            next_observation, reward, terminated, truncated, info = env.step(action)
            next_state = int(np.digitize(next_observation[2], states))
            V[state] = V[state] + alpha * (reward + gamma * V[next_state] - V[state]) # Temporal Difference (TD) Learning
            observation = next_observation

    for state in V:
        print(f"State: {state}, Value: {V[state]:.3f}")
            