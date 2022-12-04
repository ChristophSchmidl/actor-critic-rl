# Actor-Critic methods for Deep Reinforcement Learning

This repository is basically a copy of the [repo](https://github.com/philtabor/Actor-Critic-Methods-Paper-To-Code) by Phil Tabor. However, in this repository I'm using Python version 3.9.12 and an updated version of OpenAI gym.

## Installation

- ``python -m venv venv ``
- ``source venv/bin/activate`` (if you are using Linux)
- ``pip install --upgrade pip`` (optional)
- ``pip install -r requirements.txt``

If you run into compatibility issues between Pytorch and CUDA, in my case I could resolve that by installing a specific Pytorch version for CUDA 11:

- ``pip install torch==1.7.1+cu110 -f https://download.pytorch.org/whl/torch_stable.html``

# Papers

- DDPG/Deep Deterministic Policy Gradient : [Continuous control with deep reinforcement learning](https://arxiv.org/abs/1509.02971)
- TD3/Twin Delayed Deep Deterministic Policy Gradient: [Addressing Function Approximation Error in Actor-Critic Methods](https://arxiv.org/pdf/1802.09477)
- SAC/Soft Actor Critic: [Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor](https://arxiv.org/pdf/1801.01290v2.pdf)
- TRPO: [Trust Region Policy Optimization](https://arxiv.org/abs/1502.05477)
- PPO: [Proximal Policy Optimization Algorithms](https://arxiv.org/pdf/1707.06347.pdf)
- A3C/Asynchronous Advantage Actor Critic: [Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/pdf/1602.01783.pdf)
- D4PG/Distributed Distributional DDPG: [Distributed Distributional Deterministic Policy Gradients](https://openreview.net/pdf?id=SyZipzbCb)
- [Benchmarking Deep Reinforcement Learning for Continuous Control](https://arxiv.org/pdf/1604.06778)
- [Setting up a Reinforcement Learning Task with a Real-World Robot](https://arxiv.org/pdf/1803.07067.pdf)

# Websites

- [The idea behind Actor-Critics and how A2C and A3C improve them](https://theaisummer.com/Actor_critics/)







