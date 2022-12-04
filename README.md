# Actor-Critic methods for Deep Reinforcement Learning

This repository is basically a copy of the [repo](https://github.com/philtabor/Actor-Critic-Methods-Paper-To-Code) by Phil Tabor. However, in this repository I'm using Python version 3.9.12 and an updated version of OpenAI gym.

## Installation

- ``python -m venv venv ``
- ``source venv/bin/activate`` (if you are using Linux)
- ``pip install --upgrade pip`` (optional)
- ``pip install -r requirements.txt``

If you run into compatibility issues between Pytorch and CUDA, in my case I could resolve that by installing a specific Pytorch version for CUDA 11:

- ``pip install torch==1.7.1+cu110 -f https://download.pytorch.org/whl/torch_stable.html``