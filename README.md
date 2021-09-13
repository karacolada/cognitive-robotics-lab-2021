# Cognitive Robotics Lab: State Space Models for Reinforcement Learning

This repository contains information on relevant background, the provided code and exercises of the cognitive robotics lab on state space models for reinforcement learning.

## Installation
I recommend creating a new environment for this project using [Conda](https://docs.conda.io/projects/conda/en/latest/index.html). Then, clone this environment and change into the base directory by running
```
git clone https://github.com/maltemosbach/cognitive-robotics-lab-2021.git
cd cognitive-robotics-lab-2021
```

After activating the environment created for the cognitive robotics lab, running
```
pip install -e .
```
should install the code in this project and required dependencies.

## Training models
To start a run of a PlaNet agent execute the following line.
```
 python -m lab.run algo=planet env=Pendulum-v0
```

## Exercises
You find an overview of the problems and details on how to solve them in [exercises](/../../issues/5).
