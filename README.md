# LunarLander-v2 (Discrete)
Implementation of keyboard code in the OpenAI environment LunarLander-v2 with a discrete action space.
Link to environment: https://gym.openai.com/envs/LunarLander-v2/

## Installation

The following dependencies need to be installed:

```
pip install gym
pip install Box2D
pip install torch torchvision
pip install ipython
pip install matplotlib
pip install nose
```

## Run

To run the program:

```
python main.py
```

The playing mode allows the user to play using keyboard input (using 'w', 'a', 's', and 'd'). There is a DecisionRule class that can be used to modify the playing mode and how the action is chosen for the environment.

