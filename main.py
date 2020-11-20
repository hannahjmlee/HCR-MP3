import gym
import sys
import os
import time
import numpy as np
from collections import deque
from argparse import ArgumentParser
from decision_rule import DecisionRule
from IPython import display
import matplotlib.pyplot as plt


# ------------------------- Code for keyboard agent ------------------------- #
# from https://github.com/openai/gym/blob/master/examples/agents/keyboard_agent.py

# Keyboard controls:
# w - Nop
# a - fire right engine
# s - fire main engine
# d - fire left engine

do_user_action = False
user_action = -1

def key_press(k, mod):
    global do_user_action, user_action
    if k == ord('w'):
        user_action = 0
        do_user_action = True
    if k == ord('a'):
        user_action = 3
        do_user_action = True
    if k == ord('s'):
        user_action = 2
        do_user_action = True
    if k == ord('d'):
        user_action = 1
        do_user_action = True

def key_release(k, mod):
    global do_user_action, user_action
    do_user_action = False
    user_action = -1

# -------------------------------------------------------------------------- #

def main(args):
    env = gym.make('LunarLander-v2')

    # for playing with keyboard
    # enable key presses
    env.render()
    env.unwrapped.viewer.window.on_key_press = key_press
    env.unwrapped.viewer.window.on_key_release = key_release
    global do_user_action, user_action
    for i_episode in range(args.num_episodes):
        state = env.reset()
        total_reward = 0
        decision_rule = DecisionRule()

        while True:
            chosen_action = decision_rule.get_action(state, user_action)
            next_state, reward, done, info = env.step(chosen_action)
    #             if reward != 0:
    #                 print("reward : ", reward)

            total_reward += reward
            env.render()
            if done:
                break
            state = next_state
            time.sleep(0.05)
        print('Episode', i_episode, ': reward =', total_reward)
    env.close()


if __name__ == "__main__":
    parser = ArgumentParser(description='LunarLander-v2 Discrete')
    parser.add_argument('--num_episodes', type=int, default = 2000,
                        help='number of episodes for training')
    args = parser.parse_args()
    main(args)


