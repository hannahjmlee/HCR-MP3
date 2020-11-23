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




# OLD MODEL MAIN --------------------------------------------------------------------------------
## AGENT TRAINING ------------------------------------------------------------------------------
#batch_size = 100 # how many episodes to run in a single batch
#session_size = 100 # how many training epochs. each epoch runs one batch
#percentile = 80 # used to determine elite reward threshold
#hidden_size = 200
#learning_rate = 0.0025
#completion_score = 200 # average reward over 100 episodes to be considered solved.
#
## initialize learning environment
#env = gym.make("LunarLander-v2")
#n_states = env.observation_space.shape[0]
#n_actions = env.action_space.n
#
##neural network
#net = Net(n_states, hidden_size, n_actions)
##loss function - carries out both softmax activation and cross entropy loss in one
#objective = nn.CrossEntropyLoss()
##optimisation function
#optimizer = optim.Adam(params=net.parameters(), lr=learning_rate)
#
#for i in range(session_size):
#    #generate batch of episode data
#    batch_states,batch_actions,batch_rewards = generate_batch(env, batch_size, t_max=5000)
#
#    # filter out bad episodes - keep elite ones
#    elite_states, elite_actions = filter_batch(batch_states,batch_actions,batch_rewards,percentile)
#
#    # pass elite episodes through neural network
#    optimizer.zero_grad()
#
#    tensor_states = torch.FloatTensor(elite_states)
#    tensor_actions = torch.LongTensor(elite_actions)
#
#    action_scores_v = net(tensor_states)
#    loss_v = objective(action_scores_v, tensor_actions)
#    loss_v.backward()
#    optimizer.step()
#
#    #show results
#    mean_reward, threshold = np.mean(batch_rewards), np.percentile(batch_rewards, percentile)
#    print("%d: loss=%.3f, reward_mean=%.1f, reward_threshold=%.1f" % (
#            i, loss_v.item(), mean_reward, threshold))
#
#    #check if
#    if np.mean(batch_rewards)> completion_score:
#        print("Environment has been successfullly completed!")
#
##env = gym.wrappers.Monitor(gym.make("LunarLander-v2"), directory="videos", force=True)
##generate_batch(env, 1, t_max=5000)
##env.close()
