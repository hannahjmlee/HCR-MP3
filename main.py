import gym
import sys
import os
import time
import numpy as np
from collections import deque, namedtuple
from argparse import ArgumentParser
from IPython import display
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# ------------------------- Code for keyboard agent ------------------------- #
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

class Net(nn.Module):
    def __init__(self, obs_size, hidden_size, n_actions):
        super(Net, self).__init__()
        # first fully connected layer - input layer takes in a size(tensor) == size(state)
        # outputs a tensor that is the size of our hidden nodes (200)
        self.fc1 = nn.Linear(obs_size, hidden_size)
        # second fully connected layer - outputs a tensor that is the size of action space
        self.fc2 = nn.Linear(hidden_size, n_actions)

    def forward(self, x):
        # pass state through first layer of NN and apply relu activation to the output of fc1
        x = F.relu(self.fc1(x))
        return self.fc2(x)

def generate_batch(env,batch_size, net, t_max=5000):
    # make activation function
    activation = nn.Softmax(dim=1)
    # 3 lists store episode data
    # batch actions and bat states are lists where each index stores all of the actions/states for a particular episode.
    # batch rewards stores the total reward achieved during each episode
    batch_actions,batch_states, batch_rewards = [],[],[]

    # iterate through batch size -> run an episode each iteration.
    for b in range(batch_size):
        states,actions = [],[]
        total_reward = 0 # total reward of episode
        s = env.reset() # starts a new game
        # loops through a single step in the game environment until time limit is reached.
        for t in range(t_max):
            # get current state (s) and tun into a torch float tesnro
            s_v = torch.FloatTensor([s])
            act_probs_v = activation(net(s_v))
            act_probs = act_probs_v.data.numpy()[0] # action probability distribution
            # choose from given probabilities
            a = np.random.choice(len(act_probs), p=act_probs)

            # action is carried out by environment
            #returns new state, reward received by action, whether or not episode is finished, and other information(?)
            new_s, r, done, info = env.step(a)

            # record states and actions, update total reward
            states.append(s)
            actions.append(a)
            total_reward += r

            # update current state with new state
            s = new_s

            if done: # if episode has finished during the step
                # record batch states, actions, and rewards.
                batch_actions.append(actions)
                batch_states.append(states)
                batch_rewards.append(total_reward)
                break

    return batch_states, batch_actions, batch_rewards


def filter_batch(states_batch,actions_batch,rewards_batch,percentile=50):
    # selects only the best episodes from the latest batch
    # top 20% of batch using reward as a threshold
    reward_threshold = np.percentile(rewards_batch, percentile)

    elite_states = []
    elite_actions = []


    for i in range(len(rewards_batch)):
        if rewards_batch[i] > reward_threshold:
            for j in range(len(states_batch[i])):
                elite_states.append(states_batch[i][j])
                elite_actions.append(actions_batch[i][j])

    return elite_states,elite_actions


def main(args):
    # TRAINING ------------------------------------------------------------------------------
    batch_size = 100 # how many episodes to run in a single batch
    session_size = 100 #100 # how many training epochs. each epoch runs one batch
    percentile = 80 # used to determine elite reward threshold
    hidden_size = 200
    learning_rate = 0.0025
    completion_score = 200 # average reward over 100 episodes to be considered solved.


    # initialize learning environment
    env = gym.make("LunarLander-v2")
    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.n


    #neural network
    net = Net(n_states, hidden_size, n_actions)
    #loss function - carries out both softmax activation and cross entropy loss in one
    objective = nn.CrossEntropyLoss()
    #optimisation function
    optimizer = optim.Adam(params=net.parameters(), lr=learning_rate)

    f= open("training.txt", "w+")
    f.write("session, loss, reward mean, reward threshold\n")
    for i in range(session_size):
        #generate batch of episode data
        batch_states,batch_actions,batch_rewards = generate_batch(env, batch_size, net, t_max=5000)

        # filter out bad episodes - keep elite ones
        elite_states, elite_actions = filter_batch(batch_states,batch_actions,batch_rewards,percentile)

        # pass elite episodes through neural network
        optimizer.zero_grad()

        tensor_states = torch.FloatTensor(elite_states)
        tensor_actions = torch.LongTensor(elite_actions)

        action_scores_v = net(tensor_states)
        loss_v = objective(action_scores_v, tensor_actions)
        loss_v.backward()
        optimizer.step()

        #show results
        mean_reward, threshold = np.mean(batch_rewards), np.percentile(batch_rewards, percentile)
        print("%d: loss=%.3f, reward_mean=%.1f, reward_threshold=%.1f" % (
                i, loss_v.item(), mean_reward, threshold))
        f.write(str(i))
        f.write(", ")
        f.write(str(loss_v.item()))
        f.write(", ")
        f.write(str(mean_reward))
        f.write(", ")
        f.write(str(threshold))
        f.write("\n")

        #check if
        if np.mean(batch_rewards)> completion_score:
            print("Environment has been successfullly completed!")
    f.close()


    # TESTING ----------------------------------------------------------------------------------
    # for playing with keyboard
    # enable key presses
    print("STARTING GAME")
    env.render()
    env.unwrapped.viewer.window.on_key_press = key_press
    env.unwrapped.viewer.window.on_key_release = key_release
    global do_user_action, user_action
    activation = nn.Softmax(dim=1)
    for i_episode in range(args.num_episodes):
        state = env.reset()
        total_reward = 0

        while True:
            if user_action != -1 and user_action != 0:
                chosen_action = user_action
            else:
                s_v = torch.FloatTensor([state])
                act_probs_v = activation(net(s_v))
                act_probs = act_probs_v.data.numpy()[0]
                #print(act_probs)
                chosen_action = np.argmax(act_probs)
                #print("Model chosen action:", chosen_action)

            next_state, reward, done, info = env.step(chosen_action)

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


