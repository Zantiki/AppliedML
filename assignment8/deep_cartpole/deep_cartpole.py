import gym
import random
import math
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import seaborn as sns
import matplotlib.pyplot as plt
from torch import FloatTensor, LongTensor

# hyper parameters
episodes = 500  # number of episodes
# EPS_START = 0.9  # e-greedy threshold start value
min_epsilon = 0.01  # e-greedy threshold end value
decay = int(225 / 8)  # e-greedy threshold decay
gamma = 0.7  # Q-learning discount factor
lr = 0.0001  # NN optimizer learning rate
hidden = 256  # NN hidden layer size
batch = 64  # Q-learning batch size

epsilons = []

random_amounts = 0
total = 0

class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def push(self, transition):
        self.memory.append(transition)
        if len(self.memory) > self.capacity:
            del self.memory[0]

    def sample(self, batch):
        return random.sample(self.memory, batch)

    def __len__(self):
        return len(self.memory)


class Network(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.l1 = nn.Linear(4, hidden)
        self.l2 = nn.Linear(hidden, 2)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = self.l2(x)
        return x


env = gym.make('CartPole-v0')

model = Network()
memory = ReplayMemory(10000)
optimizer = optim.Adam(model.parameters(), lr)
episode_durations = []

def get_epsilon(t):
    """
    Balance out randomness as the number of episodes get higher
    :param t: the episode
    :return:
    """
    return max(min_epsilon, min(1., 1. - math.log10((t + 1) / decay)))


def select_action(state, e):
    global total
    global random_amounts
    # sample = random.random()
    # eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY
    # return model(Variable(state, volatile=True).type(FloatTensor)).data.max(1)[1].view(1, 1)
    total = total + 1
    if np.random.random() < get_epsilon(e):
        random_amounts = random_amounts + 1
        return LongTensor([[env.action_space.sample()]])
    else:
        # print(model(Variable(state, volatile=True).type(FloatTensor)).data.max(1)[1])
        return model(Variable(state, volatile=True).type(FloatTensor)).data.max(1)[1].view(1, 1)


def run_episode(e, environment, demo=False):
    state = environment.reset()
    steps = 0
    while True:
        # action = select_action(FloatTensor([state]), e)
        action = select_action(FloatTensor([state]), e)
        next_state, reward, done, _ = environment.step(action[0, 0].numpy())
        # negative reward when attempt ends
        if done:
            reward = - 1
            """if steps < 195:
                reward = -1
            else:
                reward += 1"""

        memory.push((FloatTensor([state]),
                     torch.tensor(action),  # action is already a tensor
                     FloatTensor([next_state]),
                     FloatTensor([reward])))
        if not demo:
            learn()

        state = next_state
        steps += 1

        if done:
            print(e, steps)
            epsilons.append(get_epsilon(e)*200)
            episode_durations.append(steps)
            break


def learn():
    if len(memory) < batch:
        return

    # random transition batch is taken from experience replay memory
    transitions = memory.sample(batch)
    batch_state, batch_action, batch_next_state, batch_reward = zip(*transitions)

    batch_state = Variable(torch.cat(batch_state))
    batch_action = Variable(torch.cat(batch_action))
    batch_reward = Variable(torch.cat(batch_reward))
    batch_next_state = Variable(torch.cat(batch_next_state))

    # current Q values are estimated by NN for all actions
    current_q_values = model(batch_state).gather(1, batch_action)
    # expected Q values are estimated from actions which gives maximum Q value
    max_next_q_values = model(batch_next_state).detach().max(1)[0]
    expected_q_values = batch_reward + (gamma * max_next_q_values)

    loss = F.smooth_l1_loss(current_q_values, expected_q_values.reshape(64, 1))

    # backpropagation of loss to NN
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def plot_durations(label_string):
        sns.lineplot(range(len(episode_durations)), episode_durations)
        sns.lineplot(range(len(epsilons)), epsilons)
        plt.xlabel("Episodes - %s" % label_string)
        plt.ylabel("Steps")
        plt.show()

        t = 0
        for i in range(episodes):
            if episode_durations[i] > 195:
                t += 1
        print(t, "episodes were successfully completed.")


for e in range(episodes):
    run_episode(e, env)

plot_durations("Training")

old_episodes = episodes
# min_epsilon = 0
episodes = 100
episode_durations = []
epsilons = []
for e in range(100):
    run_episode(old_episodes, env, demo=True)

plot_durations("Testing")

print('Average Score {}'.format(round(sum(episode_durations)/100)))
# print("{}% of actions were random".format(round((random_amounts*100) / total, 2)))
plt.show()