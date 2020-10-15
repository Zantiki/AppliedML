import math
import gym
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

buckets=(3, 3, 6, 6)
num_episodes=1000
min_lr=0.01

# Epsilon: Describes the lower bound of exploring, i.e doing something at random
min_epsilon=0.01
# Gamma: discount factor between immediate or future rewards
gamma=0.99
# Number at which we adjust the epsilon/learning_rate-convergence towards min_lr/min_epsilon
decay=25
env = gym.make("CartPole-v0")
# Used to scale discrete states properly
upper_bounds = [env.observation_space.high[0], 0.5, env.observation_space.high[2], math.radians(50) / 1.]
lower_bounds = [env.observation_space.low[0], -0.5, env.observation_space.low[2], -math.radians(50) / 1.]

Q_table = np.zeros(buckets + (env.action_space.n,))
steps = np.zeros(num_episodes)


def get_epsilon(t):
    """
    Balance out randomness as the number of episodes get higher
    :param t: the episode
    :return:
    """
    return max(min_epsilon, min(1., 1. - math.log10((t + 1) / decay)))


def get_learning_rate(t):
    """
    Balance out the learning rate as the number of episodes get higher
    :param t: the episode
    :return:
    """
    return max(min_lr, min(1., 1. - math.log10((t + 1) / decay)))


def discretize_state(obs):
    """
    Discretize the contionous possible states of the cartpole to approximate by q-value.
    We do this by adding the otherwise tiny continous values as a range described by the bucket
    :param obs:
    :return:
    """
    discretized = list()
    for i in range(len(obs)):
        scaling = ((obs[i] + abs(lower_bounds[i]))/(upper_bounds[i] - lower_bounds[i]))
        new_obs = int(round((buckets[i] - 1) * scaling))
        new_obs = min(buckets[i] - 1, max(0, new_obs))
        discretized.append(new_obs)
    return tuple(discretized)


def choose_action(state, e):
    """Select action from Q if epsilon is high enough"""
    if (np.random.random() < get_epsilon(e)):
        return env.action_space.sample()
    else:
        return np.argmax(Q_table[state])


def update_q(state, action, reward, new_state, e):
    """
    Update q-table by relevant formula.
    :param state:
    :param action:
    :param reward:
    :param new_state:
    :param e:
    :return:
    """
    Q_table[state][action] += (get_learning_rate(e) *
                                    (reward
                                     + gamma * np.max(Q_table[new_state])
                                     - Q_table[state][action]))

def train():
    for e in range(num_episodes):
        # Initializes the state
        current_state = discretize_state(env.reset())

        done = False
        # Looping for each step
        while not done:
            steps[e] += 1
            # Choose A from S
            action = choose_action(current_state, e)
            # Take action
            obs, reward, done, _ = env.step(action)
            new_state = discretize_state(obs)
            # Update Q(S,A)
            update_q(current_state, action, reward, new_state, e)
            current_state = new_state


def plot_learning():
    sns.lineplot(range(len(steps)),steps)
    plt.xlabel("Episode")
    plt.ylabel("Steps")
    plt.show()
    t = 0
    for i in range(num_episodes):
        if steps[i] > 195:
            t+=1
    print(t, "episodes were successfully completed.")


if __name__ == "__main__":
    train()
    plot_learning()
    t = 0
    done = False
    current_state = discretize_state(env.reset())

    for i in range(100):
        current_score = 0
        env.reset()
        done = False
        while not done:
            # env.render()
            t = t + 1
            current_score = current_score + 1
            action = choose_action(current_state, 500)
            obs, reward, done, _ = env.step(action)
            new_state = discretize_state(obs)
            current_state = new_state
        print(i, ":", current_score)
    print("AVG SCORE: {}".format(t/100))