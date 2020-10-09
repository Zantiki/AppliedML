import math
import gym
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

buckets=(3, 3, 6, 6)
num_episodes=600
min_lr=0.05
min_epsilon=0.1
discount=1.0
decay=25
env = gym.make("CartPole-v1")

upper_bounds = [env.observation_space.high[0], 0.5, env.observation_space.high[2], math.radians(50) / 1.]
lower_bounds = [env.observation_space.low[0], -0.5, env.observation_space.low[2], -math.radians(50) / 1.]
Q_table = np.zeros(buckets + (env.action_space.n,))
steps = np.zeros(num_episodes)


def get_epsilon(t):
    return max(min_epsilon, min(1., 1. - math.log10((t + 1) / decay)))


def get_learning_rate(t):
    return max(min_lr, min(1., 1. - math.log10((t + 1) / decay)))


def normalize(action_vector, epsilon):

    total = sum(action_vector)
    new_vector = (1 - epsilon) * action_vector / (total)
    new_vector += epsilon / 2.0
    return new_vector


def get_action( state, e):
    obs = discretize_state(state)
    action_vector = Q_table[obs]
    epsilon = get_epsilon(e)
    action_vector = normalize(action_vector, epsilon)
    return action_vector


def discretize_state(obs):
    discretized = list()
    for i in range(len(obs)):
        scaling = ((obs[i] + abs(lower_bounds[i]))
                   / (upper_bounds[i] - lower_bounds[i]))
        new_obs = int(round((buckets[i] - 1) * scaling))
        new_obs = min(buckets[i] - 1, max(0, new_obs))
        discretized.append(new_obs)
    return tuple(discretized)


def choose_action(state, e):
    if (np.random.random() < get_epsilon(e)):
        return env.action_space.sample()
    else:
        return np.argmax(Q_table[state])


def update_q(state, action, reward, new_state, e):
    Q_table[state][action] += (get_learning_rate(e) *
                                    (reward
                                     + discount * np.max(Q_table[new_state])
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
        if steps[i] > 200:
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