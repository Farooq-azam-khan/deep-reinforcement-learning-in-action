import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import random

import os

"""
n slot machines (i.e. n actions) 
-> each slot machine has one action (pulling the lever) 
at each step you can pull one of the 10 leavers (step k) 
you take action a and recieve reward R_k (i.e. the reward at step k)

behind the scenes: 
    each slot machine has a probability distribution that you are not aware of
    based on that distribution you get the reaward

"""


# Q_k(a) = sum(R_1 + ... + R_k) / k
# i.e. the mean of the rewards for taking action a
def expected_reward(action: int, history: list[list[float]]):
    all_rewards_for_a = history[action]
    return np.sum(all_rewards_for_a) / len(all_rewards_for_a)


# game environment
def create_environment(num_slots: int):
    probs = np.random.rand(num_slots)
    return probs


# stepping in the envornmnet
def get_reward(prob: float, max_dollar_win: int = 10):  # n=number of actions
    # there is a prob change you will win 10$
    reward = 0
    for i in range(max_dollar_win):
        if random.random() < prob:
            reward += 1
    return reward  # if prob == 0.7, on avg you will win 7$ if max_dollary_win = 10


# epsilon greedy - probablity of it doing a random action
# i.e. what percentage should it explore?
epsilon = 0.4

# print(np.mean([get_reward(0.7) for _ in range(1_000)]))

""" udated average fuction 
mu_new = (k * mu_old + x) / (k+1) 
"""

k_idx = 0
reward_idx = 1

"""
exploration vs exploitation 
exploration: select action randomly and observe the reward 
exploitation: choosing the best action that will maximize reward 
"""


def get_best_action(record) -> int:
    return np.argmax(record[:, reward_idx], axis=0)


def test_best_action():
    record_test = np.array([[1, 10]])
    assert get_best_action(record_test) == 0


def update_record(record, action, reward: float):
    k = record[action, k_idx]
    mu_old = record[action, reward_idx]
    new_reward = (k * mu_old + reward) / (k + 1)  # mu_new
    record[action, reward_idx] = new_reward
    record[action, k_idx] += 1
    return record


def plot_agent_performance(rewards):
    fig, ax = plt.subplots(1, 1)
    ax.set_xlabel("Plays")
    ax.set_ylabel("Avg Reard")
    ax.scatter(np.arange(len(rewards)), rewards)
    plt.show()


E_GREEDY = "e-greedy"
SOFTMAX_SELECTION = "softmax-selection"


def train_agent():
    selection_policy = SOFTMAX_SELECTION
    n = 10
    probs = create_environment(n)
    record = np.zeros(
        (n, 2)
    )  # running average, number of times action a has been observed
    rewards = [0]  # graphing the mean reward
    for i in tqdm(range(500)):
        if selection_policy == E_GREEDY:
            if (
                random.random() > epsilon
            ):  # with 80% probability, choose the best action
                choice = get_best_action(record)
            else:  # with 20% prob, choose a random action
                choice = np.random.randint(n)
        elif selection_policy == SOFTMAX_SELECTION:
            action_list = np.arange(n)
            action_probs = softmax(record[:, reward_idx])
            choice = np.random.choice(action_list, p=action_probs)

        r = get_reward(probs[choice])  # stepping in the env
        record = update_record(record, choice, r)
        mean_reward = ((i + 1) * rewards[-1] + r) / (i + 2)  # for graphing
        rewards.append(mean_reward)
    return probs, record, rewards


PLOT = os.environ.get("PLOT", 0) == "1"
print(f"{PLOT=}")

""" SOFTMAX SELECTION POLICY 
gives us a probability distribution across our actions
this means that the action with the higest reward is more likely to be chosen on avg.
we will also know the second and thrid best action, etc. 
tau = temperature and it exagerate differences in probabilities, i.e. 
small tau = small differences, high tau = large differences 
"""


def softmax(vals, tau=1.12):
    return np.exp(vals / tau) / np.sum(np.exp(vals / tau))


if __name__ == "__main__":
    probs, record, mean_reward = train_agent()
    if np.argmax(probs) == np.argmax(record[:, reward_idx]):
        print("found best slot machine")
    else:
        print("did not find best slot machine: ", np.argmax(probs))
    if PLOT:
        plot_agent_performance(mean_reward)
