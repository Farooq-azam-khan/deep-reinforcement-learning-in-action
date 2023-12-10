import torch as pt
import random
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt


class ContextBandit:
    def __init__(self, state: int = 10, arms: int = 10):
        self.arms = arms
        self.num_states: int = state
        self.bandit_matrix = ContextBandit.init_distribution(arms)
        self.state = self.update_state()

    def reward(self, prob):
        reward = 0
        for i in range(self.arms):
            if random.random() < prob:
                reward += 1
        return reward

    def get_reward(self, arm: int):
        return self.reward(prob=self.bandit_matrix[self.state][arm])

    def choose_arm(self, arm: int):
        reward = self.get_reward(arm=arm)
        self.state = self.update_state()
        return reward

    def update_state(self):
        # the number of state = number of arms (for simplicity)
        # in general, the state space >> number of actions
        return pt.randint(low=0, high=self.num_states, size=(1,))[0]

    @classmethod
    def init_distribution(cls, arms):
        return pt.rand((arms, arms))

    def one_hot_encoder(self, state: int):
        return pt.nn.functional.one_hot(state, num_classes=self.num_states).float()


def test_successfully_created_env():
    env = ContextBandit()
    _ = env.state
    _ = env.choose_arm(1)
    assert True


E_GREEDY = "e-greedy"
SOFTMAX_SELECTION = "softmax-selection"


def softmax(vals, tau=1.12):
    return pt.nn.Softmax(vals / tau).dim


def train(
    env: ContextBandit, model, loss_fn, optimizer, selection_policy=SOFTMAX_SELECTION
):
    epochs = 500
    rewards = np.zeros(epochs)

    for i in tqdm(range(epochs)):
        if selection_policy == E_GREEDY:
            raise NotImplementedError("needs to be implemented")
        elif selection_policy == SOFTMAX_SELECTION:
            try:
                model_actions = model(env.one_hot_encoder(env.state))
                action_probs = softmax(model_actions, tau=2.0).detach().numpy()
                action_probs /= action_probs.sum()
                choice = np.random.choice(np.arange(env.arms), p=action_probs)
            except Exception as e: 
                print(action_probs)
                raise e 

        reward = env.choose_arm(choice)
        rewards_one_hot = pt.zeros(env.arms)
        rewards_one_hot[choice] = reward

        # update the model weights
        rewards[i] = ((i + 1) * rewards[i - 1] + reward) / (i + 2)
        loss = loss_fn(model_actions, rewards_one_hot)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return np.array(rewards)




def plot_agent_performance(rewards):
    fig, ax = plt.subplots(1, 1)
    ax.set_xlabel("Plays")
    ax.set_ylabel("Avg Reard")
    ax.scatter(np.arange(len(rewards)), rewards)
    plt.show()


"""
state (one-hot) -> nn (w/relu) -> action (one-hot)
"""
if __name__ == "__main__":
    state, arms = 10, 10
    N, D_in, H, D_out = 1, state, 100, arms
    model = pt.nn.Sequential(
        pt.nn.Linear(D_in, H), pt.nn.ReLU(), pt.nn.Linear(H, D_out), pt.nn.ReLU()
    )
    # use the reward given to calculate the loss
    env = ContextBandit(state, arms)
    learning_rate = 1e-2
    optimizer = pt.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = pt.nn.MSELoss()
    print("starting to train...")
    rewards = train(env, model, loss_fn, optimizer)
    plot_agent_performance(rewards)
