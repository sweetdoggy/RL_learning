import numpy as np
import time
from basic_env.grid_env import grid_env
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.distributions import Categorical
from collections import deque
from tqdm import tqdm


class Model(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))

    def predict(self, obs):
        action_dist = Categorical(logits=self.forward(obs))
        action = action_dist.sample()
        return action


class REINFORCE:
    def __init__(self, model, device, gamma=0.9, lr=1e-3):
        self.device = device
        self.model = model.to(self.device)
        self.gamma = gamma
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=lr)

    def update(self, state, action, reward, ):
        g = self.cal_reward_to_g(reward)
        batch_obs = torch.tensor(state, dtype=torch.float32).to(self.device)
        batch_action = torch.tensor(action, dtype=torch.int64).view(-1, 1).to(self.device)
        batch_g = torch.tensor(g, dtype=torch.float32).to(self.device).view(-1, 1)
        batch_g = (batch_g - batch_g.mean()) / (batch_g.std() + 1e-8)
        logits = self.model(batch_obs)
        log_pi = F.log_softmax(logits, dim=-1).gather(1, batch_action)
        loss = torch.mean(-log_pi * batch_g)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def cal_reward_to_g(self, reward):
        g = []
        q = 0
        for r in reversed(reward):
            q = self.gamma * q + r
            g.append(q)
        g.reverse()
        return np.array(g)


def smooth(y, weight=0.9):
    smoothed = []
    last = y[0]
    for val in y:
        smoothed_val = last * weight + (1 - weight) * val
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed


def plot_loss(loss_list, algorithm_name="REINFORCE", x_label="Episode index",
              y_label="-log_pi * q(s,a)"):
    """
    :param loss_list: 长度为 n 的 list，存储每一轮次的 td loss 值
    :param algorithm_name: 算法名称，用于图例展示，默认 "REINFORCE"
    :param x_label: x 轴标签，默认 "Episode index"
    :param y_label: y 轴标签，默认 "-log_pi * q(s,a)"
    """
    episode_indices = range(1, len(loss_list) + 1)
    plt.plot(episode_indices, loss_list, label=f"{algorithm_name}")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.show()


def run_episode(env, alg, device, max_len=100):
    state_queue = deque(maxlen=max_len)
    action_queue = deque(maxlen=max_len)
    reward_queue = deque(maxlen=max_len)
    state = torch.tensor(env.reset(), dtype=torch.float32).to(device)
    i = 0
    while i < max_len:
        action = alg.model.predict(state).item()
        next_state, reward = env.get_next_agent_state_and_reward(state, action)
        state_queue.append(state.detach().cpu())
        action_queue.append(action)
        reward_queue.append(reward)
        state = torch.tensor(next_state).to(device)
        i += 1
    return np.array(state_queue), np.array(action_queue), np.array(reward_queue)


def train(env, alg, episodes, iterations):
    return_list = []
    for i in range(iterations):
        with tqdm(total=int(episodes / iterations), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(episodes / iterations)):
                state, action, reward = run_episode(env, alg, device)
                episode_return = sum(reward)
                return_list.append(episode_return)
                alg.update(state, action, reward)
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({
                        'episode':
                            '%d' % (episodes / iterations * i + i_episode + 1),
                        'return':
                            '%.3f' % np.mean(return_list[-10:])
                    })
                pbar.update(1)
    return return_list


def get_tabular_policy(device, env, policy_net):
    x = torch.arange(env.grid_size[0], dtype=torch.float32)
    y = torch.arange(env.grid_size[1], dtype=torch.float32)
    xx, yy = torch.meshgrid(x, y, indexing='ij')  # 'ij' 表示输出维度与输入一致
    # 重塑并拼接为 [N, 2] 的张量
    grid = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1).to(device)
    with torch.no_grad():
        policy = F.softmax(policy_net(grid).reshape(env.grid_size[0], env.grid_size[1], env.action_space_size), dim=2)
    return policy.detach().cpu().numpy()


env = grid_env(grid_size=[5, 5], target=[3, 2], forbidden=[[1, 1], [1, 2], [2, 2], [3, 1], [3, 3], [4, 1]],
               start=[0, 0])
start = time.time()
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model = Model(2, 64, 5)
alg = REINFORCE(model, device)
return_list = train(env, alg, 2000, 20)
policy = get_tabular_policy(device, env, alg.model)
end = time.time()
converge_time = end - start
print("converge_time: %.2f" % converge_time)
env.visualize_best_actions(policy)
