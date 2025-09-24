import numpy as np
import time
import collections
from basic_env.grid_env import grid_env
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import matplotlib.pyplot as plt
class Qnet(nn.Module):
    def __init__(self):
        super(Qnet, self).__init__()
        self.linear1 = nn.Linear(2, 100)
        self.linear2 = nn.Linear(100, 5)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x
class DQN:
    def __init__(self, env, device, alpha=1e-3, gamma=0.9, buffer_size=10000, batch_size=100, update_frequency=10):
        self.env = env
        self.device = device
        self.alpha = alpha
        self.gamma = gamma
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.update_frequency = update_frequency
        self.replay_buffer = ReplayBuffer(buffer_size)
        self.main_net = Qnet().to(device)
        self.target_net = Qnet().to(device)
        self.optimizer = torch.optim.Adam(self.main_net.parameters(), lr=self.alpha)
        self.actor_policy = np.ones((env.grid_size[0], env.grid_size[1], env.action_space_size)) / env.action_space_size

    def get_data(self, episode_num=1, episode_length=1000):
        num = 0
        while num < episode_num:
            i = np.random.randint(0, self.env.grid_size[0])
            j = np.random.randint(0, self.env.grid_size[1])
            for _ in range(episode_length):
                state = [i,j]
                action_probs = self.actor_policy[i, j]
                action = np.random.choice(len(action_probs), p=action_probs)
                next_state, reward = self.env.get_next_agent_state_and_reward([i, j], action)
                self.replay_buffer.append({"state": state, "action": action, "reward": reward, "next_state": next_state})
                i, j = next_state
            num += 1

    def learning(self):
        self.get_data()
        td_error = []
        for rounds in range(10000):
            x = torch.tensor(self.replay_buffer.sample(self.batch_size), dtype=torch.float32).to(self.device)
            state = x[:, 0:2]
            action = x[:, 2].long().view(-1, 1)
            reward = x[:, 3].unsqueeze(1)
            next_state = x[:, 4:6]

            with torch.no_grad():
                max_next_q = self.target_net(next_state).max(dim=1)[0].unsqueeze(1)
                td_target = reward + self.gamma * max_next_q
                td_target=(td_target - td_target.mean()) / (td_target.std() + 1e-8)

            q_hat_all = self.main_net(state)
            q_hat = q_hat_all.gather(1, action)

            loss = F.mse_loss(q_hat, td_target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            td_error.append(loss.item())

            if rounds % self.update_frequency == 0:
                self.target_net.load_state_dict(self.main_net.state_dict())
        return td_error

    def get_policy(self):
        policy = np.zeros((self.env.grid_size[0], self.env.grid_size[1], self.env.action_space_size))
        with torch.no_grad():
            for i in range(self.env.grid_size[0]):
                for j in range(self.env.grid_size[1]):
                    state = torch.tensor([[i , j ]],
                                         dtype=torch.float32).to(self.device)
                    q_values = self.target_net(state)
                    max_q = torch.max(q_values).item()
                    max_actions = (q_values == max_q).squeeze().nonzero(as_tuple=True)[0]
                    for a in max_actions:
                        policy[i, j, a.item()] = 1.0 / len(max_actions)
        return policy



class ReplayBuffer:
    def __init__(self,buffer_size):
        self.buffer=collections.deque(maxlen=buffer_size)

    def append(self,info):
        x,y=info["state"]
        action=info["action"]
        reward=info["reward"]
        x_prime,y_prime=info["next_state"]
        self.buffer.append([x,y,action,reward,x_prime,y_prime])
    def sample(self,n):
        samples = random.sample(self.buffer, n)
        return np.array(samples, dtype=np.float32)
    @property
    def size(self):
        return len(self.buffer)

def smooth(y, weight=0.9):
    smoothed = []
    last = y[0]
    for val in y:
        smoothed_val = last * weight + (1 - weight) * val
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed

def plot_loss(loss_list, algorithm_name="DQN", x_label="Episode index",
              y_label="TD error"):
    """
    :param loss_list: 长度为 n 的 list，存储每一轮次的 td loss 值
    :param algorithm_name: 算法名称，用于图例展示，默认 "DQN"
    :param x_label: x 轴标签，默认 "Episode index"
    :param y_label: y 轴标签，默认 "TD LOSS"
    """
    episode_indices = range(1, len(loss_list) + 1)
    plt.plot(episode_indices, loss_list, label=f"{algorithm_name}")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.show()



env = grid_env(grid_size=[5, 5], target=[3, 2], forbidden=[[1, 1], [1, 2], [2, 2], [3, 1],[3,3],[4,1]], start=[0, 0])
start = time.time()
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
agent = DQN(env,device)
total_error=agent.learning()
end = time.time()
converge_time = end - start
print("converge_time: %.2f" % converge_time)
plot_loss(smooth(total_error))
policy=agent.get_policy()
env.visualize_best_actions(policy)
###能够收敛，但是收敛需要的轮次很多，书中的收敛只需要500轮，但是这里至少需要1e5次


