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


class NET(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(NET, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class A2C:
    def __init__(self, env, device, episode_length=100, gamma=0.9, policy_lr=1e-3, value_lr=1e-3):
        self.env = env
        self.device = device
        self.gamma = gamma
        self.policy_lr = policy_lr
        self.value_lr = value_lr
        self.policy_net = NET(2, 100, 5).to(self.device)
        self.value_net = NET(2, 100, 1).to(self.device)
        self.policy_optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=policy_lr)
        self.value_optimizer = torch.optim.Adam(self.value_net.parameters(), lr=value_lr)
        self.episode_length = episode_length
        self.episode = []
        self.episode_list = []

    def sample(self):
        state = torch.tensor([np.random.randint(0, self.env.grid_size[0]), np.random.randint(0, self.env.grid_size[1])],
                             dtype=torch.float32).to(self.device)
        steps = 0
        state_list = deque(maxlen=self.episode_length)
        action_list = deque(maxlen=self.episode_length)
        reward_list = deque(maxlen=self.episode_length)
        next_state_list = deque(maxlen=self.episode_length)
        while steps < self.episode_length:
            logits = self.policy_net(state)
            action_probs = Categorical(logits=logits)
            action = action_probs.sample().item()
            next_state, reward = self.env.get_next_agent_state_and_reward([state[0], state[1]], action)
            next_state = torch.tensor(next_state, dtype=torch.float32).to(self.device)
            state_list.append(state.detach().cpu())
            action_list.append(action)
            reward_list.append(reward)
            next_state_list.append(next_state.detach().cpu())
            state = next_state
            steps += 1
        return {"state": np.array(state_list),
                "action": np.array(action_list),
                "reward": np.array(reward_list),
                "next_state": np.array(next_state_list)}

    def update(self, transition_dict):
        state = torch.tensor(transition_dict["state"],dtype=torch.float32).to(self.device)
        action = torch.tensor(transition_dict["action"],dtype=torch.int64).unsqueeze(1).to(self.device)
        reward = torch.tensor(transition_dict["reward"],dtype=torch.float32).unsqueeze(1).to(self.device)
        next_state = torch.tensor(transition_dict["next_state"],dtype=torch.float32).to(self.device)
        with torch.no_grad():
            td_target = reward + self.gamma * self.value_net(next_state)
        action_dist = Categorical(logits=self.policy_net(state))
        action_prob = action_dist.probs.gather(1,action)
        value_state=self.value_net(state)
        advantage = td_target - value_state
        policy_loss = torch.mean(torch.log(action_prob) * (-advantage.detach()))
        value_loss = F.mse_loss(value_state, td_target.detach())
        self.policy_optimizer.zero_grad()
        self.value_optimizer.zero_grad()
        policy_loss.backward()
        value_loss.backward()
        self.value_optimizer.step()
        self.policy_optimizer.step()
        return policy_loss,value_loss,reward

    def get_tabular_policy(self):
        x = torch.arange(self.env.grid_size[0], dtype=torch.float32)
        y = torch.arange(self.env.grid_size[1], dtype=torch.float32)
        xx, yy = torch.meshgrid(x, y, indexing='ij')
        grid = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1).to(self.device)
        with torch.no_grad():
            policy = F.softmax(self.policy_net(grid).reshape(self.env.grid_size[0], self.env.grid_size[1],
                                                             self.env.action_space_size), dim=2)
        return policy.detach().cpu().numpy()

    def get_tabular_value_state(self):
        x = torch.arange(self.env.grid_size[0], dtype=torch.float32)
        y = torch.arange(self.env.grid_size[1], dtype=torch.float32)
        xx, yy = torch.meshgrid(x, y, indexing='xy')  # 'ij' 表示输出维度与输入一致
        grid = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1).to(self.device)
        with torch.no_grad():
            v_s = self.value_net(grid).reshape(self.env.grid_size[0], self.env.grid_size[1])
        return v_s.detach().cpu().numpy()



env = grid_env(grid_size=[5, 5], target=[3, 2], forbidden=[[1, 1], [1, 2], [2, 2], [3, 1], [3, 3], [4, 1]],
               start=[0, 0])
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
agent = A2C(env,device)


def train(alg, episodes, iterations):
    return_list = []
    for i in range(iterations):
        with tqdm(total=int(episodes / iterations), desc='Iteration %d' % (i+1)) as pbar:
            for i_episode in range(int(episodes / iterations)):
                transition_dict = alg.sample()
                policy_loss,value_loss,reward = alg.update(transition_dict)
                reward = reward.detach().cpu()
                episode_return = sum(reward)
                return_list.append(episode_return)
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({
                        'episode':
                            '%d' % (episodes / iterations * i + i_episode + 1),
                        'return':
                            '%.3f' % np.mean(return_list[-10:])
                    })
                pbar.update(1)
    return return_list

start = time.time()
train(agent,5000,50)
policy = agent.get_tabular_policy()
value_states = agent.get_tabular_value_state()
agent.env.visualize_value_function(value_states)
env.visualize_best_actions(policy)
end = time.time()
converge_time = end - start
print("converge_time: %.2f" % converge_time)


