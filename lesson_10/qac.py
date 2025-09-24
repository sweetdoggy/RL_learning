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


class PolicyNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim):
        super(PolicyNet, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x


class ValueNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim):
        super(ValueNet, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x


class QAC:
    def __init__(self, env, device, episode_length=100, gamma=0.9, policy_lr=1e-2, value_lr=1e-2):
        self.env = env
        self.device = device
        self.gamma = gamma
        self.policy_lr = policy_lr
        self.value_lr = value_lr
        self.policy_net = PolicyNet(2, 100, 5).to(self.device)
        self.value_net = ValueNet(2, 100, 5).to(self.device)
        self.policy_optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=policy_lr)
        self.value_optimizer = torch.optim.Adam(self.value_net.parameters(), lr=value_lr)
        self.episode_length = episode_length
        self.episode = []
        self.episode_list = []

    def update(self, transition_dict):
        state = torch.tensor(transition_dict["state"], dtype=torch.float32).to(self.device)
        action = torch.tensor(transition_dict["action"]).unsqueeze(1).to(self.device)
        reward = torch.tensor(transition_dict["reward"], dtype=torch.float32).unsqueeze(1).to(self.device)
        next_state = torch.tensor(transition_dict["next_state"], dtype=torch.float32).to(self.device)
        next_action = torch.tensor(transition_dict["next_action"]).unsqueeze(1).to(self.device)
        with torch.no_grad():
            q_next_states = self.value_net(next_state)
            q_next_state = q_next_states.gather(1, next_action)
            td_target = reward + self.gamma * q_next_state
        qsa_all = self.value_net(state)
        qsa = qsa_all.gather(1, action)
        td_error = F.mse_loss(td_target, qsa)
        logits = self.policy_net(state)
        action_probs = Categorical(logits=logits).probs
        action_prob = action_probs.gather(1, action)
        entropy = Categorical(logits=logits).entropy().unsqueeze(1)  # shape: [B, 1]
        policy_loss = torch.mean(torch.log(action_prob) * (-qsa.detach()) - 0.01 * entropy)
        self.value_optimizer.zero_grad()
        self.policy_optimizer.zero_grad()
        td_error.backward()
        policy_loss.backward()
        self.value_optimizer.step()
        self.policy_optimizer.step()
        return {"td_error": td_error,
                "policy_loss": policy_loss,
                "reward": reward}

    def sample(self):
        state = torch.tensor([np.random.randint(0, self.env.grid_size[0]), np.random.randint(0, self.env.grid_size[1])],
                             dtype=torch.float32).to(self.device)
        logits = self.policy_net(state)
        action_probs = Categorical(logits=logits)
        action = action_probs.sample().item()
        steps = 0
        state_list = deque(maxlen=self.episode_length)
        action_list = deque(maxlen=self.episode_length)
        reward_list = deque(maxlen=self.episode_length)
        next_state_list = deque(maxlen=self.episode_length)
        next_action_list = deque(maxlen=self.episode_length)
        while steps < self.episode_length:
            next_state, reward = self.env.get_next_agent_state_and_reward([state[0], state[1]], action)
            next_state = torch.tensor(next_state, dtype=torch.float32).to(self.device)
            next_logits = self.policy_net(next_state)
            next_action_probs = Categorical(logits=next_logits)
            next_action = next_action_probs.sample().item()
            state_list.append(state.detach().cpu())
            action_list.append(action)
            reward_list.append(reward)
            next_state_list.append(next_state.detach().cpu())
            next_action_list.append(next_action)
            state = next_state
            action = next_action
            steps += 1
        return {"state": np.array(state_list),
                "action": np.array(action_list),
                "reward": np.array(reward_list),
                "next_state": np.array(next_state_list),
                "next_action": np.array(next_action_list)}

    def get_tabular_policy(self):
        x = torch.arange(self.env.grid_size[0], dtype=torch.float32)
        y = torch.arange(self.env.grid_size[1], dtype=torch.float32)
        xx, yy = torch.meshgrid(x, y, indexing='ij')
        grid = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1).to(self.device)
        with torch.no_grad():
            policy = F.softmax(self.policy_net(grid).reshape(self.env.grid_size[0], self.env.grid_size[1],
                                                             self.env.action_space_size), dim=2)
        return policy.detach().cpu().numpy()

    def get_tabular_q_value(self):
        x = torch.arange(self.env.grid_size[0], dtype=torch.float32)
        y = torch.arange(self.env.grid_size[1], dtype=torch.float32)
        xx, yy = torch.meshgrid(x, y, indexing='xy')
        grid = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1).to(self.device)
        with torch.no_grad():
            qsa = self.value_net(grid).reshape(self.env.grid_size[0], self.env.grid_size[1],
                                               self.env.action_space_size)
        return qsa.detach().cpu().numpy()

    def get_value_state(self, policy, q):
        return np.sum(policy * q, axis=2)


env = grid_env(grid_size=[5, 5], target=[3, 2], forbidden=[[1, 1], [1, 2], [2, 2], [3, 1], [3, 3], [4, 1]],
               start=[0, 0])
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
agent = QAC(env, device)


def train(alg, episodes, iterations):
    return_list = []
    for i in range(iterations):
        with tqdm(total=int(episodes / iterations), desc='Iteration %d' % (i + 1)) as pbar:
            for i_episode in range(int(episodes / iterations)):
                transition_dict = alg.sample()
                info = alg.update(transition_dict)
                reward = info["reward"].detach().cpu()
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
train(agent, 2000, 10)
policy = agent.get_tabular_policy()
q_values = agent.get_tabular_q_value()
value_states = agent.get_value_state(policy, q_values)
agent.env.visualize_value_function(value_states)
env.visualize_best_actions(policy)
end = time.time()
converge_time = end - start
print("converge_time: %.2f" % converge_time)
