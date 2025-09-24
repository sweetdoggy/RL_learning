import numpy as np
from basic_env.grid_env import grid_env
import time
import copy


def random_policy(i, j, k):
    psa = np.random.rand(i, j, k)
    sums = np.sum(psa, axis=2, keepdims=True)
    normalized_psa = psa / sums
    return normalized_psa


class n_step_sarsa:
    def __init__(self, env, alpha=1e-2, gamma=0.9, epsilon=0.1, n=5):
        self.env = env
        self.policy = random_policy(env.grid_size[0], env.grid_size[1], env.action_space_size)
        self.qsa = np.ones(shape=(env.grid_size[0], env.grid_size[1], env.action_space_size))
        self.new_policy = None
        self.new_qsa = None
        self.state = None
        self.next_state = None
        self.action = None
        self.next_action = None
        self.trajectory = []
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.n = n
        self.stable_steps = 0
        self.done = False

    def sample(self):
        x, y = self.state
        action_probs = self.policy[x, y]
        action = np.random.choice(len(action_probs), p=action_probs)
        [x_prime, y_prime], reward = self.env.get_next_agent_state_and_reward([x, y], action)
        self.trajectory.append({"state": [x, y], "action": action, "reward": reward})
        self.next_state = [x_prime, y_prime]
        if self.state == self.env.target:
            self.done = True
        self.state=self.next_state

    def policy_evaluation(self):
        new_qsa = copy.deepcopy(self.qsa)
        while len(self.trajectory) <= self.n and self.done == False:
            self.sample()
        x, y = self.trajectory[0]["state"]
        action = self.trajectory[0]["action"]
        self.action=action
        g = 0
        if len(self.trajectory)==1 and self.done is True:
            g=+self.trajectory[0]["reward"]
            (x_prime, y_prime), reward = self.env.get_next_agent_state_and_reward([x, y], action)
            action_prime_probs = self.policy[x_prime, y_prime]
            v = np.dot(action_prime_probs, self.qsa[x_prime, y_prime])
            g+=self.gamma*v
        else:
            for idx in range(len(self.trajectory) - 1):  # 最后一个元素用于 bootstrapping
                reward = self.trajectory[idx]["reward"]
                g += (self.gamma ** idx) * reward
            s_n = self.trajectory[-1]["state"]
            a_n = self.trajectory[-1]["action"]
            g += (self.gamma ** (len(self.trajectory) - 1)) * self.qsa[s_n[0], s_n[1], a_n]
        new_qsa[x, y, action] = self.qsa[x, y, action] - self.alpha * (self.qsa[x, y, action] - g)
        return new_qsa

    def policy_update(self):
        new_policy = copy.deepcopy(self.policy)
        for i in range(self.env.grid_size[0]):
            for j in range(self.env.grid_size[1]):
                q_values = []
                for action in range(self.env.action_space_size):
                    q_values.append(self.qsa[i, j, action])
                best_action = np.argwhere(q_values == np.max(q_values)).flatten()
                new_policy[i, j, :] = self.epsilon / self.env.action_space_size
                new_policy[i, j, best_action] += (1 - self.epsilon) / len(best_action)
        return new_policy

    def learning(self):
        i = np.random.randint(low=0, high=self.env.grid_size[0])
        j = np.random.randint(low=0, high=self.env.grid_size[1])
        self.state = [i, j]
        rounds = 0
        while True:
            self.new_qsa = self.policy_evaluation()
            if np.max(np.abs(self.qsa - self.new_qsa)) < 1e-6:
                self.stable_steps += 1
            else:
                self.stable_steps = 0
            self.qsa = copy.deepcopy(self.new_qsa)
            self.policy = self.policy_update()
            rounds += 1
            self.epsilon *=0.99
            if len(self.trajectory) == 1 and self.done is True:
                i = np.random.randint(low=0, high=self.env.grid_size[0])
                j = np.random.randint(low=0, high=self.env.grid_size[1])
                self.state = [i, j]
                self.done = False
                self.trajectory = []
            else:
                self.trajectory.pop(0)
            if self.stable_steps > 100:
                print(rounds)
                break
        return self.policy

    def calculate_value_function(self):
        V = np.zeros(self.env.grid_size)
        for i in range(self.env.grid_size[0]):
            for j in range(self.env.grid_size[1]):
                q_values = []
                for action in range(self.env.action_space_size):
                    q_values.append(self.qsa[i, j, action])
                V[i, j] = np.dot(self.policy[i, j], self.qsa[i, j])
        return V


env = grid_env(grid_size=[5, 5], target=[3, 2], forbidden=[[1, 1], [2, 4], [3, 3], [4, 1]], start=[0, 0])
start = time.time()
agent = n_step_sarsa(env)
policy = agent.learning()
V = agent.calculate_value_function()
end = time.time()
converge_time = end - start
print("converge_time: %.2f" % converge_time)
env.visualize_value_function(V)
env.visualize_best_actions(policy)
