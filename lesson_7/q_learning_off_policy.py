import numpy as np
import copy
import time
from basic_env.grid_env import grid_env


class QLearningOffPolicy:
    def __init__(self, env, alpha=1e-2, gamma=0.9, theta=1e-10,episode_length=1000):
        self.env = env
        self.qsa = np.zeros(shape=(env.grid_size[0], env.grid_size[1], env.action_space_size))
        self.policy = np.ones(shape=(env.grid_size[0], env.grid_size[1], env.action_space_size)) / env.action_space_size
        self.new_qsa = copy.deepcopy(self.qsa)
        self.trajectory = []
        self.trajectory_list = []
        self.stable_steps = 0
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.theta = theta
        self.steps = 0
        self.episode_length=episode_length
        self.done = False

    def sample(self):
        i = np.random.randint(low=0, high=self.env.grid_size[0])
        j = np.random.randint(low=0, high=self.env.grid_size[1])
        while len(self.trajectory)<self.episode_length:
            state = [i, j]
            action = np.random.randint(self.env.action_space_size)
            self.trajectory.append({"state": state, "action": action})
            [i, j], reward = self.env.get_next_agent_state_and_reward([i, j], action)
            self.trajectory[-1]["reward"] = reward
        self.trajectory.append({"state": [i,j]})

    def policy_evaluation(self):
        qsa = copy.deepcopy(self.qsa)
        trajectory = copy.deepcopy(self.trajectory)
        for current_pair, next_pair in zip(trajectory[:-1], trajectory[1:]):
            [x, y] = current_pair["state"]
            action = current_pair["action"]
            reward = current_pair["reward"]
            [x_prime, y_prime] = next_pair["state"]
            q = max(self.qsa[x_prime, y_prime])
            qsa[x, y, action] = self.qsa[x, y, action] - self.alpha * (
                    self.qsa[x, y, action] - (reward + self.gamma * q))
        return qsa

    def learning(self):
        rounds = 0
        while True:
            self.sample()
            self.new_qsa = self.policy_evaluation()
            delta = np.max(np.abs(self.qsa - self.new_qsa))
            if delta < self.theta:
                self.stable_steps += 1
            else:
                self.stable_steps = 0
            self.qsa = copy.deepcopy(self.new_qsa)
            self.trajectory_list.append(self.trajectory)
            self.trajectory = []
            if self.stable_steps > 100:
                print(rounds)
                break
            rounds += 1


    def calculate_value_function(self):
        self.policy=self.get_policy()
        V = np.zeros(self.env.grid_size)
        for i in range(self.env.grid_size[0]):
            for j in range(self.env.grid_size[1]):
                q_values = []
                for action in range(self.env.action_space_size):
                    q_values.append(self.qsa[i, j, action])
                V[i, j] = np.dot(self.policy[i, j], self.qsa[i, j])
        return V

    def get_policy(self):
        policy = np.zeros(shape=(self.env.grid_size[0], self.env.grid_size[1], self.env.action_space_size))
        for i in range(self.env.grid_size[0]):
            for j in range(self.env.grid_size[1]):
                q_values = self.qsa[i, j]
                best_action = np.argwhere(q_values == np.max(q_values)).flatten()
                policy[i, j, best_action] = 1 / len(best_action)
        return policy

env = grid_env(grid_size=[5, 5], target=[3, 2], forbidden=[[1, 1], [1, 2], [2, 2], [3, 1],[3,3],[4,1]], start=[0, 0])
start = time.time()
agent = QLearningOffPolicy(env)
agent.learning()
policy=agent.get_policy()
V = agent.calculate_value_function()
end = time.time()
converge_time = end - start
print("converge_time: %.2f" % converge_time)
env.visualize_value_function(V)
env.visualize_best_actions(policy)
