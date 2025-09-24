import numpy as np
import copy
import time
from basic_env.grid_env import grid_env
def random_policy(i, j, k):
    psa = np.random.rand(i, j, k)
    sums = np.sum(psa, axis=2, keepdims=True)
    normalized_psa = psa / sums
    return normalized_psa

class QLearningOnPolicy:
    def __init__(self,env,alpha=1e-2,gamma=0.9,theta=1e-6,epsilon=0.1):
        self.qsa=np.zeros(shape=(env.grid_size[0], env.grid_size[1], env.action_space_size))
        self.policy=random_policy(env.grid_size[0], env.grid_size[1], env.action_space_size)
        self.new_qsa=copy.deepcopy(self.qsa)
        self.trajectory=[]
        self.trajectory_list=[]
        self.stable_steps=0
        self.env=env
        self.alpha=alpha
        self.gamma=gamma
        self.theta=theta
        self.epsilon=epsilon
        self.steps=0
        self.done=False



    def policy_evaluation(self):
        qsa=copy.deepcopy(self.qsa)
        state=self.trajectory[-1]["state"]
        if state !=self.env.target:
            self.sample()
            action = self.trajectory[-2]["action"]
            reward = self.trajectory[-2]["reward"]
            next_state = self.trajectory[-1]["state"]
            q = max(self.qsa[next_state[0], next_state[1]])
            qsa[state[0], state[1], action] = self.qsa[state[0], state[1], action] - self.alpha * (
                        self.qsa[state[0], state[1], action] - (reward + self.gamma * q))
        else:
            self.done=True
        return qsa

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
        self.trajectory.append({"state":[i,j]})
        rounds=0
        while True:
            self.new_qsa=self.policy_evaluation()
            delta=np.max(np.abs(self.qsa-self.new_qsa))
            if delta<self.theta:
                self.stable_steps+=1
            else:
                self.stable_steps=0
            self.qsa=copy.deepcopy(self.new_qsa)
            self.policy=self.policy_update()
            if self.done is True:
                self.trajectory_list.append(self.trajectory)
                self.trajectory=[]
                i = np.random.randint(low=0, high=self.env.grid_size[0])
                j = np.random.randint(low=0, high=self.env.grid_size[1])
                self.trajectory.append({"state": [i, j]})
                self.done=False
                continue
            if self.stable_steps>50:
                print(rounds)
                break
            rounds+=1
        return self.policy

    def sample(self):
        x, y = self.trajectory[-1]["state"]
        action=np.random.randint(self.env.action_space_size)
        [i,j],reward=self.env.get_next_agent_state_and_reward([x,y ], action)
        self.trajectory[-1]["action"]=action
        self.trajectory[-1]["reward"]=reward
        self.trajectory.append({"state":[i,j]})



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
agent = QLearningOnPolicy(env)
policy = agent.learning()
V = agent.calculate_value_function()
end = time.time()
converge_time = end - start
print("converge_time: %.2f" % converge_time)
env.visualize_value_function(V)
env.visualize_best_actions(policy)