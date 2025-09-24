import numpy as np
from basic_env.grid_env import grid_env
import time
def value_iteration(env, gamma=0.9, theta=1e-4):
    V = np.zeros(env.grid_size)
    total_rounds=0
    while True:
        delta = 0
        for i in range(env.grid_size[0]):
            for j in range(env.grid_size[1]):
                v = V[i, j]
                action_values = []
                for action in range(env.action_space_size):
                    next_state, reward = env.get_next_agent_state_and_reward([i, j], action)
                    x, y = next_state
                    action_values.append(reward + gamma * V[x, y])
                V[i, j] = np.max(action_values)
                delta = max(delta, abs(v - V[i, j]))
        if delta < theta:
            break
        total_rounds+=1
    qsa = np.zeros(shape=(env.grid_size[0], env.grid_size[1], env.action_space_size))
    policy = np.zeros(shape=(env.grid_size[0], env.grid_size[1], env.action_space_size))
    # calculate q(s,a)
    for i in range(env.grid_size[0]):
        for j in range(env.grid_size[1]):
            q_values = []
            for action in range(env.action_space_size):
                next_state, reward = env.get_next_agent_state_and_reward((i, j), action)
                x, y = next_state
                qsa[i, j, action] = reward + gamma * V[x, y]
                q_values.append(qsa[i, j, action])
            best_actions = np.argwhere(q_values == np.max(q_values)).flatten()
            policy[i, j, best_actions] = 1 / len(best_actions)
    return V, policy,total_rounds
if __name__=="__main__":
    env = grid_env(grid_size=[5, 5], target=[3, 2], forbidden=[[1, 1], [2, 4], [3, 3], [4, 1]], start=[0, 0])
    start=time.time()
    V,policy,total_rounds=value_iteration(env)
    end=time.time()
    converge_time=end-start
    print("converge_time: %.2f" %converge_time)
    print("total_rounds: %d" %total_rounds)
    env.visualize_value_function(V)
    env.visualize_best_actions(policy)
