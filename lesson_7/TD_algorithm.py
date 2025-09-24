import numpy as np
from basic_env.grid_env import grid_env
import time

def random_policy(i, j, k):
    psa = np.random.rand(i, j, k)
    sums = np.sum(psa, axis=2, keepdims=True)
    normalized_psa = psa / sums
    return normalized_psa


def td_algorithm(env, alpha=1e-2, gamma=0.9, theta=1e-4):
    policy=random_policy(env.grid_size[0],env.grid_size[1],env.action_space_size)
    trajectory = []
    trajectory_list=[]
    V=np.zeros(env.grid_size)
    new_V = V.copy()
    i = np.random.randint(low=0, high=env.grid_size[0])
    j = np.random.randint(low=0, high=env.grid_size[1])
    rounds = 0
    while True:
        action_probs = policy[i, j]
        action = np.random.choice(len(action_probs), p=action_probs)
        trajectory.append({"state": [i, j], "action": action})
        next_state, reward = env.get_next_agent_state_and_reward([i, j], action)
        x_prime, y_prime = next_state
        new_V[i, j] = V[i, j] - alpha * (V[i, j] - (reward + gamma * V[x_prime, y_prime]))
        if rounds > 1000000:
            break
        V = new_V.copy()
        rounds += 1
        if next_state==env.target:
            i = np.random.randint(low=0, high=env.grid_size[0])
            j = np.random.randint(low=0, high=env.grid_size[1])
            trajectory_list.append(trajectory)
            trajectory=[]
            continue
        i,j = next_state
    return V


np.random.seed(42)
env = grid_env(grid_size=[5, 5], target=[3, 2], forbidden=[[1, 1], [2, 4], [3, 3], [4, 1]], start=[0, 0])
# env.random_initial(forbidden_num=3)
start = time.time()
V = td_algorithm(env)
end = time.time()
converge_time = end - start
print("converge_time: %.2f" % converge_time)
env.visualize_value_function(V)

