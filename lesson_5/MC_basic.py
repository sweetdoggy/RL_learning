import numpy as np
from basic_env.grid_env import grid_env
import time


def random_policy(i, j, k):
    psa = np.random.rand(i, j, k)
    sums = np.sum(psa, axis=2, keepdims=True)
    normalized_psa = psa / sums
    return normalized_psa


def sample_trajectory_and_cal_qsa(env, state, action,policy,gamma=0.9,episode_length=1, episode_num=1):
    """

    :param env:
    :param state:
    :param action:
    :param policy:
    :param episode_length:
    :param episode_num: if policy is deterministic, one episode would be sufficient to get the action value,which means episode num=1
    :return:
    """
    reward_list=[]
    trajectory_list=[]
    for _ in range(episode_num):
        reward = []
        trajectory = []
        trajectory.append({"state":state})
        current_state=state
        current_action=action
        for i in range(episode_length):
            next_state,r=env.get_next_agent_state_and_reward(current_state,current_action)
            trajectory.append({"action":current_action})
            trajectory.append({"state":next_state})
            reward.append(r)
            current_state=next_state
            x,y=current_state
            action_probs=policy[x,y]
            current_action=np.random.choice(len(action_probs),p=action_probs)
        reward_list.append(reward)
        trajectory_list.append(trajectory)
    discounted_gt_list = []
    for episode in reward_list:
        discounted_gt = 0
        for r in reversed(episode):
            discounted_gt = discounted_gt * gamma + r
        discounted_gt_list.append(discounted_gt)
    qsa = sum(discounted_gt_list) / len(discounted_gt_list)
    return qsa

def mc_basic(env,gamma=0.9,episode_length=10,episode_num=1):
    policy=random_policy(env.grid_size[0],env.grid_size[1],env.action_space_size)
    while True:
        new_policy=policy.copy()
        for i in range(env.grid_size[0]):
            for j in range(env.grid_size[1]):
                q_values=[]
                for action in range(env.action_space_size):
                    qsa=sample_trajectory_and_cal_qsa(env,[i,j],action,policy,gamma=gamma,episode_length=episode_length,episode_num=1)
                    q_values.append(qsa)
                best_action = np.argwhere(q_values == np.max(q_values)).flatten()
                bad_action = np.argwhere(q_values != np.max(q_values)).flatten()
                new_policy[i, j, best_action] = 1 / len(best_action)
                new_policy[i, j, bad_action] = 0
        if np.array_equal(policy,new_policy):
            break
        policy=new_policy
    #calculate v(s)
    V=np.zeros(env.grid_size)
    q=np.zeros((env.grid_size[0],env.grid_size[1],env.action_space_size))
    for i in range(env.grid_size[0]):
        for j in range(env.grid_size[1]):
            q_values=[]
            reward_list=[]
            for action in range(env.action_space_size):
                qsa=sample_trajectory_and_cal_qsa(env,[i,j],action,policy,gamma,episode_length,episode_num=episode_num)
                q[i,j,action]=qsa
                next_state,reward=env.get_next_agent_state_and_reward([i,j],action)
                q_values.append(qsa)
                reward_list.append(reward)
            V[i,j]=np.dot(policy[i,j],np.array(q_values))
    return policy,V,q


env=grid_env(grid_size=[5,5],target=[3,2],forbidden=[[1,1],[2,4],[3,3],[4,1]],start=[0,0])
#env.random_initial(forbidden_num=3)
start=time.time()
policy,V,q=mc_basic(env,episode_length=100,episode_num=1)
end=time.time()
converge_time=end-start
print("converge_time: %.2f" %converge_time)
env.visualize_best_actions(policy)
env.visualize_value_function(V)

