import numpy as np
from basic_env.grid_env import grid_env
import time
from collections import defaultdict

def random_policy(i, j, k):
    psa = np.random.rand(i, j, k)
    sums = np.sum(psa, axis=2, keepdims=True)
    normalized_psa = psa / sums
    return normalized_psa

def sample_trajectory(env, state, action,policy,gamma=0.9,episode_length=1, episode_num=1):
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
        current_state=state
        current_action=action
        for i in range(episode_length):
            trajectory.append({"state": current_state, "action": current_action})
            next_state,r=env.get_next_agent_state_and_reward(current_state,current_action)
            reward.append(r)
            current_state=next_state
            x,y=current_state
            action_probs=policy[x,y]
            current_action=np.random.choice(len(action_probs),p=action_probs)
        reward_list.append(reward)
        trajectory_list.append(trajectory)
    return trajectory_list,reward_list

def mc_exploring_starts(env,gamma=0.9,episode_length=10,episode_num=1):
    policy=random_policy(env.grid_size[0],env.grid_size[1],env.action_space_size)
    qsa=np.zeros((env.grid_size[0],env.grid_size[1],env.action_space_size))
    new_policy=policy.copy()
    while True:
        for i in range(env.grid_size[0]):
            for j in range(env.grid_size[1]):
                for action in range(env.action_space_size):
                    trajectory_list,reward_list=sample_trajectory(env,[i,j],action,new_policy,gamma=gamma,episode_length=episode_length,episode_num=1)
                    for trajectory,reward in zip(trajectory_list,reward_list):
                        new_policy,qsa=update_policy(env,trajectory, reward, qsa, gamma)
        if np.array_equal(policy,new_policy):
            break
        policy=new_policy
    #calculate v(s)
    V=np.zeros(env.grid_size)
    for i in range(env.grid_size[0]):
        for j in range(env.grid_size[1]):
            V[i,j]=np.dot(np.array(policy[i,j]),np.array(qsa[i,j]))
    return policy,V,qsa

def update_policy(env,trajectory,reward,qsa,gamma):
    policy=np.zeros((env.grid_size[0],env.grid_size[1],env.action_space_size))
    new_qsa=qsa.copy()
    g=0
    records=[]
    for state_action_pair, immediate_reward in zip(reversed(trajectory), reversed(reward)):
        state = state_action_pair['state']
        action = state_action_pair['action']
        g=gamma*g+immediate_reward
        records.append({'state':state,'action':action,'q':g})
    sum_dict = defaultdict(float)  # 每个键的 q 值总和
    count_dict = defaultdict(int)  # 每个键的出现次数
    # 遍历并累积每个键的 q 值
    for item in records:
        state = item['state']
        action = item['action']
        q = item['q']
        # 将 state 和 action 组合为元组作为键
        key = ((state[0],state[1]), action)
        sum_dict[key] += q
        count_dict[key] += 1

    # 计算平均值
    average_dict = {
        key: sum_dict[key] / count_dict[key]
        for key in sum_dict
    }
    result = [
        {'state': state, 'action': action, 'avg_q': avg_q}
        for (state, action), avg_q in average_dict.items()
    ]
    for item in result:
        state=item['state']
        x,y=state
        action=item['action']
        avg_q=item['avg_q']
        new_qsa[x,y,action]=avg_q
    for i in range(env.grid_size[0]):
        for j in range(env.grid_size[1]):
            q_values=new_qsa[i,j]
            best_action = np.argwhere(q_values == np.max(q_values)).flatten()
            policy[i, j, best_action] = 1 / len(best_action)
    return policy,new_qsa


env=grid_env(grid_size=[5,5],target=[3,2],forbidden=[[1,1],[2,4],[3,3],[4,1]],start=[0,0])
#env.random_initial(forbidden_num=3)
start=time.time()
policy,V,qsa=mc_exploring_starts(env,episode_length=1000,episode_num=1)
end=time.time()
converge_time=end-start
print("converge_time: %.2f" %converge_time)
env.visualize_best_actions(policy)
env.visualize_value_function(V)
#本算法中一定要把episode_length设置大一点，当设置的比较小时，无法从一个episode探索到尽可能多的(state,action)pair，会导致对qsa的估计产生偏差，从而不断的改变策略，并且收敛特别慢
# (基本上是无法收敛，即使收敛了也不是最佳策略)
#并且这个episode_length的数值和最终target_state的价值相关，episode_length的数值越大，target_state的状态价值越精确。具体原因为
#对target_state的最佳动作采样时，得到的序列为[(target_state,stay),(target_state,stay),(target_state,stay),(target_state,stay),(target_state,stay),(target_state,stay),……]其中
#共有episode_length个(target_state,stay)pair,对于每个该状态动作的qsa，按照倒序计算，他们的qsa分别为[10,10*0.9+10,(10*0.9+10)*0.9+10,((10*0.9+10)*0.9+10)*0.9+10,……]
#将他们求和得到的值除去episode_length得到的结果即为q(target_state,stay)的值，


