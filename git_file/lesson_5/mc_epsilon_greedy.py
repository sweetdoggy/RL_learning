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

def mc_epsilon_greedy(env,gamma=0.9,epsilon=0.1,episode_length=10,episode_num=1):
    policy=random_policy(env.grid_size[0],env.grid_size[1],env.action_space_size)
    qsa=np.zeros((env.grid_size[0],env.grid_size[1],env.action_space_size))
    new_policy=policy.copy()
    rounds=0
    while True:
        i=np.random.randint(low=0,high=env.grid_size[0])
        j=np.random.randint(low=0,high=env.grid_size[1])
        action=np.random.randint(low=0,high=env.action_space_size)
        trajectory_list,reward_list=sample_trajectory(env,[i,j],action,new_policy,gamma=gamma,episode_length=episode_length,episode_num=1)
        for trajectory,reward in zip(trajectory_list,reward_list):
            new_policy,qsa=update_policy(env,trajectory, reward, qsa, gamma,epsilon)
        if np.max(np.abs(policy-new_policy))<1e-10:
            break
        policy=new_policy
        epsilon*=0.9
        rounds+=1
        print("rounds: ",rounds)

    #calculate v(s)
    V=np.zeros(env.grid_size)
    for i in range(env.grid_size[0]):
        for j in range(env.grid_size[1]):
            V[i,j]=np.dot(np.array(policy[i,j]),np.array(qsa[i,j]))
    return policy,V,qsa

def update_policy(env,trajectory,reward,qsa,gamma,epsilon):
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
            q_values=qsa[i,j]
            best_action = np.argwhere(q_values == np.max(q_values)).flatten()
            bad_action = np.argwhere(q_values != np.max(q_values)).flatten()
            policy[i, j, best_action] = (1-epsilon*(len(bad_action))/env.action_space_size)/ len(best_action)
            policy[i, j, bad_action] = epsilon/env.action_space_size
    return policy,new_qsa


env=grid_env(grid_size=[5,5],target=[3,2],forbidden=[[1,1],[2,4],[3,3],[4,1]],start=[0,0])
#env.random_initial(forbidden_num=3)
start=time.time()
policy,V,qsa=mc_epsilon_greedy(env,episode_length=10000,episode_num=1,epsilon=0.01)
end=time.time()
converge_time=end-start
print("converge_time: %.2f" %converge_time)
env.visualize_best_actions(policy)
env.visualize_value_function(V)
#epsilon greedy要得到最好的策略，最好让epsilon逐渐递减，最终能够收敛，并且epsilon一开始也要太大


