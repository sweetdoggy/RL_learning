import numpy as np
from basic_env.grid_env import grid_env
import time
def policy_iteration(env,gamma=0.9,theta=1e-4):
    V=np.zeros(env.grid_size)
    def generate_normalized_random_matrix(m,n,p):
        random_matrix=np.random.rand(m,n,p)
        sums=np.sum(random_matrix,axis=2,keepdims=True)
        normalized_matrix=random_matrix/sums
        return normalized_matrix
    # random choice psa
    psa = generate_normalized_random_matrix(env.grid_size[0], env.grid_size[1], env.action_space_size)
    total_rounds=0
    while True:
        while True:
            new_V=V.copy()
            for i in range(env.grid_size[0]):
                for j in range(env.grid_size[1]):
                    Vs=[]
                    immediate_reward=[]
                    for action in range(env.action_space_size):
                        next_state,reward=env.get_next_agent_state_and_reward([i, j], action)
                        x,y=next_state
                        v=V[x,y]
                        Vs.append(v)
                        immediate_reward.append(reward)
                    new_V[i,j]=np.dot(np.array(immediate_reward),psa[i,j])+gamma*np.dot(psa[i,j],np.array(Vs))
            delta=max(0,np.max(abs(new_V-V)))
            V=new_V
            total_rounds+=1
            if delta<theta:
                break
        new_psa=psa.copy()
        for i in range(env.grid_size[0]):
            for j in range(env.grid_size[1]):
                q=[]
                for action in range(env.action_space_size):
                    next_state, reward = env.get_next_agent_state_and_reward([i, j], action)
                    x, y = next_state
                    v = V[x, y]
                    q.append(reward+gamma*v)
                best_action=np.argwhere(q==np.max(q)).flatten()
                bad_action=np.argwhere(q!=np.max(q)).flatten()
                new_psa[i,j,best_action]=1/len(best_action)
                new_psa[i,j,bad_action]=0
        if np.array_equal(new_psa,psa):
            break
        psa=new_psa
    return V,psa,total_rounds
if __name__=="__main__":
    env=grid_env(grid_size=[5,5],target=[3,2],forbidden=[[1,1],[2,4],[3,3],[4,1]],start=[0,0])
    start=time.time()
    V,policy,total_rounds=policy_iteration(env)
    end=time.time()
    converge_time=end-start
    print("converge_time: %.2f" %converge_time)
    print("total_rounds: %d" %total_rounds)
    env.visualize_value_function(V)
    env.visualize_best_actions(policy)




