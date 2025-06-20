from typing import Optional, Union, List, Tuple, SupportsFloat, Any
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from gymnasium.core import ActType, ObsType
import matplotlib.pyplot as plt

class grid_env(gym.Env):
    def __init__(self, grid_size: Union[list, tuple, np.ndarray], target: Union[list, tuple, np.ndarray],
                 forbidden: Union[list, tuple, np.ndarray], start: Union[list, tuple, np.ndarray]):
        """
        :param grid_size:the shape of this grid world
        :param target: the target point of this grid world,only one start point
        :param forbidden: the forbidden area of this grid world
        :param start: the start point of this grid world, only one start point
        """
        self.grid_size = grid_size
        self.num_states = grid_size[0] * grid_size[1]
        self.target = target
        self.start = start
        self.forbidden = forbidden
        self.agent_state_spaces = spaces.Box(low=np.array([0, 0]), high=np.array([grid_size[0] - 1, grid_size[1] - 1]),
                                             shape=(2,), dtype=np.int8)
        # action choices:
        # 0:go up
        # 1:go right
        # 2:go down
        # 3:go left
        # 4:stay
        self.action_space = spaces.Discrete(5)
        self.action_space_size = spaces.Discrete(5).n
        self.action2direction = {
            0: (0, 1),  # go up
            1: (1, 0),  # go right
            2: (0, -1),  # go down
            3: (-1, 0),  # go left
            4: (0, 0)  # stay
        }
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(low=np.array([0, 0]), high=np.array([grid_size[0] - 1, grid_size[1] - 1]),
                                    shape=(2,), dtype=np.int8),
                "target": spaces.Box(0, grid_size[0] - 1, shape=(2,), dtype=np.int8),
                "grid": spaces.Box(-1, 0, shape=(grid_size[0], grid_size[1]), dtype=np.int8)
            }
        )
        self.obs = None
        self.init_obs()
        self.agent_state = None
        self.psa = None
        self.get_psa()

    def step(
            self, action: ActType
    ):
        assert action in self.action_space, "Invalid action"
        next_agent_state, reward = self.get_next_agent_state_and_reward(action)
        self.agent_state = next_agent_state
        done = self.is_done()
        obs = self.get_obs()
        return obs, reward, done,

    def get_next_agent_state_and_reward(self,state, action):
        assert action in self.action_space, "Invalid action"
        dx, dy = self.action2direction[action]
        x, y = state
        grid_x = self.grid_size[0]
        grid_y = self.grid_size[1]
        new_x = x + dx
        new_y = y + dy
        reward_forbidden = -10
        reward_step = 0
        reward_target = 10
        if new_x > grid_x - 1 or new_x < 0 or new_y > grid_y - 1 or new_y < 0:
            new_x = x
            new_y = y
            reward = reward_forbidden
        elif [new_x, new_y] in self.forbidden:
            reward = reward_forbidden
        elif [new_x, new_y] == self.target:
            reward = reward_target
        else:
            reward = reward_step
        return [new_x, new_y], reward

    def init_obs(self):
        grid = np.zeros(self.grid_size)
        for pos in self.forbidden:
            x,y=pos
            grid[x,y]=-1
        agent = np.array(self.start)
        self.agent_state = agent
        target = np.array(self.target)
        return {
                "agent": agent,
                "target": target,
                "grid": grid
            }


    def get_obs(self):
        self.obs["agent"] = self.agent_state
        return self.obs

    def is_done(self):
        return self.agent_state == self.target

    def get_psa(self):
        self.psa = np.random.dirichlet(
            alpha=np.ones(self.action_space_size),  # 对称Dirichlet分布
            size=self.grid_size[0] * self.grid_size[1]
        ).reshape(self.grid_size[0], self.grid_size[1], self.action_space_size)

    def value_iteration(self,gamma=0.9,theta=1e-4):
        V=np.zeros(self.grid_size)
        while True:
            delta=0
            for i in range(self.grid_size[0]):
                for j in range(self.grid_size[1]):
                    v=V[i,j]
                    action_values=[]
                    for action in range(self.action_space_size):
                        next_state,reward=self.get_next_agent_state_and_reward([i,j],action)
                        x,y=next_state
                        action_values.append(reward+gamma*V[x,y])
                    V[i,j]=np.max(action_values)
                    delta=max(delta,abs(v-V[i,j]))
            #self.visualize_value_function(V)
            if delta<theta:
                break
        qsa=np.zeros(shape=(self.grid_size[0],self.grid_size[1],self.action_space_size))
        policy=np.zeros(shape=(self.grid_size[0],self.grid_size[1],self.action_space_size))
        #calculate q(s,a)
        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                q_values=[]
                for action in range(self.action_space_size):
                    next_state,reward=self.get_next_agent_state_and_reward((i,j),action)
                    x,y=next_state
                    qsa[i,j,action]=reward+gamma*V[x,y]
                    q_values.append(qsa[i,j,action])
                best_actions=np.argwhere(q_values==np.max(q_values)).flatten()
                policy[i,j,best_actions]=1/len(best_actions)
        return V,qsa,policy

    def visualize_value_function(self, V):
        """可视化状态价值函数，坐标原点在左下角"""
        plt.figure(figsize=(self.grid_size[1], self.grid_size[0]))

        # 反转y轴，使原点在左下角
        plt.gca().invert_yaxis()

        im = plt.imshow(V, cmap='viridis', interpolation='nearest')

        # 添加网格线
        plt.grid(which='major', axis='both', linestyle='-', color='k', linewidth=1)
        plt.xticks(np.arange(-0.5, V.shape[1], 1))
        plt.yticks(np.arange(-0.5, V.shape[0], 1))

        # 添加数值标签
        for i in range(V.shape[0]):
            for j in range(V.shape[1]):
                # 坐标 (a,b) 对应图中 (j, n-i-1)
                text = plt.text(j, i, f'{V[i, j]:.2f}',
                                ha='center', va='center', color='w')

        plt.colorbar(label='State Value')
        plt.title('State Value Function')
        plt.tight_layout()
        plt.show()

    def visualize_best_actions(self, policy):
        """可视化每个状态的最佳动作，方向与环境定义一致"""
        plt.figure(figsize=(self.grid_size[0], self.grid_size[1]))  # 调整画布大小为列数×行数

        # 使用与环境一致的动作到方向映射
        action_to_direction = {
            0: (0, -1),  # 上
            1: (1, 0),  # 右
            2: (0, 1),  # 下
            3: (-1, 0),  # 左
            4: (0, 0)  # 停留
        }

        # 创建网格
        X, Y = np.meshgrid(np.arange(0, self.grid_size[1]), np.arange(0, self.grid_size[0]))

        # 初始化箭头数据
        U = np.zeros_like(X, dtype=float)
        V = np.zeros_like(Y, dtype=float)

        # 填充箭头数据
        for i in range(self.grid_size[0]):  # 行
            for j in range(self.grid_size[1]):  # 列
                # 获取最优动作
                best_action = np.argmax(policy[i, j])
                dx, dy = action_to_direction[best_action]
                U[i, j] = -dy
                V[i, j] = -dx

        # 绘制箭头图
        plt.quiver(X, Y, U, V, scale=self.grid_size[0] * 2, pivot='middle', color='blue')

        # 标记目标和禁区
        def plot_position(row, col, color, marker):
            """绘制指定位置的点"""
            plt.scatter(col, row, s=200, c=color, marker=marker)  # 转换坐标

        for pos in self.forbidden:
            plot_position(*pos, 'red', 'x')  # 禁区

        plot_position(self.target[0], self.target[1], 'green', 'o')  # 目标

        # 添加网格线和标签
        plt.grid(which='major', axis='both', linestyle='-', color='k', linewidth=1)
        plt.xticks(np.arange(-0.5, self.grid_size[1], 1))
        plt.yticks(np.arange(-0.5, self.grid_size[0], 1))

        # 反转 y 轴
        plt.gca().invert_yaxis()

        plt.title('Optimal Actions for Each State')
        plt.show()
    def random_forbidden(self,num):
        pass
    def random_start(self):
        pass
    def random_target(self):
        pass







