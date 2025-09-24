import numpy as np
import time
from basic_env.grid_env import grid_env
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


def random_policy(i, j, k):
    psa = np.random.rand(i, j, k)
    sums = np.sum(psa, axis=2, keepdims=True)
    normalized_psa = psa / sums
    return normalized_psa


class V_hat(nn.Module):
    def __init__(self):
        super(V_hat, self).__init__()
        self.linear1 = nn.Linear(2, 8)
        self.linear2 = nn.Linear(8, 32)
        self.linear3 = nn.Linear(32, 1)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class TDLearning:
    def __init__(self, env, alpha=1e-4, gamma=0.9, theta=1e-6, episode_length=100000):
        self.trajectory = []
        self.stable_steps = 0
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.theta = theta
        self.steps = 0
        self.episode_length = episode_length
        self.done = False
        self.state = None
        self.v = V_hat()
        self.policy = random_policy(env.grid_size[0], env.grid_size[1], env.action_space_size)
        self.true_v_s = get_true_vs(env, self.policy, gamma, theta)
        self.optimizer = torch.optim.Adam(self.v.parameters(), lr=self.alpha)

    def sample(self):
        state = self.state
        x, y = state
        action_probs = self.policy[x, y]
        action = np.random.choice(len(action_probs), p=action_probs)
        [x_prime, y_prime], reward = self.env.get_next_agent_state_and_reward([x, y], action)
        self.trajectory.append({"state": [x, y], "reward": reward, "next_state": [x_prime, y_prime]})
        self.state = [x_prime, y_prime]

    def learning(self):
        i = np.random.randint(low=0, high=self.env.grid_size[0])
        j = np.random.randint(low=0, high=self.env.grid_size[1])
        self.state = [i, j]
        rounds = 0
        total_error = []
        while len(self.trajectory) < self.episode_length:
            self.sample()
            x, y = self.trajectory[-1]["state"]
            reward = self.trajectory[-1]["reward"]
            x_prime, y_prime = self.trajectory[-1]["next_state"]
            self.update_value_function(self.v, [x, y], [x_prime, y_prime], reward, self.gamma)
            v_hat = self.calculate_v()
            total_error.append(self.calculate_total_error(v_hat, self.true_v_s))
        return total_error

    def update_value_function(self,model, state, next_state, reward, gamma):
        model.train()
        optimizer=self.optimizer
        # 转换为 tensor
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # shape: [1, 2]
        next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
        reward = torch.tensor([reward], dtype=torch.float32)  # shape: [1]

        # 计算当前状态值和下一个状态值
        v_s = model(state)  # shape: [1, 1]
        v_s_next = model(next_state).detach()  # shape: [1, 1]，detach 防止反向传播

        # 构造 TD(0) 目标
        v_target = reward + gamma * v_s_next  # shape: [1, 1]

        # 使用 MSE loss
        loss = F.mse_loss(v_s, v_target)

        # 反向传播和参数更新
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    @staticmethod
    def calculate_total_error(matrix1, matrix2):
        # 确保两个矩阵形状相同
        if matrix1.shape != matrix2.shape:
            raise ValueError("两个矩阵必须具有相同的形状")

        # 计算误差平方和
        squared_error = (matrix1 - matrix2) ** 2

        # 计算均方误差 (MSE)
        mse = np.mean(squared_error)

        # 计算均方根误差 (RMSE)
        rmse = np.sqrt(mse)

        return rmse

    def calculate_v(self):
        mat = np.zeros(shape=(self.env.grid_size[0], self.env.grid_size[1]))
        with torch.no_grad():
            for i in range(self.env.grid_size[0]):
                for j in range(self.env.grid_size[1]):
                    state = torch.tensor([i, j], dtype=torch.float32)
                    mat[i, j] = self.v(state)
        return mat


def visualize_matrix_3d(matrix, cmap="viridis",
                        figsize=(10, 8), elevation=30, azimuth=45,
                        show_wireframe=False, wireframe_step=5):
    """
    将2D NumPy矩阵可视化为3D曲面

    参数:
    matrix: 2D NumPy数组，要可视化的矩阵
    title: 图表标题
    cmap: 颜色映射名称，如"viridis", "plasma", "coolwarm"等
    figsize: 图表大小元组 (宽度, 高度)
    elevation: 仰角（视角高度）
    azimuth: 方位角（视角旋转）
    show_wireframe: 是否显示线框
    wireframe_step: 线框采样步长，数值越大线框越稀疏
    """
    # 获取矩阵的尺寸
    rows, cols = matrix.shape

    # 创建网格坐标
    x = np.arange(0, cols)
    y = np.arange(0, rows)
    X, Y = np.meshgrid(x, y)

    # 创建图形和3D坐标轴
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    # 绘制3D曲面
    surf = ax.plot_surface(X, Y, matrix, cmap=plt.get_cmap(cmap),
                           linewidth=0, antialiased=True, alpha=0.8)

    # 添加线框（可选）
    if show_wireframe:
        ax.plot_wireframe(X, Y, matrix, rstride=wireframe_step,
                          cstride=wireframe_step, color='black', alpha=0.3)

    # 设置视角
    ax.view_init(elev=elevation, azim=azimuth)

    # 添加颜色条
    cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
    cbar.set_label('num')

    # 设置坐标轴标签和标题
    ax.set_xlabel('column index')
    ax.set_ylabel('row index')
    ax.set_zlabel('value')
    ax.set_title("the value of different state")

    plt.tight_layout()
    plt.show()


def plot_rmse(rmse_list, algorithm_name="TD-Linear", x_label="Episode index",
              y_label="State value error (RMSE)"):
    """
    绘制 RMSE 随 Episode 变化的曲线

    :param rmse_list: 长度为 n 的 list，存储每一轮次的 RMSE 值
    :param algorithm_name: 算法名称，用于图例展示，默认 "TD-Linear"
    :param x_label: x 轴标签，默认 "Episode index"
    :param y_label: y 轴标签，默认 "State value error (RMSE)"
    """
    episode_indices = range(1, len(rmse_list) + 1)
    plt.plot(episode_indices, rmse_list, label=f"{algorithm_name}")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.show()


def get_true_vs(env, policy, gamma, theta):
    V = np.zeros(env.grid_size)
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
                V[i, j] = np.dot(policy[i, j], action_values)
                delta = max(delta, abs(v - V[i, j]))
        if delta < theta:
            break
    return V


env = grid_env(grid_size=[5, 5], target=[3, 2], forbidden=[[1, 1], [2, 4], [3, 3], [4, 1]], start=[0, 0])
start = time.time()
agent = TDLearning(env)
total_error = agent.learning()
end = time.time()
converge_time = end - start
print("converge_time: %.2f" % converge_time)
plot_rmse(total_error)
v = agent.true_v_s
v_hat = agent.calculate_v()
visualize_matrix_3d(v)
visualize_matrix_3d(v_hat)
