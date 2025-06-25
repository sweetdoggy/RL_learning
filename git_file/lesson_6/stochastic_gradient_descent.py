import numpy as np
import matplotlib.pyplot as plt


def data_sample(n):
    """生成n个坐标点，x和y均在(-10, 10)范围内"""
    x = np.random.uniform(low=-10, high=10, size=(n, 2))
    return x


def stochastic_gradient_descent(x, theta=1e-10, max_rounds=1000):
    w_k = np.array([10.0, 10.0])  # 初始点
    rounds = 1
    path = [w_k.copy()]

    while rounds < max_rounds:
        alpha_k = 1.0 / rounds  # 学习率

        # 抽取一个样本并计算梯度
        sample_x = x[np.random.choice(x.shape[0])]
        grad = compute_gradient(w_k, sample_x)

        # 更新参数
        w_k_plus_1 = w_k - alpha_k * grad
        path.append(w_k_plus_1.copy())

        # 检查收敛
        if np.max(np.abs(w_k - w_k_plus_1)) < theta:
            print(f"Converged after {rounds} rounds")
            break

        rounds += 1
        w_k = w_k_plus_1

    return w_k, np.array(path)


def f(w, x_i):
    """计算单个样本点到w的欧氏距离平方的一半"""
    diff = w - x_i
    return 0.5 * np.sum(diff ** 2)


def compute_gradient(w_k, x, eps=1e-10):
    """
    计算每个样本的梯度
    返回形状为 (n_samples, n_dims) 的梯度数组
    """
    n_samples = x.shape[0]
    grad = np.zeros((n_samples,))

    # 对每个样本计算梯度
    for i in range(n_samples):
        x_i = x[i]  # 单个样本

        # 对w_k的每个维度计算偏导数
        w_plus = w_k.copy()
        w_minus = w_k.copy()
        w_plus[i] += eps
        w_minus[i] -= eps
        # 计算中心差分
        grad[i,] = (f(w_plus, x_i) - f(w_minus, x_i)) / (2 * eps)

    return grad


# 测试代码
np.random.seed(42)
x = data_sample(100)
res, path = stochastic_gradient_descent(x)
print("优化结果:", res)
print("数据均值:", np.mean(x, axis=0))
# 计算数据均值
data_mean = np.mean(x, axis=0)

# 可视化收敛过程
plt.figure(figsize=(10, 8))

# 绘制数据点
plt.scatter(x[:, 0], x[:, 1], c='blue', alpha=0.5, label='Data Points')

# 绘制初始点
plt.scatter(path[0, 0], path[0, 1], c='red', marker='*', s=200, label='Initial Point')

# 绘制最终点
plt.scatter(res[0], res[1], c='green', marker='*', s=200, label='Final Point')

# 绘制数据均值点
plt.scatter(data_mean[0], data_mean[1], c='purple', marker='x', s=100, label='Data Mean')

# 绘制收敛路径
plt.plot(path[:, 0], path[:, 1], 'k-', alpha=0.5, label='Convergence Path')

# 添加箭头显示收敛方向
for i in range(len(path) - 1):
    plt.arrow(path[i, 0], path[i, 1],
              path[i+1, 0] - path[i, 0],
              path[i+1, 1] - path[i, 1],
              head_width=0.5, head_length=0.7, fc='black', ec='black', alpha=0.3)

# 英文标题和标签
plt.title('Stochastic Gradient Descent Convergence Visualization')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.legend()
plt.grid(True)
plt.axis('equal')  # Ensure equal scaling on x and y axes
plt.tight_layout()

plt.show()
