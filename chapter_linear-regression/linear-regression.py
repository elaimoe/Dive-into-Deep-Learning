import math
import time
import numpy as np
import torch
from d2l import torch as d2l

# 实例化两个 10,000 维向量 包含所有 1
n = 10000
a = torch.ones(n)
b = torch.ones(n)

# 使用python循环实现加法，计算时间 0.05609 sec
c = torch.zeros(n)
t = time.time()
for i in range(n):
    c[i] = a[i] + b[i]
print(f'{time.time() - t:.5f} sec')

# 使用向量加法实现加法，计算时间 0.0000 sec，可以看到速度提升非常多
t = time.time()
d = a + b
print(f'{time.time() - t:.5f} sec')


# 定义一个函数来计算正态分布，实际上使用的是正态分布概率密度函数公式 https://baike.baidu.com/item/%E6%AD%A3%E6%80%81%E5%88%86%E5%B8%83
def normal(x, mu, sigma):
    p = 1 / math.sqrt(2 * math.pi * sigma ** 2)
    return p * np.exp(-0.5 * (x - mu) ** 2 / sigma ** 2)


x = np.arange(-7, 7, 0.01) # 生成一个从 -7 到 7 的等差数列，步长为 0.01
params = [(0, 1), (0, 2), (3, 1)] # 包含多个均值和标准差的元组列表，每个元组 (mu, sigma) 表示一个正态分布的参数
d2l.plot(x, [normal(x, mu, sigma) for mu, sigma in params], xlabel='x', # 这里的循环写法有些不常见
         ylabel='p(x)', figsize=(4.5, 2.5),
         legend=[f'mean {mu}, std {sigma}' for mu, sigma in params])
d2l.plt.show() # 画出正态分布，这里实际上是d2l将matplotlib的封装起来了

"""
# Exercises
1. 我的第一反应是使用梯度下降法。
    而如果数据集符合正态分布，那么均值就是数据的期望，期望恰好是最优估计。https://zhuanlan.zhihu.com/p/356850764
    当使用绝对值损失时，最优解是中位数而不是均值。
2. 仿射函数与线性函数严格意义上讲区别只在于有没有截距 https://www.zhihu.com/question/52571431
    证明很简单，实际上就是把b增加一个维度将截距“吞”进去而已。
3. 这个目前还不知道，但是经过查询，貌似可以在神经网络中加入隐藏层以避免手动实现
4. 矩阵不满秩，意味着特征之间存在线性依赖关系，可能导致不稳定或不可解
    添加正则化项变成满秩阵或加入高斯噪声打破依赖关系
    矩阵不满秩时，随机梯度下降可能会在某些方向上无法更新参数，导致收敛缓慢或者根本无法收敛
5. 是一个绝对误差损失函数
    封闭形式解一般较难求出
    可能会在静止点附近出现震荡，因为绝对误差损失函数在零梯度处不光滑。
    解决方法可以是在更新过程中引入动量或者采用平滑的近似损失函数（如Huber损失函数），以减轻震荡问题
6. 两层线性变换的组合本质上仍然是一个线性变换。如果两个线性层没有非线性激活函数的参与，模型无法捕捉到复杂的非线性关系
7. 高斯分布具有对称性和无限支持，这意味着噪声可以取负值，这会导致出现负价格，这在实际中是不可能的
    波动性方面，价格波动往往也表现出异方差性（波动不稳定），而高斯噪声假设的方差是固定的
    价格通常具有右偏分布，而取对数后可以将数据变得更加正态化
8. 苹果的销售数量是一个计数值（离散数据），而不是连续数据。
    高斯噪声模型通常适用于连续数据，而计数数据往往不符合高斯分布假设，可能会导致不合理的预测值（如负数）。
    更合适的选择是使用离散的概率分布，如泊松分布。
"""