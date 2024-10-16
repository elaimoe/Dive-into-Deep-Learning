import torch
from d2l import torch as d2l


# 这节将从0开始实现线性回归，首先来定义模型
class LinearRegressionScratch(d2l.Module):  #@save
    """The linear regression model implemented from scratch."""

    def __init__(self, num_inputs, lr, sigma=0.01):
        # num_inputs为输入个数，lr为学习率，sigma为权重的随机初始化标准差
        super().__init__()  # super是调用父类的一个方法，这里是调用父类的构造函数
        self.save_hyperparameters()  # 保存超参数
        self.w = torch.normal(0, sigma, (num_inputs, 1), requires_grad=True)
        # 权重初始化为标准正态分布，形状为(num_inputs, 1)，requires_grad=True表示需要计算梯度
        self.b = torch.zeros(1, requires_grad=True)  # 偏置初始化为0，需要计算梯度


# 自己实现前向传播
@d2l.add_to_class(LinearRegressionScratch)  #@save
def forward(self, X):
    return torch.matmul(X, self.w) + self.b  # matmul 矩阵相乘


# 定义损失函数，返回均方误差
@d2l.add_to_class(LinearRegressionScratch)  #@save
def loss(self, y_hat, y):
    l = (y_hat - y) ** 2 / 2
    return l.mean()


# 虽然在线性回归里这样做没有必要，但是为了后续神经网络的学习，将在这里学习小批量随机梯度下降（SGD）的实现
# 暂时忽略学习率需要针对不同大小的小批量进行调整的依赖关系
class SGD(d2l.HyperParameters):  #@save
    """Minibatch stochastic gradient descent."""

    def __init__(self, params, lr):  # params为模型参数，lr为学习率
        self.save_hyperparameters()

    def step(self):
        for param in self.params:
            param -= self.lr * param.grad  # 梯度下降中更新参数的公式

    # 将所有梯度更新为0
    def zero_grad(self):
        for param in self.params:
            if param.grad is not None:
                param.grad.zero_()


@d2l.add_to_class(LinearRegressionScratch)  #@save
def configure_optimizers(self):
    return SGD([self.w, self.b], self.lr)
