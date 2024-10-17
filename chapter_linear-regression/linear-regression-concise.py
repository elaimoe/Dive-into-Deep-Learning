import matplotlib
import numpy as np
import torch
from torch import nn
from d2l import torch as d2l


# 定义线性回归类，继承自 Module 类
class LinearRegression(d2l.Module):  #@save
    """The linear regression model implemented with high-level APIs."""
    def __init__(self, lr):
        super().__init__()
        self.save_hyperparameters() # 保存超参数
        self.net = nn.LazyLinear(1) # LazyLinear 类不用指定输入维度，只需指定输出维度即可，简化操作
        self.net.weight.data.normal_(0, 0.01) # 正态分布初始化，均值为0，方差为0.01
        self.net.bias.data.fill_(0) # 偏差初始化为0，bias是偏置的意思，fill后_表示在原地操作


# 定义前向传播函数
@d2l.add_to_class(LinearRegression)  #@save
def forward(self, X):
    return self.net(X) # 使用网络计算 X 的输出
    # 线性模型是神经网络的一个特例，因为它可以被看作是一个只包含一个线性层且没有激活函数的神经网络


# 计算损失，均方误差
@d2l.add_to_class(LinearRegression)  #@save
def loss(self, y_hat, y):
    fn = nn.MSELoss() # 使用 PyTorch 内置的均方误差函数
    return fn(y_hat, y)


# 定义优化器配置函数
@d2l.add_to_class(LinearRegression)  #@save
def configure_optimizers(self):
    return torch.optim.SGD(self.parameters(), self.lr) # 使用 SGD 优化器
    # self.parameters()：获取模型中需要被更新的所有参数，self.lr是学习率


# 训练
model = LinearRegression(lr=0.03) # 学习率
data = d2l.SyntheticRegressionData(w=torch.tensor([2, -3.4]), b=4.2) # 生成数据
trainer = d2l.Trainer(max_epochs=3) # 训练轮数
trainer.fit(model, data)
# 这里的动图建议在 jupyter notebook 中查看，改库有点麻烦，不知道如何下手


# 获取训练后模型的权重和偏置
@d2l.add_to_class(LinearRegression)  #@save
def get_w_b(self):
    return (self.net.weight.data, self.net.bias.data)
w, b = model.get_w_b()

print(f'error in estimating w: {data.w - w.reshape(data.w.shape)}')
print(f'error in estimating b: {data.b - b}')


""" Exercises
1. 参见结尾讨论区，此题存疑。
    Q：什么是“minibatch上的总损失”？
    A：处理时将数据集分成多个小批量（minibatches），在每次参数更新只使用一个小批量的数据，这就是 minibatch 梯度下降。
    对于一个 minibatch，我们计算每个样本的损失，然后将这些损失加总起来，得到总损失。
    例如，如果 minibatch 包含 64 个样本，我们会得到 64 个损失值，将它们相加就是总损失。
2. 之前提到过，Huber 损失函数结合了均方误差和绝对误差的优点，具有鲁棒性。在误差较小时，它表现为均方误差，
    而当误差较大时，它则转为绝对误差，从而减少了异常值对模型的影响。
3. model.parameters().grad
4. 适度增大学习率可以加快收敛，但如果学习率过高，可能会导致模型不稳定或错过最优解。
    增加 epoch 通常有助于改善模型性能，但在达到一定点后，模型可能会出现过拟合或收敛缓慢，因此不一定总是改进。
5. 估计误差通常随着数据量的增加而减少，但误差减少的速度并不是线性的。
    线性增加数据量会错过数据规模对模型性能产生较大影响的初始阶段。
"""