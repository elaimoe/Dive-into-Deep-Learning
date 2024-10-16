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


# 配置优化器
@d2l.add_to_class(LinearRegressionScratch)  #@save
def configure_optimizers(self):
    return SGD([self.w, self.b], self.lr)


# 前置组件都已经完成，下面开始介绍训练主循环
@d2l.add_to_class(d2l.Trainer)  #@save
def prepare_batch(self, batch):
    return batch


@d2l.add_to_class(d2l.Trainer)  #@save
def fit_epoch(self):
    self.model.train() # 设置为训练模式
    for batch in self.train_dataloader: # 遍历训练数据
        loss = self.model.training_step(self.prepare_batch(batch)) # 计算损失
        self.optim.zero_grad() # 梯度清零
        with torch.no_grad(): # 禁用梯度计算
            loss.backward() # 反向传播
            if self.gradient_clip_val > 0:  # To be discussed later
                self.clip_gradients(self.gradient_clip_val, self.model) # 裁剪梯度
            self.optim.step() # 更新参数
        self.train_batch_idx += 1 # 更新批次索引
    if self.val_dataloader is None: # 如果没有验证数据，则跳过验证
        return
    self.model.eval() # 设置为评估模式
    for batch in self.val_dataloader: # 遍历验证数据
        with torch.no_grad(): # 禁用梯度计算
            self.model.validation_step(self.prepare_batch(batch)) # 计算损失
        self.val_batch_idx += 1 # 更新批次索引


model = LinearRegressionScratch(2, lr=0.03) # 初始化模型
data = d2l.SyntheticRegressionData(w=torch.tensor([2, -3.4]), b=4.2) # 初始化数据
trainer = d2l.Trainer(max_epochs=3) # 初始化训练器
trainer.fit(model, data) # 训练


with torch.no_grad():
    print(f'error in estimating w: {data.w - model.w.reshape(data.w.shape)}')
    print(f'error in estimating b: {data.b - model.b}')


# 深度学习模型中的参数一般是通过优化算法（如随机梯度下降）来学习的，而这些算法往往无法保证找到全局最优解。由于模型的复杂性，
# 多个不同的参数配置可能都会带来相近的预测精度。换句话说，深度学习更关心模型的预测性能，而不是找到唯一的参数解。
""" Exercises
1. 权重初始化为 0，会导致梯度不变，模型无法收敛。方差初始化为 1，可能会导致梯度过大权重爆炸的问题。
    适当的权重初始化（如 Xavier 或 He 初始化）对深度网络的收敛非常重要
2. 自动微分技术可以用来计算模型的参数梯度
5. 计算复杂度过高，最好使用一些近似算法。
6. 较高的学习率可能会导致较快的初始损失下降，但也可能导致模型在接近最优解时不稳定，甚至会跳过最优解。
    增加训练的 epochs 可以帮助模型获得更低的误差。
7. 如果指定了 drop_last=True ，那么不完整的批次将被丢弃。
8. 绝对值损失函数可以减少异常值的影响。
    如果输入中存在大规模的噪声或异常值，绝对值损失函数的表现会更稳定，而平方损失会更敏感，导致梯度变得非常大。
    可以使用 Huber 损失函数，它结合了绝对值和平方损失的优点。
9. 重新洗牌可以避免模型过度依赖特定的训练样本顺序，防止模型过拟合到特定的顺序模式。
"""