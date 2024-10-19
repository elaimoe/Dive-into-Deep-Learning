import torch
from torch import nn
from d2l import torch as d2l


# 高维线性回归，以展示过拟合的效果
class Data(d2l.DataModule):
    def __init__(self, num_train, num_val, num_inputs, batch_size):
        self.save_hyperparameters()
        n = num_train + num_val
        self.X = torch.randn(n, num_inputs)
        noise = torch.randn(n, 1) * 0.01  # 噪声为符合均值为0，标准差为0.01的正态分布的随机数
        w, b = torch.ones((num_inputs, 1)) * 0.01, 0.05  # 权重向量 w 和偏差 b，请看教程中的公式
        self.y = torch.matmul(self.X, w) + b + noise

    def get_dataloader(self, train):
        i = slice(0, self.num_train) if train else slice(self.num_train, None)
        return self.get_tensorloader([self.X, self.y], train, i)


# 定义范数惩罚
def l2_penalty(w):
    return (w ** 2).sum() / 2


# 定义模型，与之前的线性回归相比，这里添加了L2范数惩罚项
class WeightDecayScratch(d2l.LinearRegressionScratch):
    def __init__(self, num_inputs, lambd, lr, sigma=0.01):
        super().__init__(num_inputs, lr, sigma)
        self.save_hyperparameters()  # 保存超参数

    def loss(self, y_hat, y):
        return (super().loss(y_hat, y) +
                self.lambd * l2_penalty(self.w))  # 相比之前，加入范数惩罚项


data = Data(num_train=20, num_val=100, num_inputs=200, batch_size=5)  # 构建数据集
# num_train 为训练集大小，num_val 为验证集大小，num_inputs 为输入维度，batch_size 为批量大小
trainer = d2l.Trainer(max_epochs=10)  # max_epochs 为训练轮数


def train_scratch(lambd):  # lambd 为正则化系数，控制权重衰减强度
    model = WeightDecayScratch(num_inputs=200, lambd=lambd, lr=0.01)  # 构建模型
    model.board.yscale = 'log'  # 将y轴设置成对数标度
    trainer.fit(model, data)  # 训练
    print('L2 norm of w:', float(l2_penalty(model.w)))  # 打印权重向量的L2范数


''' # 这里涉及大量计算，影响后续内容，故注释
# 这里建议使用jupyter notebook运行，因为会输出图像
train_scratch(0) # 不使用权重衰减，展示过拟合效果
train_scratch(3) # 使用权重衰减，会看到训练误差增加，但是验证误差减少
'''


# 下面是简洁实现，使用pytorch中集成的优化器来实现
class WeightDecay(d2l.LinearRegression):
    def __init__(self, wd, lr):
        super().__init__(lr) # 继承父类，初始化
        self.save_hyperparameters()
        self.wd = wd

    def configure_optimizers(self):
        return torch.optim.SGD([
            {'params': self.net.weight, 'weight_decay': self.wd},
            {'params': self.net.bias}], lr=self.lr)


model = WeightDecay(wd=3, lr=0.01) # 权重衰减系数为3，学习率为0.01
model.board.yscale = 'log'
trainer.fit(model, data)

print('L2 norm of w:', float(l2_penalty(model.get_w_b()[0])))


""" Exercises
1.当 𝜆 较小时，模型在训练集上的表现会较好，但可能出现过拟合，导致验证集上表现较差。
    随着 𝜆 增大，训练集的误差增加，但验证集的误差可能减小，模型的泛化能力提升。
    过大的 𝜆 可能导致欠拟合，训练和验证集上的误差都增大。
2. 通过验证集误差的最小化可以找到一个看似最优的 𝜆 值。但该值是否真正最优取决于模型在未见数据上的泛化能力，
    并且在不同的数据集或问题上，最优值可能会有所不同。因此，它是一个近似解，而非唯一解。
3. l1 正则化的更新方程会导致权重值缩减，并且比 l2 更倾向于产生稀疏解（即部分权重被推向 0）。
    具体的梯度下降更新规则中，l1 正则化会对每个权重项增加一个与符号相关的更新，而不像l2那样均匀。
4. 查资料
5. 数据增强：通过扩展训练集的多样性，例如使用图像旋转、翻转等操作。
    Dropout：在训练过程中随机丢弃部分神经元，以减少过度依赖特定的权重。
    交叉验证：使用交叉验证来评估模型的泛化能力，避免过拟合。
6. 在贝叶斯框架中，正则化可以解释为对参数施加先验分布。例如，l2 正则化相当于在参数上施加高斯先验，
    l1 正则化相当于施加拉普拉斯先验。这种先验会通过最大后验估计（MAP）影响损失函数的优化过程。
"""
