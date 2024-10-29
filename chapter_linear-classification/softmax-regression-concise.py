import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l


# softmax 线性回归的简明实现
class SoftmaxRegression(d2l.Classifier):  #@save
    """The softmax regression model."""

    def __init__(self, num_outputs, lr):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential(nn.Flatten(), nn.LazyLinear(num_outputs))
        # nn.Sequential 是 PyTorch 中用于顺序构建神经网络的容器。它允许我们将多个层按照定义的顺序连接起来。
        # nn.Sequential 的工作原理是：将传入的多个层按顺序“串联”在一起，每一层的输出会自动成为下一层的输入。
        # Flatten 默认从第 2 个维度开始展平，而保持第 1 个维度不变。
        # LazyLinear 是一个延迟初始化的全连接层，它会在第一次前向传播时自动确定输入的特征数。

    def forward(self, X):
        return self.net(X) # 这个参数的传递有点离谱


@d2l.add_to_class(d2l.Classifier)  #@save
def loss(self, Y_hat, Y, averaged=True):
    # Y_hat 是模型的预测输出，Y 是真实标签
    Y_hat = Y_hat.reshape((-1, Y_hat.shape[-1]))
    # 假设 Y_hat 的原始形状为 (batch_size, num_samples, num_classes)，
    # 重塑后的形状为 (batch_size * num_samples, num_classes)
    # 换句话说就是压缩前面所有维度为一维，保留最后一个维度
    Y = Y.reshape((-1,))
    # 假设 Y 的原始形状为 (batch_size, num_samples)，即：
    # batch_size：一个 batch 中的批量大小。
    # num_samples：每个样本的子样本数量。
    # 重塑过程 Y.reshape((-1,)) 将 Y 的所有维度合并为一个一维向量。
    return F.cross_entropy(Y_hat, Y, reduction='mean' if averaged else 'none')
    # cross_entropy 是 Pytorch 中的交叉熵损失函数。
    # reduction='mean' 表示对损失求平均，reduction='none' 表示不求平均。

""" 运行需要去 jupyter notebook 中
data = d2l.FashionMNIST(batch_size=256)
model = SoftmaxRegression(num_outputs=10, lr=0.1)
trainer = d2l.Trainer(max_epochs=10)
trainer.fit(model, data)
"""

""" Exercises
1. 对于FP32单精度浮点数，通常可以安全地处理的指数范围大约在 [-88.72, 88.72] 之间
2. 映射
3. 过拟合
4. 过大不稳定，过小太慢
"""