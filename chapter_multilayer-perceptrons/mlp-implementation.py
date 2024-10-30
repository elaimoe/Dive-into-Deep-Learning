import torch
from torch import nn
from d2l import torch as d2l
import os

# 解决库冲突
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# 要想正常运行而不报线程错误，需要将数据加载器中，即d2l中torch.py的这部分内容
# num_workers=4 改成 num_workers=0
#
# class DataModule(d2l.HyperParameters):
#     """The base class of data.
#
#     Defined in :numref:`subsec_oo-design-models`"""
#     def __init__(self, root='../data', num_workers=4):
#         self.save_hyperparameters()


class MLPScratch(d2l.Classifier):
    def __init__(self, num_inputs, num_outputs, num_hiddens, lr, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.W1 = nn.Parameter(torch.randn(num_inputs, num_hiddens) * sigma)
        self.b1 = nn.Parameter(torch.zeros(num_hiddens))
        self.W2 = nn.Parameter(torch.randn(num_hiddens, num_outputs) * sigma)
        self.b2 = nn.Parameter(torch.zeros(num_outputs))


# 这里先自己实现一下relu函数
def relu(X):
    a = torch.zeros_like(X)
    return torch.max(X, a)


# 展平并定义前向传播函数
@d2l.add_to_class(MLPScratch)
def forward(self, X):
    X = X.reshape((-1, self.num_inputs))
    H = relu(torch.matmul(X, self.W1) + self.b1)
    return torch.matmul(H, self.W2) + self.b2


# 训练模型
model = MLPScratch(num_inputs=784, num_outputs=10, num_hiddens=256, lr=0.1)
data = d2l.FashionMNIST(batch_size=256)
trainer = d2l.Trainer(max_epochs=10)
trainer.fit(model, data)


# 要想看图片，还是去 jupyter notebook 里看吧，实在太麻烦了，不调整了


# 简洁实现
class MLP(d2l.Classifier):
    def __init__(self, num_outputs, num_hiddens, lr):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential(nn.Flatten(), nn.LazyLinear(num_hiddens),
                                 nn.ReLU(), nn.LazyLinear(num_outputs))


model = MLP(num_outputs=10, num_hiddens=256, lr=0.1)
trainer.fit(model, data)


""" Exercises
"""