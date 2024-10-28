import torch
from d2l import torch as d2l


# 复习一下求和操作
X = torch.tensor([[1.0, 2.0, 3.0],
                  [4.0, 5.0, 6.0]])
print(X.sum(0, keepdims=True), X.sum(1, keepdims=True))


# 仅作展示说明，对于大参数/小参数不可靠，应使用torch内置函数
def softmax(X):
    X_exp = torch.exp(X)  # 每个元素都分别求指数
    partition = X_exp.sum(1, keepdims=True)  # 按行求和，保留维度
    return X_exp / partition  # 按元素相除


X = torch.rand((2, 5))  #
X_prob = softmax(X)
print(X_prob, X_prob.sum(1))


class SoftmaxRegressionScratch(d2l.Classifier):
    def __init__(self, num_inputs, num_outputs, lr, sigma=0.01):
        # num_inputs: 输入特征数量由 28*28 计算得到 784
        # num_outputs: 输出类别数量 10
        super().__init__()
        self.save_hyperparameters()  # 保存超参数
        self.W = torch.normal(0, sigma, size=(num_inputs, num_outputs), requires_grad=True)
        # 权重尺寸为 784*10
        self.b = torch.zeros(num_outputs, requires_grad=True)
        # 偏置定为0。有的时候噪声是好东西，可以用来打破平衡。

    def parameters(self):  # 一个没什么用的参数方法
        return [self.W, self.b]


@d2l.add_to_class(SoftmaxRegressionScratch)
def forward(self, X):
    X = X.reshape((-1, self.W.shape[0]))  # 图片是 28*28 像素的，需要重塑成一个长度为 784 的一维向量
    return softmax(torch.matmul(X, self.W) + self.b)  # 简单的计算


# 下面创建 2 个标记有 3 类别的数据进行举例说明
y = torch.tensor([0, 2])  # 表示两个数据的类别分别为 0 和 2
y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])  # 写入两个数据
print(y_hat[[0, 1], y])  # 是一种没见过的索引方法，使用数组索引


# 交叉熵损失的定义是计算预测的概率分布在真实标签对应位置上的负对数似然值
def cross_entropy(y_hat, y):
    return -torch.log(y_hat[list(range(len(y_hat))), y]).mean()


print(cross_entropy(y_hat, y))


# 加入进去
@d2l.add_to_class(SoftmaxRegressionScratch)
def loss(self, y_hat, y):
    return cross_entropy(y_hat, y)

""" 这里存在bug，应该是训练超时之类的了
    请移步 jupyter notebook 运行

# 训练
data = d2l.FashionMNIST(batch_size=256)
model = SoftmaxRegressionScratch(num_inputs=784, num_outputs=10, lr=0.1)
trainer = d2l.Trainer(max_epochs=10)
trainer.fit(model, data)


# 预测
X, y = next(iter(data.val_dataloader()))
preds = model(X).argmax(axis=1)
print(preds.shape)

wrong = preds.type(y.dtype) != y
X, y, preds = X[wrong], y[wrong], preds[wrong]
labels = [a+'\n'+b for a, b in zip(
    data.text_labels(y), data.text_labels(preds))]
data.visualize([X, y], labels=labels)
"""


""" Exercises
1. 这是溢出的问题
2. 太慢了，可以直接使用PyTorch内置的torch.nn.CrossEntropyLoss函数，优化更好。
    对数的定义域为正数，需要确保预测概率大于 0
3. 特定场景下还是需要返回多个标签的
4. softmax是个分类算法，词汇量太大会导致矩阵爆炸（
5. 增加或减少学习率会影响收敛速度和模型稳定性。
    更大的批量大小通常会减少噪声，但增加计算时间。
    较小的批量大小则可能导致较大波动。
"""
