import random
import torch
from d2l import torch as d2l


# 合成回归数据，需要生成数据集，方法如下
class SyntheticRegressionData(d2l.DataModule):  #@save 该类继承自 d2l.DataModule
    """Synthetic data for linear regression."""

    def __init__(self, w, b, noise=0.01, num_train=1000, num_val=1000, batch_size=32):
        # w, b为权重和偏置    noise 为噪声的标准差
        # num_train, num_val 为训练集和验证集的大小    batch_size为批量大小
        super().__init__()  # 继承父类的构造函数
        self.save_hyperparameters()  # 保存超参数，这个函数我们在上一节用到过
        n = num_train + num_val  # 数据量是训练集和验证集的总大小
        self.X = torch.randn(n, len(w))  # 以标准正态分布采样，生成(n,len(w))形状的数据集
        noise = torch.randn(n, 1) * noise  # 以标准正态分布采样，乘以噪声标准差，生成(n,1)形状的噪声项
        self.y = torch.matmul(self.X, w.reshape((-1, 1))) + b + noise
        # 叠加特征矩阵 X 和重塑形状后的权重 w，叠加偏置 b 和噪声项 noise
        # matmul 是矩阵乘法，这里相当于计算 X * w.T + b
        # reshape(-1, 1) 将 w 转换为列向量，参数 -1 是自动计算的行数，参数 1 是表示有 1 列


data = SyntheticRegressionData(w=torch.tensor([2, -3.4]), b=4.2)  # 生成数据集，给定 w 和 b，其余取默认值
print('features:', data.X[0], '\tlabel:', data.y[0])  # 访问对象 data 中属性 X 的第一个元素和属性 y 的第一个元素
# 可以看到，返回的(2,1)和(1,1)形状的数据符合预期


# 下一步需要读取数据集
# 该方法接收一个批量大小（batch size）、特征矩阵和标签向量，并生成指定大小的小批量（minibatches）
# 对于读取数据，需要区分训练模式和验证模式：
# 在训练模式中，通常希望以随机顺序读取数据，而在验证模式中，以预定义的顺序读取数据对于调试目的可能很重要。
@d2l.add_to_class(SyntheticRegressionData)  # 将方法添加到类中
# 这里是利用 d2l 包里的 add.add_to_class 函数将下面的方法添加到刚才实现的 SyntheticRegressionData 类中
def get_dataloader(self, train):  # train 为 True 时为训练模式，为 False 时为验证模式
    if train:
        indices = list(range(0, self.num_train))  # 训练集索引，生成一个从 0 到 num_train 的序列
        # The examples are read in random order # 随机顺序读取数据
        random.shuffle(indices)  # 该函数直接作用于列表，将其元素顺序随机打乱
    else:
        indices = list(range(self.num_train, self.num_train + self.num_val))  # 验证集索引,
        # 生成一个从 num_train 到 num_train+num_val 的序列，换句话说就是从训练集的最后一个元素开始到数据集结尾
    for i in range(0, len(indices), self.batch_size):  # 遍历验证集索引序列，每次取 batch_size 个元素即步长
        batch_indices = torch.tensor(indices[i: i + self.batch_size])  # 生成一个 batch_size 个元素的索引序列
        yield self.X[batch_indices], self.y[batch_indices]  # 返回 batch_size 个元素的特征矩阵 X 和标签向量 y
        # 关于关键字 yield，参考 https://www.runoob.com/python3/python3-iterator-generator.html
        # 当在生成器函数中使用 yield 语句时，函数的执行将会暂停，并将 yield 后面的表达式作为当前迭代的值返回。
        # 然后，每次调用生成器的 next() 方法或使用 for 循环进行迭代时，函数会从上次暂停的地方继续执行，直到再次遇到 yield 语句。
        # 这样，生成器函数可以逐步产生值，而不需要一次性计算并返回所有结果。


X, y = next(iter(data.train_dataloader()))  # 创建一个迭代器，返回函数 get_dataloader 中 yield 最后返回的两个值
print('X shape:', X.shape, '\ty shape:', y.shape)


# 可以看到，X 和 y 的形状符合预期：batch_size=32，所以 X 和 y 的行数都是 32


# 虽然上面实现的迭代在实际问题中效率不高，它要求将所有数据加载到内存中，并执行大量随机的内存访问。
# 深度学习框架中实现的内置迭代器要高效得多，下面将调用框架中现有的 API 来加载数据，以实现同样的功能。
@d2l.add_to_class(d2l.DataModule)  #@save
def get_tensorloader(self, tensors, train, indices=slice(0, None)): # slice 是一个切片函数，None表示到结束。
    # 这个函数的作用是创建一个迭代器，返回一个数据集。
    tensors = tuple(a[indices] for a in tensors)
    # 这是一个推导式语法，从 tensors 中依次取出元素赋值给 a，并选取指定索引 indices 的元素，组成一个新的元组。
    dataset = torch.utils.data.TensorDataset(*tensors)
    # *tensors：表示将元组解包，将每个张量传入 TensorDataset 中
    # 这个数据集对象会将传入的张量一一对应组合在一起
    # 例如，如果有两个张量 X 和 y，TensorDataset 会将 X[i] 和 y[i] 组合在一起，形成一个新的数据样本
    return torch.utils.data.DataLoader(dataset, self.batch_size, shuffle=train)
    # 返回一个数据加载器，用于从数据集中加载数据，它实现了我们之前实现的功能


# 功能类似，这里不重复说明
@d2l.add_to_class(SyntheticRegressionData)  #@save
def get_dataloader(self, train):
    i = slice(0, self.num_train) if train else slice(self.num_train, None)
    return self.get_tensorloader((self.X, self.y), train, i)


# 与之前的功能完全一致
X, y = next(iter(data.train_dataloader()))
print('X shape:', X.shape, '\ty shape:', y.shape)
# 框架 API 提供的数据加载器支持的内置方法，因此我们可以查询它的长度，即批次数
print(len(data.train_dataloader()))


""" Exercises
1. 如果样本数量不能被批大小整除，最后一个批次可能会比其他批次小，导致数据处理的不一致。
    在某些框架中可以通过 API 选项来丢弃最后一个不完整的批次，或者通过补充数据（如添加填充样本）确保所有批次大小一致。
2. 使用外部存储设备进行数据分批加载
    分块读取，块内打乱，这样减少随机读写
    参考资料：
    https://en.wikipedia.org/wiki/Pseudorandom_permutation
    https://en.wikipedia.org/wiki/Fisher%E2%80%93Yates_shuffle
3. def data_generator():
    while True:
        yield generate_new_data_sample()  # 每次生成新样本
4. 使用种子
"""


