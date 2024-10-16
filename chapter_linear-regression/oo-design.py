import time
import numpy as np
import torch
from torch import nn
from d2l import torch as d2l


# 通过这个函数可以将一个类拆分成多个代码块
def add_to_class(Class):  #@save
    """动态添加方法到类"""

    def wrapper(obj):
        setattr(Class, obj.__name__, obj)  # 通过 setattr 函数，将 obj 函数
        # 动态添加到 Class 类中，并将其名字设为 obj.__name__，也就是该函数的名字。

    return wrapper


# 使用add_to_class的方法，我们首先定义一个A类
class A:
    def __init__(self):
        self.b = 1


a = A()


# 然后在这里使用装饰器将函数 do 添加到类 A 中
# 装饰器的使用方法详见 https://www.runoob.com/python3/python-decorators.html
@add_to_class(A)
def do(self):
    print('Class attribute "b" is', self.b)  # 通过这种方法可以访问到类属性 b


a.do()  # 这里报警告是因为在一般情况下我们不会这么操作，IDE 希望你提前把这个函数写进去


# 下面我们将看到 d2l 中的一个类的功能
class HyperParameters:  #@save
    """模型超参数，它将类方法中的所有参数保存为类属性"""

    def save_hyperparameters(self, ignore=[]):  # ignore=[] 是需要忽略的参数
        raise NotImplemented  # 表示该方法尚未实现，这是一个占位符
        # 如文中所述，它的实现将推迟到 Section 23.7 中


# 这里我们使用一个类 B 继承了 d2l.HyperParameters
class B(d2l.HyperParameters):
    def __init__(self, a, b, c):
        self.save_hyperparameters(ignore=['c'])  # 将除 c 之外的参数保存为类属性
        print('self.a =', self.a, 'self.b =', self.b)
        print('There is no self.c =', not hasattr(self, 'c'))
        # hasattr(要检查的对象, 字符串形式的属性名称) 该函数返回一个布尔值


b = B(a=1, b=2, c=3)


# 这个类可以以交互方式绘制实验进度，因为有了更强大的 TensorBoard，这个类的实现将被推迟到 Section 23.7 中
class ProgressBoard(d2l.HyperParameters):  #@save
    """The board that plots data points in animation."""

    def __init__(self, xlabel=None, ylabel=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 ls=['-', '--', '-.', ':'], colors=['C0', 'C1', 'C2', 'C3'],
                 fig=None, axes=None, figsize=(3.5, 2.5), display=True):
        self.save_hyperparameters()

    def draw(self, x, y, label, every_n=1):
        raise NotImplemented


# 尝试运行上面这个在 d2l 中保存好的类
board = d2l.ProgressBoard('x')
for x in np.arange(0, 10, 0.1):
    board.draw(x, np.sin(x), 'sin', every_n=2)
    board.draw(x, np.cos(x), 'cos', every_n=10)
d2l.plt.show()  # 显示图像


# 下面这个“模型”类会在以后实现，这里先暂时跳过，看看教程中的简介就好了
class Module(nn.Module, d2l.HyperParameters):  #@save
    """The base class of models."""
    pass


# 下面这个“数据”类会在以后实现，这里先暂时跳过，看看教程中的简介就好了
class DataModule(d2l.HyperParameters):  #@save
    """The base class of data."""
    pass


# 下面这个“训练”类会在以后实现，这里先暂时跳过，看看教程中的简介就好了
class Trainer(d2l.HyperParameters):  #@save
    """The base class for training models with data."""
    pass


""" Exercises
1. 如翻译所述：在 D2L 库中找到上述类的完整实现。我们强烈建议你在对深度学习建模有更多了解后详细查看这些实现
2. 移除了 save_hyperparameters，仍然可以打印 self.a 和 self.b
    前提是在 __init__ 方法中明确地将这些参数赋值给 self.a 和 self.b
"""