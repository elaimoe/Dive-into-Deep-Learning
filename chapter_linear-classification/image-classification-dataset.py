import time
import torch
import torchvision
from torchvision import transforms
from d2l import torch as d2l

d2l.use_svg_display()


# 这里使用内置框架来加载 Fashion-MNIST 数据集
class FashionMNIST(d2l.DataModule):  #@save
    """The Fashion-MNIST dataset."""

    def __init__(self, batch_size=64, resize=(28, 28)):
        super().__init__()
        self.save_hyperparameters()
        trans = transforms.Compose([transforms.Resize(resize),
                                    transforms.ToTensor()])
        self.train = torchvision.datasets.FashionMNIST(
            root=self.root, train=True, transform=trans, download=True)
        self.val = torchvision.datasets.FashionMNIST(
            root=self.root, train=False, transform=trans, download=True)


data = FashionMNIST(resize=(32, 32))
len(data.train), len(data.val)

# 查看数据集，data.train[0] 是该数据集中第一个样本，该样本使用元组 (image, label) 存储，读取图片的大小
# 结果为 torch.Size([1, 32, 32]) 其中 1 表示只有一个通道，因为是灰度图。后面的两个数字表示图片大小
print(data.train[0][0].shape)


# 这个数据集中自带 10 个类别如下
@d2l.add_to_class(FashionMNIST)  #@save
def text_labels(self, indices):
    """Return text labels."""
    labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
              'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [labels[int(i)] for i in indices]


# 使用内置的迭代器，省得造轮子
@d2l.add_to_class(FashionMNIST)  #@save
def get_dataloader(self, train):
    data = self.train if train else self.val
    return torch.utils.data.DataLoader(data, self.batch_size, shuffle=train,
                                       num_workers=self.num_workers)


# 加载一个小批量的图像，包含 64 张图像 (前面定义过 batch_size 大小)
X, y = next(iter(data.train_dataloader()))
print(X.shape, X.dtype, y.shape, y.dtype)

# 虽然数据读取比较缓慢，但后面处理需要更长时间，所以并不会在这里产生速度瓶颈
tic = time.time()
for X, y in data.train_dataloader():
    continue
print(f'{time.time() - tic:.2f} sec')


# 可视化，只展示接口如下
def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):  #@save
    """Plot a list of images."""
    raise NotImplementedError


# 这里建议使用 jupyter notebook 运行并查看图像
@d2l.add_to_class(FashionMNIST)  #@save
def visualize(self, batch, nrows=1, ncols=8, labels=[]):
    X, y = batch
    if not labels:
        labels = self.text_labels(y)
    d2l.show_images(X.squeeze(1), nrows, ncols, titles=labels)


batch = next(iter(data.val_dataloader()))  # 从验证集中加载一批次图像
print(data.visualize(batch))  # 展示图像


""" Exercises
1. 影响，这就是随机读取和顺序读取的区别了
2. 挺慢的。"I’m a Windows user. Try it next time!"
3. https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
"""