import torch
from d2l import torch as d2l


# 对每一个批次计算损失和准确率，以计算平均损失和准确率，忽略最终不足一个批次的数据产生的影响
class Classifier(d2l.Module):  #@save
    """The base class of classification models."""
    def validation_step(self, batch):
        Y_hat = self(*batch[:-1])
        self.plot('loss', self.loss(Y_hat, batch[-1]), train=False)
        self.plot('acc', self.accuracy(Y_hat, batch[-1]), train=False)


# 使用随机梯度下降优化器
@d2l.add_to_class(d2l.Module)  #@save
def configure_optimizers(self):
    return torch.optim.SGD(self.parameters(), lr=self.lr)


# 分类的准确率就是正确预测数占所有预测数的比例，它不可微分，直接优化准确率很有难度
@d2l.add_to_class(Classifier)  #@save
def accuracy(self, Y_hat, Y, averaged=True):
    """Compute the number of correct predictions."""
    # 这个函数用来计算分类准确率，Y_hat 是预测值，Y 是真实值，averaged 表示是否对每个样本的预测求平均
    Y_hat = Y_hat.reshape((-1, Y_hat.shape[-1]))  # 重塑预测输出的形状，使其适应计算需求，形状为 (batch_size, num_classes)
    preds = Y_hat.argmax(axis=1).type(Y.dtype)  # 使用 argmax 得到每行的最大值索引，作为预测类别
    # 0 表示按列操作，1 表示按行操作。最后类型转换以保证正确比较。
    compare = (preds == Y.reshape(-1)).type(torch.float32)  # 将预测与真实值进行比较，生成 0 或 1 的张量
    return compare.mean() if averaged else compare  # 如果 averaged 为 True，则返回平均准确率，否则返回比较结果


""" Exercises
1. 略
2. 这玩意真不太符合直觉，可能是因为目前没学过统计学吧
    其方差更大，稳定性较差
3. 略

ChatGPT-o1-preview: https://chatgpt.com/share/671e0925-e9a0-8007-b901-87a5d96f9584

1. **表达 \(L_v\)、\(L_v^q\)、\(L_v^b\)：**

设验证集有 \(N\) 个样本，小批量大小为 \(B\)，则共有 \(M = \lceil \frac{N}{B} \rceil\) 个小批量。定义：

- **验证损失 \(L_v\)：**
  \[
  L_v = \frac{1}{N} \sum_{i=1}^{N} l_i
  \]
  其中，\(l_i\) 是第 \(i\) 个样本的损失。

- **快速估计 \(L_v^q\)：**
  \[
  L_v^q = \frac{1}{M} \sum_{m=1}^{M} L_m
  \]
  其中，\(L_m = \frac{1}{n_m} \sum_{i=1}^{n_m} l_i\) 为第 \(m\) 个小批量的平均损失，\(n_m\) 是第 \(m\) 个小批量的样本数。

- **最后一个小批量的损失 \(L_v^b\)：**
  \[
  L_v^b = L_M = \frac{1}{n_M} \sum_{i=1}^{n_M} l_i
  \]
  其中，\(n_M\) 是最后一个小批量的样本数，可能小于 \(B\)。

因此，这些量与 \(L_v\)、\(L_v^q\)、样本大小 \(N\) 和小批量大小 \(B\) 的关系如上所示。

2. **证明 \(L_v^q\) 是无偏估计且为何仍偏向使用 \(L_v\)：**

- **证明：**
  快速估计 \(L_v^q\) 的期望为：
  \[
  E[L_v^q] = E\left[ \frac{1}{M} \sum_{m=1}^{M} L_m \right] = \frac{1}{M} \sum_{m=1}^{M} E[L_m]
  \]
  由于每个小批量损失 \(L_m\) 是样本损失的平均值，且样本是独立同分布的，所以：
  \[
  E[L_m] = E[l_i] = E[L_v]
  \]
  因此：
  \[
  E[L_v^q] = \frac{1}{M} \times M \times E[L_v] = E[L_v]
  \]
  这证明了 \(L_v^q\) 是 \(L_v\) 的无偏估计。

- **为何仍使用 \(L_v\)：**
  尽管 \(L_v^q\) 是无偏的，但它的方差较大，因为它是基于每个小批量的平均损失计算的，样本数量较少。而 \(L_v\) 利用所有验证集数据计算，方差较小，估计更稳定和准确。因此，在可能的情况下，我们更倾向于使用 \(L_v\) 来评估模型的性能。

3. **推导最优选择 \(y'\) 的规则：**

目标是找到使期望损失最小的预测 \(y'\)。期望损失定义为：
\[
E_{y \sim p(y|x)} [l(y, y')] = \sum_{y} l(y, y') p(y|x)
\]
最优的 \(y'\) 满足：
\[
y'^* = \arg \min_{y'} E_{y \sim p(y|x)} [l(y, y')] = \arg \min_{y'} \sum_{y} l(y, y') p(y|x)
\]
因此，最优选择规则是：对每个可能的 \(y'\)，计算其对应的期望损失，选择使期望损失最小的 \(y'\)。
"""