# Learning with Rethinking：反馈迭代修正

> Li, X., Jie, Z., Feng, J., Liu, C., & Yan, S. (2017). Learning with Rethinking: Recurrently Improving Convolutional Neural Networks through Feedback.
>
> arXiv: https://arxiv.org/abs/1708.04481

## 解决的问题

传统CNN只有前馈结构，信息从输入单向传到输出，没有自上而下的反馈机制来修正预测。人类视觉系统中，高层语义信息会反馈到低层，指导重新审视输入。

## 核心思想

### Feedback Layer（反馈层）

将第一次前向传播得到的后验概率 p（高层信息），通过全连接层传回低层，生成 emphasis vector（强调向量）：

$$e = f(p)$$

### Emphasis Layer（强调层）

用 emphasis vector 对低层特征图的通道重新加权：

$$\tilde{F} = E \otimes F$$

emphasis vector 经 softmax 归一化（乘以通道数使均值为1），保证加权前后特征的期望值不变。

### 迭代过程

- T=1：emphasis 全为1，等价于普通前馈
- T>=2：用高层预测生成反馈，调制低层特征，重新前向传播
- 多次迭代"重新思考"，逐步修正预测

## 实验结果

在 CIFAR-100、CIFAR-10、MNIST-background-image、ILSVRC-2012 上验证了反馈机制的有效性。


