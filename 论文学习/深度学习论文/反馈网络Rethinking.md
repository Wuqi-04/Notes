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

## 在本项目中的应用与分析

实现简化版反馈网络：
1. 第一次前向传播得到初始预测
2. 用高层信息生成门控反馈，调制浅层特征
3. 第二次前向传播得到修正预测
4. 损失 = 0.3 * loss1 + 0.7 * loss2

**预期效果有限**：反馈机制在噪声/遮挡/视频等场景优势明显，但在干净的静态图像分类任务上提升约0-1%。这是因为干净图像中第一次前向传播已经提取了足够信息，反馈修正的空间不大。在PPT汇报中应如实分析这一结果。
