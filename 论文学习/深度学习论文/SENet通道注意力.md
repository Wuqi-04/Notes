# SENet：Squeeze-and-Excitation 通道注意力

> Hu, J., Shen, L., & Sun, G. (2018). Squeeze-and-Excitation Networks. CVPR 2018.
>
> arXiv: https://arxiv.org/abs/1709.01507

## 解决的问题

之前的工作主要关注空间维度的特征增强，忽略了特征通道之间的关系。不同通道对应不同的特征模式，网络应该自适应地关注重要通道、抑制不重要通道。

## 核心思想：Squeeze + Excitation

### Squeeze（压缩）

对每个通道做全局平均池化，得到通道描述符：

$$z_c = \frac{1}{H \times W}\sum_{i=1}^{H}\sum_{j=1}^{W} x_c(i,j)$$

将 H×W 的空间信息压缩为一个标量，获得全局感受野。

### Excitation（激励）

用两层全连接网络学习通道间的依赖关系：

$$s = \sigma(W_2 \cdot \text{ReLU}(W_1 \cdot z))$$

- W1 降维（reduction ratio=16），W2 升维
- Sigmoid 将权重限制在 (0,1)
- 得到每个通道的重要性权重

### Scale（重标定）

用学到的权重对原特征图逐通道加权：

$$\tilde{x}_c = s_c \cdot x_c$$

## 特点

- 即插即用，可以嵌入任何CNN
- 增加的参数量和计算量很小
- ILSVRC 2017 分类冠军

## 在本项目中的应用

在 ResNet-50 的每个 stage 后加入 SE Block，让网络自动关注对犬种识别重要的特征通道（如毛发纹理、耳朵形状对应的通道）。
