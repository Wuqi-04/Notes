# SENet：Squeeze-and-Excitation 通道注意力

> Hu, J., Shen, L., & Sun, G. (2018). Squeeze-and-Excitation Networks. CVPR 2018.
>
> arXiv: https://arxiv.org/abs/1709.01507

## 解决的问题

以往的工作大多在空间维度上做特征增强，通道之间的关系没有得到充分利用；不同通道对应着不同的特征模式，有的通道检测耳朵，有的通道检测毛色，网络应当自适应地增强重要通道、抑制无关通道。

## 核心思想：Squeeze + Excitation

![SE Block结构](images/senet_block.png)

### Squeeze（压缩）

对每个通道做全局平均池化，将整个通道压缩为一个标量：

$$z_c = \frac{1}{H \times W}\sum_{i=1}^{H}\sum_{j=1}^{W} x_c(i,j)$$

H乘W的空间信息由此被压缩到一个数值当中，每个通道都获得了全局感受野。

### Excitation（激励）

用两层全连接网络来学习通道之间的依赖关系：

$$s = \sigma(W_2 \cdot \text{ReLU}(W_1 \cdot z))$$

W1先做降维，reduction ratio通常设为16，W2再把维度升回去；Sigmoid函数将权重限制在0到1之间，最终得到每个通道的重要性权重。

### Scale（重标定）

用学到的权重对原特征图逐通道相乘：

$$\tilde{x}_c = s_c \cdot x_c$$

重要通道的权重接近1，无关通道的权重接近0，特征由此完成了重新标定。

## 特点

该模块即插即用，能够嵌入到任意CNN当中，增加的参数量和计算量都很小，SENet也凭借这一设计拿到了ILSVRC 2017分类任务的冠军。
