# 创建Tensor

## 从numpy导入

```python
import numpy as np
import torch

data = np.ones([2, 3])
t = torch.from_numpy(data)
# tensor([[1., 1., 1.],
#         [1., 1., 1.]], dtype=torch.float64)
```

注意：numpy导入的float型实际上是double型（float64）。

## 从list导入

```python
torch.tensor([1.0, 2.0, 3.0])
```

注意区分：
- `torch.Tensor(shape)` / `torch.FloatTensor(shape)`：接收shape（维度大小）
- `torch.tensor([...])`：直接接收数据（小写t）
- 大写接收维度，小写接收数据；若大写传入list则也视为接收数据

## 未初始化数据

```python
torch.empty([2, 3])           # 注意shape有[]
torch.FloatTensor(2, 3, 4)    # 用维度创建
torch.IntTensor(2, 3)
```

- tensor默认类型是float
- 未初始化数据内实际上存在数据，但是是随机值（不干净）
- 使用未初始化数据前必须覆盖写入，否则可能引入噪声

## 初始化数据

### 随机分布

```python
torch.rand(3, 4)          # 均匀分布 [0,1)
torch.rand_like(a)        # 读取a的维度，用rand随机化
torch.randint(1, 10, [3, 4])  # 整数均匀分布 [min, max)
```

### 正态分布

```python
torch.randn(3, 4)         # 标准正态 N(0,1)，方差为1

# 自定义均值和方差
torch.normal(
    mean=torch.full([10], 0),       # 均值全为0
    std=torch.arange(1, 0, -0.1)    # 方差从1递减到0.1
)
```

### 其他初始化方式

```python
torch.full([2, 3], 7)     # 全部填充为7
torch.arange(0, 10, 2)    # 等差数列 [0,2,4,6,8]，不包含end
torch.linspace(0, 1, 5)   # 等分数列 [0, 0.25, 0.5, 0.75, 1]
torch.logspace(0, 2, 3)   # 对数等分
torch.ones(2, 3)          # 全1
torch.zeros(2, 3)         # 全0
torch.eye(3)              # 单位矩阵
```

### randperm（随机打散）

```python
torch.randperm(10)  # 生成0~9的随机排列
```

- 等价于对 range(n) 做 shuffle
- 防止训练时学习到数据的顺序规律（如数据按时间排列）
- 一般用作索引种子，不同tensor用同一个idx保持对应关系
