# 创建Tensor

## 从numpy导入

将数据从numpy导入时，转化后的数据是一一对应的。需要注意的是，numpy导入的float型实际上是double型。

## 从list导入

从list导入数据时需要注意区分几种写法：torch.Tensor(shape)接收的是shape，torch.FloatTensor(shape)同样接收shape，torch.tensor([...])则直接接收数据。小写的tensor直接接收数据，大写的Tensor接收数据维度，如果传入的是[...]即list，则说明是接收数据。

## 未初始化数据

使用torch.empty([shape])可以创建未初始化数据，torch.FloatTensor(d1,d2,d3)和torch.IntTensor(d1,d2,d3)是用维度来创建未初始化参数。tensor的默认类型是float型。需要注意此处的shape带有[]，未初始化数据内实际上是存在数据的，只不过是random的随机值。

## 初始化数据

### 随机分布

rand/rand_like和randint用于生成随机分布数据。torch.rand(...)生成随机数，torch.rand_like(a)会读取a的维度然后用rand随机化，torch.randint(min,max,[shape])需要输入最小值和最大值，取值范围包括min但不包括max。

### 正态分布

randn默认生成N(0,1)即方差为1的正态分布。如果需要自定义方差，可以使用torch.normal，例如mean=torch.full([10],0)、std=torch.arange(1,0,-0.1)，此时得到的是dimension为1、长度为10的tensor，还需要reshape成想要的tensor维度。

### full

torch.full([shape],数值)将对应维度的tensor全部赋值为该数值。

### arange和range

torch.arange(a,b,c)生成一个不包含b的tensor，是差为c的等差数列。默认c为1，同时生成的是一维tensor。tensor里面不常用range，可以忽略。

### linspace和logspace

linspace和logspace用于生成线性间距和对数间距的张量。

### ones、zeros和eye

torch.ones、torch.zeros和torch.eye(shape)分别生成全1、全0的tensor以及单位矩阵tensor。torch.eye(数值)生成一个a乘a的单位矩阵tensor。

### randperm随机打散

randperm等价于对range(n)做了一次shuffle，可以防止在训练神经网络时学习到顺序规律，比如数据按照时间等顺序排列时，模型可能会学习到这个特征。torch.randperm(数值)生成0到a的索引，不包括a，相当于按行打乱。一般用作idx索引种子，不同的tensor使用的idx应该保持一致。
