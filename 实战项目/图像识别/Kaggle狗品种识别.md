# Kaggle狗品种识别

> 竞赛地址：https://www.kaggle.com/competitions/dog-breed-identification

## 任务简介

细粒度图像分类任务，识别120种犬种。训练集约10,222张标注图片，测试集约10,357张。

### 难点

1. **类间方差小**：不同犬种外观相似（如哈士奇和阿拉斯加）
2. **类内方差大**：同一犬种毛色、姿态、角度差异大
3. **数据量小**：平均每类约85张，从头训练不可行
4. 必须使用 ImageNet 预训练权重做迁移学习

## 数据结构

```
dog-breed-identification/
├── train/          # 训练图片
├── test/           # 测试图片
├── labels.csv      # id, breed
└── sample_submission.csv
```

## 方法设计

以 ResNet-50 为基线，对比四种改进模块：

| 模型 | 方法 | 说明 |
|------|------|------|
| BaselineResNet | ResNet-50 | 纯基线 |
| SEResNet | + SE通道注意力 | 自动关注重要特征通道 |
| CBAMResNet | + CBAM通道空间注意力 | 同时关注"什么"和"哪里" |
| MultiScaleResNet | + FPN多尺度融合 | 融合浅层细节和深层语义 |
| FeedbackResNet | + 反馈迭代修正 | 二次前向传播修正预测 |

### 代码结构

- `models.py`：所有模型定义
- `main.py`：训练、验证、测试、结果可视化
- 结果缓存：每个模型训练完保存JSON，再次运行自动跳过已训练模型

### 训练配置

- 预训练：ImageNet ResNet-50
- 优化器：AdamW (lr=1e-4, weight_decay=1e-4)
- 学习率调度：CosineAnnealingLR
- 数据增强：RandomResizedCrop, Flip, Rotation, ColorJitter, RandomErasing
- Epochs：40
- Batch size：32
- GPU：RTX 3080 Laptop

## 实验结果

（训练完成后补充）

### 评估指标

- Top-1 Accuracy（主指标）
- Top-5 Accuracy
- 参数量（M）
- 推理时间（ms）

## 结论

（训练完成后补充）

### 预期分析

- SE/CBAM 注意力机制：性价比最高，提升0.5-2%
- 多尺度融合：对细粒度分类有帮助，提升0.5-1.5%
- 反馈网络：在干净静态图像上提升有限（0-1%），如实分析
- 数据增强和输入分辨率的提升往往大于改网络结构
