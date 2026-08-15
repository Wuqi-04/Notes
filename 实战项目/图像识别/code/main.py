import os
# os 操作系统接口，用来拼路径、读文件夹、建目录。用 os.path.join() 拼路径而不是手写斜杠，因为 Windows 用 \、Linux/Mac 用 /，这个函数自动适配。
import time
import pandas as pd
# pandas 处理表格的标准库，用于读取csv文件
import numpy as np
from PIL import Image
# 将jpg图片读取为内存对象后转化为Tensor
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
# Dataset：定义 "怎么取第 i 个样本"   DataLoader：把单个样本打包成 batch、打乱、多进程加载
from torchvision import transforms, models
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import train_test_split
# 把数据切成训练集和验证集。stratify 参数保证每个类别比例一致

def main():

# ============ 配置区 ============
    class Config:
        data_dir = r"D:\Desktop\Science\Paper\Image Processing\Learning\Recognition\dog-breed-identification"
        img_dir = os.path.join(data_dir, "train")
        test_dir = os.path.join(data_dir, "test")
        label_file = os.path.join(data_dir, "labels.csv")
        num_classes = 120
        batch_size = 32
        num_epochs = 40
        # 用预训练模型时学习率要小（1e-4 级别），太大会破坏预训练学到的特征，从头训练才用 1e-3 或 1e-2
        lr = 1e-4
        # 权重衰减（L2 正则化），防止过拟合，让参数不要太大，限制模型复杂度。
        weight_decay = 1e-4
        # 将随机种子固定，方便对后续不同模型进行对比
        seed = 42
        # 最佳模型保存
        save_path = os.path.join(data_dir, "best_model.pth")

    cfg = Config()

# ============ 强制使用GPU ============
    assert torch.cuda.is_available(), "未检测到GPU！请安装CUDA版PyTorch" # 断言检查。如果没 GPU 直接报错退出
    device = torch.device("cuda:0")   # 指定第一块GPU
    torch.backends.cudnn.benchmark = True  # 输入尺寸固定时，可利用cuDNN自动调优，可以提高收敛速度
    print(f"使用设备: {device}")
    print(f"显卡型号: {torch.cuda.get_device_name(0)}")
    print(f"显存总量: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # 固定随机种子，以便后续对不同模型的效果进行横向对比
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)

# ============ 数据集 ============
    class DogBreedDataset(Dataset):

        ## df 包含id和breed的DataFrame （训练/验证集使用）
        def __init__(self, img_dir, df=None, transform=None, is_test=False):
            self.img_dir = img_dir
            self.transform = transform
            self.is_test = is_test
            if not is_test:
                ## 训练集模式
                self.df = df.reset_index(drop=True)  # 划分训练集和验证集后重置索引

                ## 将品种名字映射为数字
                self.breeds = sorted(df["breed"].unique())
                self.breed2idx = {b: i for i, b in enumerate(self.breeds)}
            else:
                ## 测试集模式，列出文件所有的jpg
                self.img_files = sorted([f for f in os.listdir(img_dir) if f.endswith(".jpg")])

        def __len__(self):
            if self.is_test:
                return len(self.img_files)
            return len(self.df)

        ## 根据索引返回一个样本
        def __getitem__(self, idx):
            ## 测试集将图像转化成rgb
            if self.is_test:
                fname = self.img_files[idx]
                img_path = os.path.join(self.img_dir, fname)
                image = Image.open(img_path).convert("RGB")
                if self.transform:
                    image = self.transform(image)
                return image, fname.replace(".jpg", "")
            ## 验证集根据id拼路径，查询品种名转成标签
            else:
                img_id = self.df.loc[idx, "id"]
                img_path = os.path.join(self.img_dir, f"{img_id}.jpg")
                image = Image.open(img_path).convert("RGB")
                breed = self.df.loc[idx, "breed"]
                label = self.breed2idx[breed]
                ## 数据增强
                if self.transform:
                    image = self.transform(image)
                return image, label

        ## 数据增强
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.6, 1.0)), # 随机裁剪
        transforms.RandomHorizontalFlip(), # 水平翻转
        transforms.RandomRotation(15), # 随机旋转
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.1), # 随机调整亮度/对比度等，模拟不同环境照片
        transforms.ToTensor(), # 转换成tensor
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), # 标准化通道
        transforms.RandomErasing(p=0.3), # 随机擦除，防止模型靠单一特征识别，防止过拟合
    ])

    ## 验证集不做随机增强，仅做预处理
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    ## 数据加载
    df = pd.read_csv(cfg.label_file)
    ##  stratify=df["breed"]可以保证每个品种在训练集和验证集比例相同
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=cfg.seed, stratify=df["breed"]) # 80%训练，20%验证
    train_dataset = DogBreedDataset(cfg.img_dir, df=train_df, transform=train_transform)
    val_dataset = DogBreedDataset(cfg.img_dir, df=val_df, transform=val_transform)
    test_dataset = DogBreedDataset(cfg.test_dir, transform=val_transform, is_test=True)

    ## pin_memory 把数据锁在内存页（高速页表） num_workers多进程
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=0, pin_memory=True)

    # ============ 模型 ============
    class DogBreedResNet(nn.Module):
        ## 选用resnet50作为backbone，狗的分类任务共120类，pretrain加载ImageNet预训练权重
        def __init__(self, num_classes=120, pretrained=True):
            super().__init__()
            # weights = ...DEFAULT 加载与训练权重
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            self.backbone = models.resnet50(weights=weights)
            # 因为resnet50输出是(2048，1000)一千类，因此要将最后一层全连接层替换为输出120类
            in_features = self.backbone.fc.in_features
            # 将一层替换为两层分类头（小数据集上效果更好）
            self.backbone.fc = nn.Sequential(
                nn.Dropout(0.5), # 随机失活，防止过拟合（小数据容易过拟合）
                nn.Linear(in_features, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(512, num_classes)
            )

        ## 前向传播，[B, 3, 224, 224]→[B, 120] 120为各个品种的得分率（非概率）
        def forward(self, x):
            return self.backbone(x)

    model = DogBreedResNet(num_classes=cfg.num_classes).to(device)

    # ============ 损失函数和优化器 ============
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    ## 余弦退火学习率，按余弦曲线减小至。前期用大学习率收敛，后期小学习率微调
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.num_epochs)

    # ============ 训练和验证 ============
    def train_one_epoch(model, loader, criterion, optimizer):
        model.train() # 设置为训练模式
        total_loss = 0
        correct = 0
        total = 0
        for images, labels in loader:
            # 用non_blocking配合pin_memory异步传输，即GPU在算上一批时，CPU传下一批，提高速率
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad() # 必须进行梯度清零
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step() # 更新梯度
            total_loss += loss.item() * labels.size(0)
            _, predicted = outputs.max(1)  ## 沿类别维度取最大值，返回(最大值，索引)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
        return total_loss / total, 100. * correct / total

    ## 验证函数
    @torch.no_grad() # 禁用梯度计算
    def evaluate(model, loader, criterion):
        model.eval() # 验证模式，关闭神经元
        total_loss = 0
        correct_top1 = 0
        correct_top5 = 0 # 正确类别出现在概率最高的5个里就算正确
        total = 0
        for images, labels in loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * labels.size(0)
            _, pred = outputs.max(1)
            correct_top1 += pred.eq(labels).sum().item()
            _, top5 = outputs.topk(5, 1, True, True)
            correct_top5 += top5.eq(labels.view(-1, 1).expand_as(top5)).sum().item()
            total += labels.size(0)
        return total_loss / total, 100. * correct_top1 / total, 100. * correct_top5 / total

    # ============ 主训练循环 ============

    best_acc = 0.0
    print(f"\n训练集: {len(train_dataset)} 张, 验证集: {len(val_dataset)} 张, 测试集: {len(test_dataset)} 张")
    print(f"开始训练...\n")

    for epoch in range(cfg.num_epochs):
        start_time = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_top1, val_top5 = evaluate(model, val_loader, criterion)
        scheduler.step() ## 每个epoch后面都要更新一次梯度
        epoch_time = time.time() - start_time

        # 打印当前显存占用
        mem_used = torch.cuda.max_memory_allocated() / 1024**3
        torch.cuda.reset_peak_memory_stats()

        print(f"Epoch [{epoch+1:2d}/{cfg.num_epochs}] "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} Top1: {val_top1:.2f}% Top5: {val_top5:.2f}% | "
              f"LR: {optimizer.param_groups[0]['lr']:.6f} | "
              f"Time: {epoch_time:.1f}s | 显存: {mem_used:.2f}GB")

        if val_top1 > best_acc: ## 保存验证精度最高的模型
            best_acc = val_top1
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_acc": best_acc,
                "breeds": train_dataset.breeds,
            }, cfg.save_path)
            print(f"  -> 保存最佳模型 (Val Acc: {best_acc:.2f}%)")

    print(f"\n训练完成！最佳验证准确率: {best_acc:.2f}%")

    # ============ 测试集推理 ============
    print("\n开始测试集推理...")

    ##加载正确率最高的模型，map_location确保模型加载到正确设备上
    checkpoint = torch.load(cfg.save_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    breeds = checkpoint["breeds"]
    model.eval()

    results = []
    with torch.no_grad():
        for images, img_ids in test_loader:
            images = images.to(device, non_blocking=True)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1).cpu().numpy()
            # 推理时，需要用Softmax将logits转化成概率（训练时CrossEntropyLoss自带，不用加）
            for i, img_id in enumerate(img_ids):
                row = {"id": img_id}
                for j, b in enumerate(breeds):
                    row[b] = probs[i][j]
                results.append(row)

    submission = pd.DataFrame(results)
    sub_path = os.path.join(cfg.data_dir, "submission.csv")
    submission.to_csv(sub_path, index=False)
    print(f"提交文件已保存到: {sub_path}")

if __name__ == '__main__':
    main()