import os                                           # 路径操作
import json                                         # 保存/加载训练结果
import time                                         # 计时
import sys                                          # 统一输出流
import pandas as pd                                 # 读csv、存结果表
import numpy as np                                  # 数值计算、随机种子
from PIL import Image                               # 读图
import matplotlib                                   # 画图
matplotlib.use("Agg")                               # 非交互后端，不弹窗直接存文件
import matplotlib.pyplot as plt                     # 画图接口
import torch                                        # PyTorch核心
import torch.nn as nn                               # 神经网络层
import torch.nn.functional as F                     # 函数式接口
from torch.utils.data import Dataset, DataLoader    # 数据加载
from torchvision import transforms                  # 数据增强
from torch.optim.lr_scheduler import CosineAnnealingLR  # 学习率调度
from sklearn.model_selection import train_test_split   # 训练/验证集划分
import matplotlib as mpl
from tqdm import tqdm                               # 终端进度条可视化
from Models import (                                # 从 Models.py 导入模型类
    BaselineResNet, SEResNet, CBAMResNet,
    MultiScaleResNet, FeedbackResNet,
    MultiScaleResNetV2, FeedbackResNetV2
)


class DogBreedDataset(Dataset):
    """数据集类：定义怎么按索引取一张图"""
    def __init__(self, img_dir, df=None, transform=None, is_test=False):
        self.img_dir = img_dir
        self.transform = transform         # 数据增强管线
        self.is_test = is_test             # 是否测试集（测试集无标签）
        if not is_test:
            self.df = df.reset_index(drop=True)     # 重置索引，防止划分后索引不连续
            self.breeds = sorted(df["breed"].unique())  # 所有品种名，排序保证一致
            self.breed2idx = {b: i for i, b in enumerate(self.breeds)}  # 品种名→数字
        else:
            self.img_files = sorted([f for f in os.listdir(img_dir) if f.endswith(".jpg")])

    def __len__(self):
        return len(self.img_files) if self.is_test else len(self.df)

    def __getitem__(self, idx):
        if self.is_test:
            fname = self.img_files[idx]
            image = Image.open(os.path.join(self.img_dir, fname)).convert("RGB")
            if self.transform:
                image = self.transform(image)
            return image, fname.replace(".jpg", "")
        else:
            row = self.df.loc[idx]
            image = Image.open(os.path.join(self.img_dir, f"{row['id']}.jpg")).convert("RGB")
            label = self.breed2idx[row["breed"]]
            if self.transform:
                image = self.transform(image)
            return image, label


def count_params(model):
    """统计模型可训练参数量，单位：百万(M)"""
    return sum(p.numel() for p in model.parameters()) / 1e6


def build_transforms(input_size):
    """根据输入分辨率构建训练和验证的transform"""
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(input_size, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.3),
    ])
    val_resize = int(input_size * 256 / 224)   # 等比例放大resize尺寸
    val_tf = transforms.Compose([
        transforms.Resize(val_resize),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return train_tf, val_tf


def build_tta_transforms(input_size):
    """
    构建TTA（测试时增强）的多组transform
    3个尺度 x (中心裁剪 + 水平翻转) = 6个视角
    """
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    scales = [int(input_size * r) for r in (1.0, 1.14, 1.28)]
    tta_list = []
    for s in scales:
        # 中心裁剪
        tta_list.append(transforms.Compose([
            transforms.Resize(s),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            normalize,
        ]))
        # 水平翻转 + 中心裁剪
        tta_list.append(transforms.Compose([
            transforms.Resize(s),
            transforms.CenterCrop(input_size),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(),
            normalize,
        ]))
    return tta_list


def main():
    # ==================== 配置 ====================
    class Config:
        data_dir = r"D:\Desktop\Science\Paper\Image Processing\Learning\Recognition\dog-breed-identification"
        img_dir = os.path.join(data_dir, "train")
        label_file = os.path.join(data_dir, "labels.csv")
        num_classes = 120
        num_epochs = 40
        lr = 1e-4
        weight_decay = 1e-4
        seed = 42
        save_dir = os.path.join(data_dir, "results")
    cfg = Config()
    os.makedirs(cfg.save_dir, exist_ok=True)        # 创建results文件夹，已存在不报错

    # ==================== GPU ====================
    assert torch.cuda.is_available(), "未检测到GPU"
    device = torch.device("cuda:0")
    torch.backends.cudnn.benchmark = True           # 固定输入尺寸时自动选最快卷积算法
    torch.manual_seed(cfg.seed)                     # 固定CPU随机种子
    np.random.seed(cfg.seed)                        # 固定numpy随机种子
    torch.cuda.manual_seed(cfg.seed)                # 固定GPU随机种子
    print(f"设备: {device} | 显卡: {torch.cuda.get_device_name(0)}")

    # ==================== 数据划分 ====================
    df = pd.read_csv(cfg.label_file)                # 读标签
    train_df, val_df = train_test_split(            # 8:2划分，按品种分层
        df, test_size=0.2, random_state=cfg.seed, stratify=df["breed"])

    # ==================== 模型配置 ====================
    # input_size: 输入分辨率; batch_size: 微batch大小
    # accumulation_steps: 梯度累积步数，effective_batch = batch_size * accumulation_steps
    # freeze_bn: 是否冻结BN层（小batch时BN统计量噪声大，冻结后使用预训练统计量）
    model_configs = {
        "ResNet-50 (Baseline)": {"builder": lambda: BaselineResNet(cfg.num_classes),
                                  "input_size": 224, "batch_size": 32,
                                  "accumulation_steps": 1, "freeze_bn": False},
        "+SE":                   {"builder": lambda: SEResNet(cfg.num_classes),
                                  "input_size": 224, "batch_size": 32,
                                  "accumulation_steps": 1, "freeze_bn": False},
        "+CBAM":                 {"builder": lambda: CBAMResNet(cfg.num_classes),
                                  "input_size": 224, "batch_size": 32,
                                  "accumulation_steps": 1, "freeze_bn": False},
        "+MultiScale":           {"builder": lambda: MultiScaleResNet(cfg.num_classes),
                                  "input_size": 224, "batch_size": 32,
                                  "accumulation_steps": 1, "freeze_bn": False},
        "+Feedback":             {"builder": lambda: FeedbackResNet(cfg.num_classes),
                                  "input_size": 224, "batch_size": 32,
                                  "accumulation_steps": 1, "freeze_bn": False},
        "+MultiScale-v2":        {"builder": lambda: MultiScaleResNetV2(cfg.num_classes),
                                  "input_size": 224, "batch_size": 32,
                                  "accumulation_steps": 1, "freeze_bn": False},
        "+Feedback-v2":          {"builder": lambda: FeedbackResNetV2(cfg.num_classes),
                                  "input_size": 224, "batch_size": 32,
                                  "accumulation_steps": 1, "freeze_bn": False},
        "+HighRes-384":          {"builder": lambda: BaselineResNet(cfg.num_classes),
                                  "input_size": 384, "batch_size": 16,
                                  "accumulation_steps": 1, "freeze_bn": False,
                                  "lr": 7e-5},
    }

    def freeze_bn_layers(model):
        """冻结BN层：设为eval模式，使用预训练的running mean/var，不更新参数"""
        for m in model.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.eval()
                m.weight.requires_grad = False
                m.bias.requires_grad = False

    # ==================== 训练函数  =======================
    def train_one_epoch(model, loader, criterion, optimizer, epoch_idx, num_epochs,
                        accumulation_steps=1, freeze_bn=False):
        model.train()
        if freeze_bn:
            freeze_bn_layers(model)   # 冻结BN，使用预训练统计量
        total_loss, correct, total = 0, 0, 0
        optimizer.zero_grad()
        pbar = tqdm(
            loader,
            desc=f"Train Epoch {epoch_idx + 1:2d}/{num_epochs}",
            leave=True,
            colour='green',
            file=sys.stdout,
            ncols=100,
            mininterval=0.5,
            dynamic_ncols=False
        )

        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(images)
            if isinstance(outputs, tuple):
                loss = 0.3 * criterion(outputs[0], labels) + 0.7 * criterion(outputs[1], labels)
                outputs = outputs[1]
            else:
                loss = criterion(outputs, labels)
            # 梯度累积：loss除以累积步数，反向传播后不立即更新
            (loss / accumulation_steps).backward()
            if (batch_idx + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            correct += outputs.max(1)[1].eq(labels).sum().item()
            total += batch_size

            current_lr = optimizer.param_groups[0]['lr']
            pbar.set_postfix_str(
                f"Loss:{total_loss/total:.4f} Acc:{100.*correct/total:.2f}% LR:{current_lr:.6f}"
            )

        pbar.close()
        return total_loss / total, 100. * correct / total

    # ==================== 验证函数 =======================
    @torch.no_grad()
    def evaluate(model, loader, criterion, epoch_idx, num_epochs):
        model.eval()
        total_loss, c1, c5, total = 0, 0, 0, 0
        pbar = tqdm(
            loader,
            desc=f"Val   Epoch {epoch_idx + 1:2d}/{num_epochs}",
            leave=True,
            colour='green',
            file=sys.stdout,
            ncols=100,
            mininterval=0.5,
            dynamic_ncols=False
        )

        for images, labels in pbar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(images)
            if isinstance(outputs, tuple):
                outputs = outputs[1]

            batch_size = labels.size(0)
            total_loss += criterion(outputs, labels).item() * batch_size
            c1 += outputs.max(1)[1].eq(labels).sum().item()
            c5 += outputs.topk(5, 1)[1].eq(labels.view(-1, 1)).sum().item()
            total += batch_size

            pbar.set_postfix_str(
                f"Loss:{total_loss/total:.4f} Top1:{100.*c1/total:.2f}% Top5:{100.*c5/total:.2f}%"
            )

        pbar.close()
        return total_loss / total, 100. * c1 / total, 100. * c5 / total

    # ==================== TTA验证函数 =======================
    @torch.no_grad()
    def evaluate_tta(model, img_dir, val_df, tta_transforms, batch_size):
        """
        TTA测试时增强：对每张图做多种增强，概率平均后计算Top-1/Top-5
        返回: (tta_top1, tta_top5)
        """
        model.eval()
        # 累加所有TTA视角的softmax概率
        probs_sum = None
        all_labels = []

        for tta_idx, tf in enumerate(tta_transforms):
            dataset = DogBreedDataset(img_dir, val_df, tf)
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                                num_workers=0, pin_memory=True)
            probs_list = []

            pbar = tqdm(loader, desc=f"TTA {tta_idx+1}/{len(tta_transforms)}",
                        leave=False, file=sys.stdout, ncols=80, mininterval=0.5)
            for images, labels in pbar:
                images = images.to(device, non_blocking=True)
                outputs = model(images)
                if isinstance(outputs, tuple):
                    outputs = (outputs[0] + outputs[1]) / 2
                probs = F.softmax(outputs, dim=1)
                probs_list.append(probs.cpu())
                if tta_idx == 0:
                    all_labels.append(labels)
            pbar.close()

            probs_cat = torch.cat(probs_list, dim=0)
            if probs_sum is None:
                probs_sum = probs_cat
            else:
                probs_sum += probs_cat

        probs_avg = probs_sum / len(tta_transforms)
        labels_all = torch.cat(all_labels, dim=0)

        tta_top1 = probs_avg.max(1)[1].eq(labels_all).sum().item() / len(labels_all) * 100
        tta_top5 = probs_avg.topk(5, 1)[1].eq(labels_all.view(-1, 1)).sum().item() / len(labels_all) * 100
        return tta_top1, tta_top5

    # ==================== 结果保存/加载 =======================
    def history_path(name):
        return os.path.join(cfg.save_dir, f"{name}.json")

    def save_history(name, history):
        data = {}
        for k, v in history.items():
            if isinstance(v, np.ndarray):
                data[k] = v.tolist()
            elif isinstance(v, (np.floating, np.integer)):
                data[k] = v.item()
            else:
                data[k] = v
        with open(history_path(name), "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load_history(name):
        with open(history_path(name), "r", encoding="utf-8") as f:
            return json.load(f)

    # ==================== 训练循环 ====================
    all_results = {}
    cmap = mpl.colormaps["tab10"]
    colors = [cmap(i % 10) for i in range(len(model_configs))]

    for name, mcfg in model_configs.items():
        input_size = mcfg["input_size"]
        batch_size = mcfg["batch_size"]

        # 构建该模型专属的数据增强和加载器
        train_tf, val_tf = build_transforms(input_size)
        train_loader = DataLoader(
            DogBreedDataset(cfg.img_dir, train_df, train_tf),
            batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
        val_loader = DataLoader(
            DogBreedDataset(cfg.img_dir, val_df, val_tf),
            batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

        if os.path.exists(history_path(name)):
            print(f"\n[跳过] {name} 已存在结果，直接加载")
            hist = load_history(name)
            # 补跑TTA：旧结果中没有tta_top1字段时，加载权重补跑
            if "tta_top1" not in hist:
                pth_path = os.path.join(cfg.save_dir, f"{name}.pth")
                if os.path.exists(pth_path):
                    print(f"  补跑TTA评估...")
                    model = mcfg["builder"]().to(device)
                    model.load_state_dict(torch.load(pth_path, weights_only=True))
                    tta_transforms = build_tta_transforms(input_size)
                    tta1, tta5 = evaluate_tta(model, cfg.img_dir, val_df,
                                              tta_transforms, batch_size)
                    hist["tta_top1"] = tta1
                    hist["tta_top5"] = tta5
                    hist["input_size"] = input_size
                    save_history(name, hist)
                    print(f"  TTA Top1: {tta1:.2f}% (提升 {tta1 - hist['best_top1']:+.2f}%)")
                    del model
                    torch.cuda.empty_cache()
                else:
                    print(f"  警告: 未找到权重文件 {pth_path}，无法补跑TTA")
                    hist["tta_top1"] = hist["best_top1"]
                    hist["tta_top5"] = hist["best_top5"]
            all_results[name] = hist
            print(f"  Best Top1: {hist['best_top1']:.2f}%"
                  f" | TTA Top1: {hist.get('tta_top1', 'N/A'):.2f}%"
                  if isinstance(hist.get('tta_top1'), (int, float))
                  else f"  Best Top1: {hist['best_top1']:.2f}% | TTA Top1: N/A")
            continue

        print(f"\n{'='*60}\n训练: {name} (输入尺寸: {input_size}x{input_size}, batch: {batch_size})\n{'='*60}")
        model = mcfg["builder"]().to(device)
        criterion = nn.CrossEntropyLoss()
        model_lr = mcfg.get("lr", cfg.lr)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=model_lr, weight_decay=cfg.weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=cfg.num_epochs)

        params_m = count_params(model)
        print(f"参数量: {params_m:.1f}M")

        history = {"train_loss": [], "train_acc": [], "val_loss": [],
                   "val_top1": [], "val_top5": [], "epoch_time": [],
                   "input_size": input_size}
        best_acc = 0.0

        accum_steps = mcfg.get("accumulation_steps", 1)
        freeze_bn = mcfg.get("freeze_bn", False)
        eff_batch = batch_size * accum_steps
        print(f"参数量: {params_m:.1f}M | 有效batch: {eff_batch}"
              f"{' | BN冻结' if freeze_bn else ''}")

        for epoch in range(cfg.num_epochs):
            t0 = time.time()
            tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer,
                                              epoch, cfg.num_epochs,
                                              accumulation_steps=accum_steps,
                                              freeze_bn=freeze_bn)
            vl, v1, v5 = evaluate(model, val_loader, criterion, epoch, cfg.num_epochs)
            scheduler.step()
            dt = time.time() - t0

            history["train_loss"].append(tr_loss)
            history["train_acc"].append(tr_acc)
            history["val_loss"].append(vl)
            history["val_top1"].append(v1)
            history["val_top5"].append(v5)
            history["epoch_time"].append(dt)

            if v1 > best_acc:
                best_acc = v1
                torch.save(model.state_dict(), os.path.join(cfg.save_dir, f"{name}.pth"))

            print()
            print(f"  E{epoch+1:2d}/{cfg.num_epochs} "
                  f"TrLoss {tr_loss:.4f} TrAcc {tr_acc:.1f}% | "
                  f"VaLoss {vl:.4f} Top1 {v1:.2f}% Top5 {v5:.2f}% | {dt:.0f}s")

        history["best_top1"] = best_acc
        history["best_top5"] = max(history["val_top5"])
        history["params_m"] = params_m
        history["avg_epoch_s"] = float(np.mean(history["epoch_time"]))

        # ==================== TTA评估 ====================
        print(f"\n--- {name} TTA评估 ---")
        model.load_state_dict(torch.load(os.path.join(cfg.save_dir, f"{name}.pth"),
                                         weights_only=True))
        tta_transforms = build_tta_transforms(input_size)
        tta_top1, tta_top5 = evaluate_tta(model, cfg.img_dir, val_df,
                                          tta_transforms, batch_size)
        history["tta_top1"] = tta_top1
        history["tta_top5"] = tta_top5
        print(f"  普通验证 Best Top1: {best_acc:.2f}% | TTA Top1: {tta_top1:.2f}%"
              f" | 提升: {tta_top1 - best_acc:+.2f}%")

        save_history(name, history)
        all_results[name] = history
        print(f"  {name} 完成! Best Top1: {best_acc:.2f}%, TTA Top1: {tta_top1:.2f}%")

        del model, optimizer
        torch.cuda.empty_cache()

    # ==================== 画图对比 ========================
    names = list(all_results.keys())
    epochs = range(1, cfg.num_epochs + 1)

    def plot_metric(metric, ylabel, title, fname):
        plt.figure(figsize=(10, 6))
        for i, name in enumerate(names):
            plt.plot(epochs, all_results[name][metric], label=name,
                     color=colors[i], linewidth=1.5)
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(cfg.save_dir, fname), dpi=150, bbox_inches="tight")
        plt.close()

    plot_metric("train_loss", "Train Loss", "Training Loss Comparison", "train_loss.png")
    plot_metric("val_loss",   "Val Loss",   "Validation Loss Comparison", "val_loss.png")
    plot_metric("train_acc",  "Train Acc (%)", "Training Accuracy Comparison", "train_acc.png")
    plot_metric("val_top1",   "Val Top-1 (%)", "Validation Top-1 Comparison", "val_top1.png")

    # ---- 总图 ----
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for i, name in enumerate(names):
        axes[0,0].plot(epochs, all_results[name]["train_loss"], label=name, color=colors[i])
        axes[0,1].plot(epochs, all_results[name]["val_loss"],   label=name, color=colors[i])
        axes[1,0].plot(epochs, all_results[name]["train_acc"],  label=name, color=colors[i])
        axes[1,1].plot(epochs, all_results[name]["val_top1"],   label=name, color=colors[i])
    axes[0,0].set_title("Train Loss"); axes[0,0].legend(fontsize=8); axes[0,0].grid(alpha=0.3)
    axes[0,1].set_title("Val Loss");   axes[0,1].legend(fontsize=8); axes[0,1].grid(alpha=0.3)
    axes[1,0].set_title("Train Acc (%)"); axes[1,0].legend(fontsize=8); axes[1,0].grid(alpha=0.3)
    axes[1,1].set_title("Val Top-1 (%)"); axes[1,1].legend(fontsize=8); axes[1,1].grid(alpha=0.3)
    plt.suptitle("All Models Comparison", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.save_dir, "comparison_all.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # ---- 柱状图（含TTA对比） ----
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    x = np.arange(len(names))
    w = 0.35
    best_accs   = [all_results[n]["best_top1"] for n in names]
    tta_accs    = [all_results[n].get("tta_top1", all_results[n]["best_top1"]) for n in names]
    params      = [all_results[n]["params_m"] for n in names]
    times       = [all_results[n]["avg_epoch_s"] for n in names]

    # Top-1 普通 vs TTA 分组柱状图
    axes[0,0].bar(x - w/2, best_accs, w, label="Best Val Top-1", color="steelblue")
    axes[0,0].bar(x + w/2, tta_accs,  w, label="TTA Top-1", color="coral")
    axes[0,0].set_xticks(x); axes[0,0].set_xticklabels(names, rotation=25, ha="right", fontsize=8)
    axes[0,0].set_title("Best Val Top-1 vs TTA Top-1 (%)"); axes[0,0].legend(fontsize=9)
    axes[0,0].grid(alpha=0.3, axis="y")
    for i, v in enumerate(best_accs):
        axes[0,0].text(i - w/2, v, f"{v:.1f}", ha="center", va="bottom", fontsize=7)
    for i, v in enumerate(tta_accs):
        axes[0,0].text(i + w/2, v, f"{v:.1f}", ha="center", va="bottom", fontsize=7)

    # TTA提升幅度
    tta_gains = [t - b for t, b in zip(tta_accs, best_accs)]
    bar_colors = ["seagreen" if g > 0 else "indianred" for g in tta_gains]
    axes[0,1].bar(x, tta_gains, color=bar_colors)
    axes[0,1].set_xticks(x); axes[0,1].set_xticklabels(names, rotation=25, ha="right", fontsize=8)
    axes[0,1].set_title("TTA Gain over Best Val Top-1 (%)"); axes[0,1].grid(alpha=0.3, axis="y")
    axes[0,1].axhline(y=0, color="black", linewidth=0.8)
    for i, v in enumerate(tta_gains):
        axes[0,1].text(i, v, f"{v:+.2f}", ha="center",
                       va="bottom" if v >= 0 else "top", fontsize=7)

    axes[1,0].bar(x, params, color=colors[:len(names)])
    axes[1,0].set_xticks(x); axes[1,0].set_xticklabels(names, rotation=25, ha="right", fontsize=8)
    axes[1,0].set_title("Parameters (M)"); axes[1,0].grid(alpha=0.3, axis="y")
    for i, v in enumerate(params):
        axes[1,0].text(i, v, f"{v:.1f}", ha="center", va="bottom", fontsize=8)

    axes[1,1].bar(x, times, color=colors[:len(names)])
    axes[1,1].set_xticks(x); axes[1,1].set_xticklabels(names, rotation=25, ha="right", fontsize=8)
    axes[1,1].set_title("Avg Epoch Time (s)"); axes[1,1].grid(alpha=0.3, axis="y")
    for i, v in enumerate(times):
        axes[1,1].text(i, v, f"{v:.0f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(cfg.save_dir, "metrics_bar.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # ==================== 结果表格 ====================
    rows = []
    for name in names:
        r = all_results[name]
        rows.append({
            "Model": name,
            "Input Size": r.get("input_size", 224),
            "Best Top-1 (%)": round(r["best_top1"], 2),
            "Best Top-5 (%)": round(r["best_top5"], 2),
            "TTA Top-1 (%)": round(r.get("tta_top1", r["best_top1"]), 2),
            "TTA Top-5 (%)": round(r.get("tta_top5", r["best_top5"]), 2),
            "TTA Gain (%)": round(r.get("tta_top1", r["best_top1"]) - r["best_top1"], 2),
            "Params (M)": round(r["params_m"], 1),
            "Avg Epoch (s)": round(r["avg_epoch_s"], 1),
        })
    results_df = pd.DataFrame(rows)
    results_df.to_csv(os.path.join(cfg.save_dir, "results.csv"), index=False)

    print(f"\n{'='*60}\n全部完成！结果汇总：\n{'='*60}")
    print(results_df.to_string(index=False))
    print(f"\n所有对比图和结果保存在: {cfg.save_dir}")


if __name__ == "__main__":
    main()
