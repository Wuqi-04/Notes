import os                                           # 路径操作
import json                                         # 保存/加载训练结果
import time                                         # 计时
import sys                                          # 新增：统一输出流
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
from Models import (                                # 从 models.py 导入模型类
    BaselineResNet, SEResNet, CBAMResNet,
    MultiScaleResNet, FeedbackResNet
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


def main():
    # ==================== 配置 ====================
    class Config:
        data_dir = r"D:\Desktop\Science\Paper\Image Processing\Learning\Recognition\dog-breed-identification"
        img_dir = os.path.join(data_dir, "train")
        label_file = os.path.join(data_dir, "labels.csv")
        num_classes = 120
        batch_size = 32
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

    # ==================== 数据增强与加载 ====================
    train_tf = transforms.Compose([                 # 训练集增强（带随机性）
        transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.3),
    ])
    val_tf = transforms.Compose([                   # 验证集预处理（无随机性）
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    df = pd.read_csv(cfg.label_file)                # 读标签
    train_df, val_df = train_test_split(            # 8:2划分，按品种分层
        df, test_size=0.2, random_state=cfg.seed, stratify=df["breed"])
    train_loader = DataLoader(                      # 训练集加载器
        DogBreedDataset(cfg.img_dir, train_df, train_tf),
        batch_size=cfg.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(                        # 验证集加载器
        DogBreedDataset(cfg.img_dir, val_df, val_tf),
        batch_size=cfg.batch_size, shuffle=False, num_workers=0, pin_memory=True)

    # ==================== 训练函数  =======================
    def train_one_epoch(model, loader, criterion, optimizer, epoch_idx):
        model.train()
        total_loss, correct, total = 0, 0, 0
        pbar = tqdm(
            loader,
            desc=f"Train Epoch {epoch_idx + 1:2d}/{cfg.num_epochs}",
            leave=True,
            colour='green',
            file=sys.stdout,        # 和print统一输出流，彻底解决错位
            ncols=100,              # 强制固定总宽度，防止超宽折行
            mininterval=0.5,        # 降低刷新频率，减少终端压力
            dynamic_ncols=False
        )

        for images, labels in pbar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad()
            outputs = model(images)
            if isinstance(outputs, tuple):
                loss = 0.3 * criterion(outputs[0], labels) + 0.7 * criterion(outputs[1], labels)
                outputs = outputs[1]
            else:
                loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

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
    def evaluate(model, loader, criterion, epoch_idx):
        model.eval()
        total_loss, c1, c5, total = 0, 0, 0, 0
        pbar = tqdm(
            loader,
            desc=f"Val   Epoch {epoch_idx + 1:2d}/{cfg.num_epochs}",
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

    # ==================== 模型字典 ====================
    model_builders = {
        "ResNet-50 (Baseline)": lambda: BaselineResNet(cfg.num_classes),
        "+SE":                  lambda: SEResNet(cfg.num_classes),
        "+CBAM":                lambda: CBAMResNet(cfg.num_classes),
        "+MultiScale":          lambda: MultiScaleResNet(cfg.num_classes),
        "+Feedback":            lambda: FeedbackResNet(cfg.num_classes),
    }

    # ==================== 训练循环 ====================
    all_results = {}
    cmap = mpl.colormaps["tab10"]
    colors = [cmap(i % 10) for i in range(len(model_builders))]

    for name, builder in model_builders.items():
        if os.path.exists(history_path(name)):
            print(f"\n[跳过] {name} 已存在结果，直接加载")
            all_results[name] = load_history(name)
            print(f"  Best Top1: {all_results[name]['best_top1']:.2f}%")
            continue

        print(f"\n{'='*60}\n训练: {name}\n{'='*60}")
        model = builder().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=cfg.num_epochs)

        params_m = count_params(model)
        print(f"参数量: {params_m:.1f}M")

        history = {"train_loss": [], "train_acc": [], "val_loss": [],
                   "val_top1": [], "val_top5": [], "epoch_time": []}
        best_acc = 0.0

        for epoch in range(cfg.num_epochs):
            t0 = time.time()
            tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, epoch)
            vl, v1, v5 = evaluate(model, val_loader, criterion, epoch)
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

            print()  # 空行分隔，避免和进度条收尾粘连
            print(f"  E{epoch+1:2d}/{cfg.num_epochs} "
                  f"TrLoss {tr_loss:.4f} TrAcc {tr_acc:.1f}% | "
                  f"VaLoss {vl:.4f} Top1 {v1:.2f}% Top5 {v5:.2f}% | {dt:.0f}s")

        history["best_top1"] = best_acc
        history["best_top5"] = max(history["val_top5"])
        history["params_m"] = params_m
        history["avg_epoch_s"] = float(np.mean(history["epoch_time"]))

        save_history(name, history)
        all_results[name] = history
        print(f"  {name} 完成! Best Top1: {best_acc:.2f}% (结果已保存)")

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

    # ---- 柱状图 ----
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    x = np.arange(len(names))
    best_accs = [all_results[n]["best_top1"] for n in names]
    params    = [all_results[n]["params_m"] for n in names]
    times     = [all_results[n]["avg_epoch_s"] for n in names]

    axes[0].bar(x, best_accs, color=colors[:len(names)])
    axes[0].set_xticks(x); axes[0].set_xticklabels(names, rotation=25, ha="right", fontsize=8)
    axes[0].set_title("Best Val Top-1 (%)"); axes[0].grid(alpha=0.3, axis="y")
    for i, v in enumerate(best_accs):
        axes[0].text(i, v, f"{v:.1f}", ha="center", va="bottom", fontsize=8)

    axes[1].bar(x, params, color=colors[:len(names)])
    axes[1].set_xticks(x); axes[1].set_xticklabels(names, rotation=25, ha="right", fontsize=8)
    axes[1].set_title("Parameters (M)"); axes[1].grid(alpha=0.3, axis="y")
    for i, v in enumerate(params):
        axes[1].text(i, v, f"{v:.1f}", ha="center", va="bottom", fontsize=8)

    axes[2].bar(x, times, color=colors[:len(names)])
    axes[2].set_xticks(x); axes[2].set_xticklabels(names, rotation=25, ha="right", fontsize=8)
    axes[2].set_title("Avg Epoch Time (s)"); axes[2].grid(alpha=0.3, axis="y")
    for i, v in enumerate(times):
        axes[2].text(i, v, f"{v:.0f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(cfg.save_dir, "metrics_bar.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # ==================== 结果表格 ====================
    rows = []
    for name in names:
        r = all_results[name]
        rows.append({
            "Model": name,
            "Best Top-1 (%)": round(r["best_top1"], 2),
            "Best Top-5 (%)": round(r["best_top5"], 2),
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
