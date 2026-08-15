import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

def _make_classifier(in_features, num_classes):
    """统一分类头"""
    return nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(in_features, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes)
    )

## ========= 各类注意力模块 ============
class SEBlock(nn.Module):
    """Squeeze-and-Excitation 通道注意力"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ChannelAttention(nn.Module):
    """CBAM 的通道注意力子模块"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        b, c, _, _ = x.size()
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        return self.sigmoid(avg_out + max_out).view(b, c, 1, 1)

class SpatialAttention(nn.Module):
    """CBAM 的空间注意力子模块"""
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        return self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))

class CBAM(nn.Module):
    """CBAM: 通道注意力 + 空间注意力"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention()
    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x

class BaselineResNet(nn.Module):
    """模型0: 纯 ResNet-50 基线"""
    def __init__(self, num_classes=120):
        super().__init__()
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        in_f = self.backbone.fc.in_features
        self.backbone.fc = _make_classifier(in_f, num_classes)
    def forward(self, x):
        return self.backbone(x)

class SEResNet(nn.Module):
    """模型1: ResNet-50 + SE 通道注意力"""
    def __init__(self, num_classes=120):
        super().__init__()
        bb = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.stem = nn.Sequential(bb.conv1, bb.bn1, bb.relu, bb.maxpool)
        self.layer1 = bb.layer1; self.se1 = SEBlock(256)
        self.layer2 = bb.layer2; self.se2 = SEBlock(512)
        self.layer3 = bb.layer3; self.se3 = SEBlock(1024)
        self.layer4 = bb.layer4; self.se4 = SEBlock(2048)
        self.avgpool = bb.avgpool
        self.fc = _make_classifier(2048, num_classes)
    def forward(self, x):
        x = self.stem(x)
        x = self.se1(self.layer1(x))
        x = self.se2(self.layer2(x))
        x = self.se3(self.layer3(x))
        x = self.se4(self.layer4(x))
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)

class CBAMResNet(nn.Module):
    """模型2: ResNet-50 + CBAM 通道+空间注意力"""
    def __init__(self, num_classes=120):
        super().__init__()
        bb = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.stem = nn.Sequential(bb.conv1, bb.bn1, bb.relu, bb.maxpool)
        self.layer1 = bb.layer1; self.cbam1 = CBAM(256)
        self.layer2 = bb.layer2; self.cbam2 = CBAM(512)
        self.layer3 = bb.layer3; self.cbam3 = CBAM(1024)
        self.layer4 = bb.layer4; self.cbam4 = CBAM(2048)
        self.avgpool = bb.avgpool
        self.fc = _make_classifier(2048, num_classes)
    def forward(self, x):
        x = self.stem(x)
        x = self.cbam1(self.layer1(x))
        x = self.cbam2(self.layer2(x))
        x = self.cbam3(self.layer3(x))
        x = self.cbam4(self.layer4(x))
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)

class MultiScaleResNet(nn.Module):
    """模型3: 多尺度特征融合（FPN 方式）"""
    def __init__(self, num_classes=120):
        super().__init__()
        bb = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.stem = nn.Sequential(bb.conv1, bb.bn1, bb.relu, bb.maxpool)
        self.layer1 = bb.layer1
        self.layer2 = bb.layer2
        self.layer3 = bb.layer3
        self.layer4 = bb.layer4
        self.lat3 = nn.Conv2d(1024, 256, 1)
        self.lat2 = nn.Conv2d(512, 256, 1)
        self.lat1 = nn.Conv2d(256, 256, 1)
        self.fuse = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True)
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Dropout(0.5), nn.Linear(256, num_classes)
        )
    def forward(self, x):
        x = self.stem(x)
        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        _ = self.layer4(c3)
        p3 = self.lat3(c3)
        p2 = self.lat2(c2) + F.interpolate(p3, scale_factor=2, mode="nearest")
        p1 = self.lat1(c1) + F.interpolate(p2, scale_factor=2, mode="nearest")
        return self.head(self.fuse(p1))

class FeedbackResNet(nn.Module):
    """模型4: 反馈网络（Rethinking 版，训练时两次前向）"""
    def __init__(self, num_classes=120):
        super().__init__()
        bb = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.stem = nn.Sequential(bb.conv1, bb.bn1, bb.relu, bb.maxpool)
        self.layer1 = bb.layer1; self.layer2 = bb.layer2
        self.layer3 = bb.layer3; self.layer4 = bb.layer4
        self.classifier = nn.Linear(2048, num_classes)
        self.feedback = nn.Sequential(
            nn.Linear(2048 + num_classes, 512), nn.ReLU(inplace=True),
            nn.Linear(512, 256), nn.Sigmoid()
        )
        self.refine = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True)
        )
    def forward(self, x):
        x0 = self.stem(x)
        c1 = self.layer1(x0); c2 = self.layer2(c1)
        c3 = self.layer3(c2); c4 = self.layer4(c3)
        feat = F.adaptive_avg_pool2d(c4, 1).flatten(1)
        logits1 = self.classifier(feat)
        if not self.training:
            return logits1
        gate = self.feedback(torch.cat([feat, logits1.detach()], dim=1))
        c1r = self.refine(c1 * gate.view(c1.size(0), -1, 1, 1))
        c4n = self.layer4(self.layer3(self.layer2(c1r)))
        logits2 = self.classifier(F.adaptive_avg_pool2d(c4n, 1).flatten(1))
        return logits1, logits2

# class SEMultiScaleResNet(nn.Module):
#     """模型5: SE + 多尺度组合（单模块结果出来后再决定是否训练）"""
#     def __init__(self, num_classes=120):
#         super().__init__()
#         bb = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
#         self.stem = nn.Sequential(bb.conv1, bb.bn1, bb.relu, bb.maxpool)
#         self.layer1 = bb.layer1; self.se1 = SEBlock(256)
#         self.layer2 = bb.layer2; self.se2 = SEBlock(512)
#         self.layer3 = bb.layer3; self.se3 = SEBlock(1024)
#         self.layer4 = bb.layer4
#         self.lat3 = nn.Conv2d(1024, 256, 1)
#         self.lat2 = nn.Conv2d(512, 256, 1)
#         self.lat1 = nn.Conv2d(256, 256, 1)
#         self.fuse = nn.Sequential(
#             nn.Conv2d(256, 256, 3, padding=1),
#             nn.BatchNorm2d(256), nn.ReLU(inplace=True)
#         )
#         self.head = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1), nn.Flatten(),
#             nn.Dropout(0.5), nn.Linear(256, num_classes)
#         )
#     def forward(self, x):
#         x = self.stem(x)
#         c1 = self.se1(self.layer1(x))
#         c2 = self.se2(self.layer2(c1))
#         c3 = self.se3(self.layer3(c2))
#         p3 = self.lat3(c3)
#         p2 = self.lat2(c2) + F.interpolate(p3, scale_factor=2, mode="nearest")
#         p1 = self.lat1(c1) + F.interpolate(p2, scale_factor=2, mode="nearest")
#         return self.head(self.fuse(p1))