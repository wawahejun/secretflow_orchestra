#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Orchestra模型定义
包含ResNet骨干网络、投影网络、聚类组件等
"""

import math
import numpy as np
from typing import Optional, Tuple, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class BasicBlock(nn.Module):
    """ResNet基本块"""
    expansion = 1
    
    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    """ResNet瓶颈块"""
    expansion = 4
    
    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    """ResNet骨干网络"""
    
    def __init__(self, block, num_blocks: List[int], num_classes: int = 0, zero_init_residual: bool = False):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.num_classes = num_classes
        
        # CIFAR-10适配：使用3x3卷积，stride=1，无最大池化
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 分类头（可选）
        if num_classes > 0:
            self.fc = nn.Linear(512 * block.expansion, num_classes)
        
        self.output_dim = 512 * block.expansion
        
        # 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        # Zero-initialize the last BN in each residual branch
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)
    
    def _make_layer(self, block, planes: int, num_blocks: int, stride: int):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    
    def forward(self, x, return_features: bool = False):
        x = F.relu(self.bn1(self.conv1(x)))
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        features = x.view(x.size(0), -1)
        
        if self.num_classes > 0:
            x = self.fc(features)
            if return_features:
                return x, features
            return x
        else:
            return features


def ResNet18(num_classes: int = 0, **kwargs):
    """ResNet-18模型"""
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, **kwargs)


def ResNet34(num_classes: int = 0, **kwargs):
    """ResNet-34模型"""
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, **kwargs)


def ResNet50(num_classes: int = 0, **kwargs):
    """ResNet-50模型"""
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, **kwargs)


class ProjectionMLP(nn.Module):
    """投影MLP网络"""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int = 2):
        super(ProjectionMLP, self).__init__()
        
        if num_layers == 1:
            self.layers = nn.Linear(input_dim, output_dim)
        else:
            layers = []
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            
            for _ in range(num_layers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.ReLU(inplace=True))
            
            layers.append(nn.Linear(hidden_dim, output_dim))
            self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layers(x)


class RotationPredictor(nn.Module):
    """旋转预测网络（用于抗退化）"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 512):
        super(RotationPredictor, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 4)  # 4个旋转角度：0, 90, 180, 270
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class SinkhornKnopp(nn.Module):
    """Sinkhorn-Knopp算法实现等大小聚类"""
    
    def __init__(self, num_iters: int = 3, epsilon: float = 0.05):
        super(SinkhornKnopp, self).__init__()
        self.num_iters = num_iters
        self.epsilon = epsilon
    
    def forward(self, Q: torch.Tensor) -> torch.Tensor:
        """
        Args:
            Q: 相似度矩阵，形状为 (batch_size, num_clusters)
        
        Returns:
            等大小聚类分配矩阵
        """
        # 应用温度缩放
        Q = torch.exp(Q / self.epsilon)
        
        # Sinkhorn-Knopp迭代
        for _ in range(self.num_iters):
            # 行归一化
            Q = Q / Q.sum(dim=1, keepdim=True)
            # 列归一化
            Q = Q / Q.sum(dim=0, keepdim=True)
        
        return Q


class OrchestraModel(nn.Module):
    """Orchestra模型主体"""
    
    def __init__(self,
                 backbone: nn.Module,
                 projection_dim: int = 128,
                 num_local_clusters: int = 16,
                 num_global_clusters: int = 128,
                 memory_size: int = 128,
                 temperature: float = 0.1,
                 ema_decay: float = 0.996,
                 sinkhorn_iterations: int = 3,
                 sinkhorn_epsilon: float = 0.05):
        super(OrchestraModel, self).__init__()
        
        self.backbone = backbone
        self.projection_dim = projection_dim
        self.num_local_clusters = num_local_clusters
        self.num_global_clusters = num_global_clusters
        self.memory_size = memory_size
        self.temperature = temperature
        self.ema_decay = ema_decay
        
        # 投影网络
        self.projector = ProjectionMLP(
            input_dim=backbone.output_dim,
            hidden_dim=backbone.output_dim,
            output_dim=projection_dim
        )
        
        # 目标网络（通过EMA更新）
        self.target_backbone = self._create_target_network(backbone)
        self.target_projector = self._create_target_network(self.projector)
        
        # 旋转预测器（抗退化）
        self.rotation_predictor = RotationPredictor(backbone.output_dim)
        
        # Sinkhorn-Knopp算法
        self.sinkhorn = SinkhornKnopp(sinkhorn_iterations, sinkhorn_epsilon)
        
        # 聚类中心
        self.register_buffer('global_centroids', torch.randn(num_global_clusters, projection_dim))
        self.register_buffer('local_centroids', torch.randn(num_local_clusters, projection_dim))
        
        # 投影内存
        self.register_buffer('projection_memory', torch.randn(memory_size, projection_dim))
        self.register_buffer('memory_ptr', torch.zeros(1, dtype=torch.long))
        
        # 归一化聚类中心
        self.global_centroids = F.normalize(self.global_centroids, dim=1)
        self.local_centroids = F.normalize(self.local_centroids, dim=1)
        self.projection_memory = F.normalize(self.projection_memory, dim=1)
    
    def _create_target_network(self, network: nn.Module) -> nn.Module:
        """创建目标网络"""
        import copy
        # 使用深拷贝创建目标网络
        target_network = copy.deepcopy(network)
        
        # 冻结目标网络参数
        for param in target_network.parameters():
            param.requires_grad = False
        
        return target_network
    
    @torch.no_grad()
    def update_target_networks(self):
        """使用EMA更新目标网络"""
        # 更新目标骨干网络
        for param_q, param_k in zip(self.backbone.parameters(), self.target_backbone.parameters()):
            param_k.data = param_k.data * self.ema_decay + param_q.data * (1.0 - self.ema_decay)
        
        # 更新目标投影网络
        for param_q, param_k in zip(self.projector.parameters(), self.target_projector.parameters()):
            param_k.data = param_k.data * self.ema_decay + param_q.data * (1.0 - self.ema_decay)
    
    @torch.no_grad()
    def update_memory(self, projections: torch.Tensor):
        """更新投影内存"""
        batch_size = projections.shape[0]
        ptr = int(self.memory_ptr)
        
        # 创建新的内存副本以避免就地操作
        new_memory = self.projection_memory.clone()
        
        if ptr + batch_size <= self.memory_size:
            new_memory[ptr:ptr + batch_size] = projections
            ptr = (ptr + batch_size) % self.memory_size
        else:
            # 处理内存溢出
            remaining = self.memory_size - ptr
            new_memory[ptr:] = projections[:remaining]
            new_memory[:batch_size - remaining] = projections[remaining:]
            ptr = batch_size - remaining
        
        # 更新缓冲区
        self.projection_memory.copy_(new_memory)
        self.memory_ptr[0] = ptr
    
    def local_clustering(self, projections: torch.Tensor) -> torch.Tensor:
        """本地聚类"""
        # 计算与本地聚类中心的相似度
        similarities = torch.mm(projections, self.local_centroids.t()) / self.temperature
        
        # 应用Sinkhorn-Knopp算法
        assignments = self.sinkhorn(similarities)
        
        return assignments
    
    def global_clustering(self, projections: torch.Tensor) -> torch.Tensor:
        """全局聚类"""
        # 计算与全局聚类中心的相似度
        similarities = torch.mm(projections, self.global_centroids.t()) / self.temperature
        
        # 应用Sinkhorn-Knopp算法
        assignments = self.sinkhorn(similarities)
        
        return assignments
    
    def contrastive_loss(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """对比损失"""
        # 归一化
        q = F.normalize(q, dim=1)
        k = F.normalize(k, dim=1)
        
        # 正样本：同一图像的不同增强
        pos_logits = torch.sum(q * k, dim=1, keepdim=True) / self.temperature
        
        # 负样本：与内存中的投影对比（使用detach避免梯度问题）
        neg_logits = torch.mm(q, self.projection_memory.detach().t()) / self.temperature
        
        # 拼接正负样本
        logits = torch.cat([pos_logits, neg_logits], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
        
        loss = F.cross_entropy(logits, labels)
        return loss
    
    def clustering_loss(self, assignments_q: torch.Tensor, assignments_k: torch.Tensor) -> torch.Tensor:
        """聚类损失"""
        # 交叉熵损失
        loss = -torch.mean(torch.sum(assignments_q * torch.log(assignments_k + 1e-8), dim=1))
        return loss
    
    def rotation_loss(self, features: torch.Tensor, rotation_labels: torch.Tensor) -> torch.Tensor:
        """旋转预测损失（抗退化）"""
        rotation_pred = self.rotation_predictor(features)
        loss = F.cross_entropy(rotation_pred, rotation_labels)
        return loss
    
    def forward(self, x1: torch.Tensor, x2: torch.Tensor, x3: Optional[torch.Tensor] = None, 
                rotation_labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Args:
            x1: 第一个增强视图
            x2: 第二个增强视图
            x3: 旋转视图（可选）
            rotation_labels: 旋转标签（可选）
        
        Returns:
            包含各种损失的字典
        """
        batch_size = x1.shape[0]
        
        # 在线网络前向传播
        features_q = self.backbone(x1)
        projections_q = F.normalize(self.projector(features_q), dim=1)
        
        # 目标网络前向传播（无梯度）
        with torch.no_grad():
            features_k = self.target_backbone(x2)
            projections_k = F.normalize(self.target_projector(features_k), dim=1)
        
        # 聚类分配
        local_assignments_q = self.local_clustering(projections_q)
        global_assignments_q = self.global_clustering(projections_q)
        
        with torch.no_grad():
            local_assignments_k = self.local_clustering(projections_k)
            global_assignments_k = self.global_clustering(projections_k)
        
        # 计算损失
        losses = {}
        
        # 对比损失
        losses['contrastive'] = self.contrastive_loss(projections_q, projections_k)
        
        # 本地聚类损失
        losses['local_clustering'] = self.clustering_loss(local_assignments_q, local_assignments_k)
        
        # 全局聚类损失
        losses['global_clustering'] = self.clustering_loss(global_assignments_q, global_assignments_k)
        
        # 旋转预测损失（抗退化）
        if x3 is not None and rotation_labels is not None:
            rotation_features = self.backbone(x3)
            losses['rotation'] = self.rotation_loss(rotation_features, rotation_labels)
        
        # 在训练模式下更新内存和目标网络
        if self.training:
            # 更新内存
            self.update_memory(projections_k.detach())
            
            # 更新目标网络
            self.update_target_networks()
        
        return losses
    
    def get_representations(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取特征表示"""
        with torch.no_grad():
            features = self.backbone(x)
            projections = F.normalize(self.projector(features), dim=1)
        return features, projections
    
    def get_cluster_assignments(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取聚类分配"""
        with torch.no_grad():
            features = self.backbone(x)
            projections = F.normalize(self.projector(features), dim=1)
            local_assignments = self.local_clustering(projections)
            global_assignments = self.global_clustering(projections)
        return local_assignments, global_assignments


def create_orchestra_model(backbone_type: str = 'resnet18',
                          num_classes: int = 0,
                          projection_dim: int = 128,
                          num_local_clusters: int = 16,
                          num_global_clusters: int = 128,
                          memory_size: int = 128,
                          temperature: float = 0.1,
                          ema_decay: float = 0.996,
                          **kwargs) -> OrchestraModel:
    """创建Orchestra模型"""
    
    # 创建骨干网络
    if backbone_type.lower() == 'resnet18':
        backbone = ResNet18(num_classes=num_classes)
    elif backbone_type.lower() == 'resnet34':
        backbone = ResNet34(num_classes=num_classes)
    elif backbone_type.lower() == 'resnet50':
        backbone = ResNet50(num_classes=num_classes)
    else:
        raise ValueError(f"不支持的骨干网络类型: {backbone_type}")
    
    # 创建Orchestra模型
    model = OrchestraModel(
        backbone=backbone,
        projection_dim=projection_dim,
        num_local_clusters=num_local_clusters,
        num_global_clusters=num_global_clusters,
        memory_size=memory_size,
        temperature=temperature,
        ema_decay=ema_decay,
        **kwargs
    )
    
    return model


if __name__ == '__main__':
    # 测试模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建模型
    model = create_orchestra_model(
        backbone_type='resnet18',
        projection_dim=128,
        num_local_clusters=16,
        num_global_clusters=128
    ).to(device)
    
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
    # 测试前向传播
    batch_size = 4
    x1 = torch.randn(batch_size, 3, 32, 32).to(device)
    x2 = torch.randn(batch_size, 3, 32, 32).to(device)
    x3 = torch.randn(batch_size, 3, 32, 32).to(device)
    rotation_labels = torch.randint(0, 4, (batch_size,)).to(device)
    
    losses = model(x1, x2, x3, rotation_labels)
    
    print("损失:")
    for key, value in losses.items():
        print(f"  {key}: {value.item():.4f}")
    
    # 测试特征提取
    features, projections = model.get_representations(x1)
    print(f"特征形状: {features.shape}, 投影形状: {projections.shape}")
    
    # 测试聚类分配
    local_assignments, global_assignments = model.get_cluster_assignments(x1)
    print(f"本地聚类分配形状: {local_assignments.shape}, 全局聚类分配形状: {global_assignments.shape}")