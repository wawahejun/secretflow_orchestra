#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Orchestra在CIFAR-10数据集上的联邦学习实验复现
基于论文: Orchestra: Unsupervised Federated Learning via Globally Consistent Clustering
"""

import os
import sys
import logging
import argparse
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

# 添加SecretFlow路径
sys.path.append('/home/wawahejun/sf/secretflow')

import secretflow as sf
from secretflow.device import PYU
from secretflow.data.ndarray import FedNdarray
from secretflow_fl.ml.nn.applications.fl_orchestra_torch import create_orchestra_model
from secretflow_fl.utils.simulation.datasets_fl import load_cifar10_horiontal


class CIFAR10Dataset(Dataset):
    """CIFAR-10数据集包装器，支持数据增强"""
    
    def __init__(self, data, targets, transform=None, return_two_views=True, return_rotation=True):
        self.data = data
        self.targets = targets
        self.transform = transform
        self.return_two_views = return_two_views
        self.return_rotation = return_rotation
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img = self.data[idx]
        target = self.targets[idx]
        
        if self.transform:
            if self.return_two_views:
                # 返回两个增强视图用于对比学习
                img1 = self.transform(img)
                img2 = self.transform(img)
                
                if self.return_rotation:
                    # 添加旋转视图用于抗退化
                    rotation_angle = np.random.choice([0, 1, 2, 3])  # 0, 90, 180, 270度
                    img3 = torch.rot90(img1, rotation_angle, [1, 2])
                    return (img1, img2, img3), (target, target, rotation_angle)
                else:
                    return (img1, img2), (target, target)
            else:
                img = self.transform(img)
                return img, target
        else:
            return img, target


class ResNet18Backbone(nn.Module):
    """ResNet-18主干网络，用于CIFAR-10"""
    
    def __init__(self, num_classes=0):
        super(ResNet18Backbone, self).__init__()
        self.num_classes = num_classes
        
        # 第一层卷积
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        # ResNet层
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        
        # 全局平均池化
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 分类头（如果需要）
        if num_classes > 0:
            self.fc = nn.Linear(512, num_classes)
        
        self.output_dim = 512
    
    def _make_layer(self, in_planes, planes, num_blocks, stride):
        layers = []
        layers.append(BasicBlock(in_planes, planes, stride))
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(planes, planes, 1))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        
        if self.num_classes > 0:
            x = self.fc(x)
        
        return x


class BasicBlock(nn.Module):
    """ResNet基本块"""
    
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


def get_cifar10_transforms():
    """获取CIFAR-10数据变换"""
    
    # 训练时的数据增强
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
    ])
    
    # 测试时的变换
    test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
    ])
    
    return train_transform, test_transform


def load_cifar10_data(data_dir: str, num_clients: int = 4, alpha: float = 0.1):
    """加载并分割CIFAR-10数据集"""
    
    # 加载原始数据
    train_dataset = datasets.CIFAR10(data_dir, train=True, download=True)
    test_dataset = datasets.CIFAR10(data_dir, train=False, download=True)
    
    x_train, y_train = train_dataset.data, np.array(train_dataset.targets)
    x_test, y_test = test_dataset.data, np.array(test_dataset.targets)
    
    # 数据预处理：转换为float32并归一化到[0,1]
    x_train = x_train.astype(np.float32) / 255.0
    x_test = x_test.astype(np.float32) / 255.0
    
    # 转换维度：(N, H, W, C) -> (N, C, H, W)
    x_train = np.transpose(x_train, (0, 3, 1, 2))
    x_test = np.transpose(x_test, (0, 3, 1, 2))
    
    # 使用Dirichlet分布进行非IID数据分割
    def dirichlet_split(y, num_clients, alpha):
        num_classes = len(np.unique(y))
        label_distribution = np.random.dirichlet([alpha] * num_clients, num_classes)
        
        class_idcs = [np.argwhere(y == i).flatten() for i in range(num_classes)]
        client_idcs = [[] for _ in range(num_clients)]
        
        for c, fracs in zip(class_idcs, label_distribution):
            for i, idcs in enumerate(np.split(c, (np.cumsum(fracs)[:-1] * len(c)).astype(int))):
                client_idcs[i] += [idcs]
        
        client_idcs = [np.concatenate(idcs) for idcs in client_idcs]
        return client_idcs
    
    # 分割训练数据
    train_client_idcs = dirichlet_split(y_train, num_clients, alpha)
    
    # 分割测试数据（均匀分割）
    test_size_per_client = len(x_test) // num_clients
    test_client_idcs = []
    for i in range(num_clients):
        start_idx = i * test_size_per_client
        end_idx = (i + 1) * test_size_per_client if i < num_clients - 1 else len(x_test)
        test_client_idcs.append(np.arange(start_idx, end_idx))
    
    # 构建客户端数据
    client_data = {}
    for i in range(num_clients):
        train_idcs = train_client_idcs[i]
        test_idcs = test_client_idcs[i]
        
        client_data[f'client_{i}'] = {
            'x_train': x_train[train_idcs],
            'y_train': y_train[train_idcs],
            'x_test': x_test[test_idcs],
            'y_test': y_test[test_idcs]
        }
        
        print(f"客户端 {i}: 训练样本 {len(train_idcs)}, 测试样本 {len(test_idcs)}")
        print(f"  类别分布: {np.bincount(y_train[train_idcs])}")
    
    return client_data


def create_model_fn():
    """创建模型构建函数"""
    def model_fn():
        return ResNet18Backbone(num_classes=0)  # 无监督学习，不需要分类头
    return model_fn


def linear_evaluation(model, test_loader, device):
    """线性评估协议"""
    
    # 提取特征
    features = []
    labels = []
    
    model.eval()
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            if isinstance(data, (list, tuple)):
                data = data[0]  # 取第一个视图
            data = data.to(device)
            
            # 获取特征表示
            feat = model(data)
            features.append(feat.cpu().numpy())
            labels.append(target.numpy())
    
    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)
    
    # 训练线性分类器
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    
    classifier = LogisticRegression(max_iter=1000, random_state=42)
    classifier.fit(features, labels)
    
    # 预测
    predictions = classifier.predict(features)
    accuracy = accuracy_score(labels, predictions)
    
    return accuracy, features, labels


def clustering_evaluation(features, labels, num_clusters=10):
    """聚类评估"""
    
    from sklearn.cluster import KMeans
    
    # K-means聚类
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(features)
    
    # 计算聚类指标
    ari = adjusted_rand_score(labels, cluster_labels)
    nmi = normalized_mutual_info_score(labels, cluster_labels)
    
    return ari, nmi, cluster_labels


def run_orchestra_experiment(args):
    """运行Orchestra实验"""
    
    # 设置日志
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # 初始化SecretFlow
    sf.init(['alice', 'bob', 'charlie', 'david'], address='local')
    
    # 创建PYU设备
    devices = [PYU(f'client_{i}') for i in range(args.num_clients)]
    
    logger.info(f"创建了 {len(devices)} 个PYU设备")
    
    try:
        # 加载数据
        logger.info("加载CIFAR-10数据集...")
        client_data = load_cifar10_data(
            data_dir=args.data_dir,
            num_clients=args.num_clients,
            alpha=args.alpha
        )
        
        # 创建数据变换
        train_transform, test_transform = get_cifar10_transforms()
        
        # 准备联邦数据
        fed_data = {}
        for i, device in enumerate(devices):
            client_key = f'client_{i}'
            
            # 创建训练数据集
            train_dataset = CIFAR10Dataset(
                client_data[client_key]['x_train'],
                client_data[client_key]['y_train'],
                transform=train_transform,
                return_two_views=True,
                return_rotation=True
            )
            
            # 创建测试数据集
            test_dataset = CIFAR10Dataset(
                client_data[client_key]['x_test'],
                client_data[client_key]['y_test'],
                transform=test_transform,
                return_two_views=False,
                return_rotation=False
            )
            
            fed_data[device] = {
                'train': train_dataset,
                'test': test_dataset
            }
        
        # 创建Orchestra模型
        logger.info("创建Orchestra联邦学习模型...")
        model_fn = create_model_fn()
        
        orchestra_model = create_orchestra_model(
            device_list=devices,
            model_fn=model_fn,
            optim="adam",
            learning_rate=args.learning_rate,
            temperature=args.temperature,
            cluster_weight=args.cluster_weight,
            contrastive_weight=args.contrastive_weight,
            deg_weight=args.deg_weight,
            num_local_clusters=args.num_local_clusters,
            num_global_clusters=args.num_global_clusters,
            memory_size=args.memory_size,
            ema_decay=args.ema_decay
        )
        
        # 准备训练数据
        train_data = {}
        for device in devices:
            train_data[device] = fed_data[device]['train']
        
        # 转换为FedNdarray格式
        # 这里需要根据SecretFlow的具体API进行调整
        
        # 开始训练
        logger.info(f"开始Orchestra联邦学习训练，轮数: {args.num_rounds}")
        
        history = orchestra_model.fit(
            x=train_data,  # 需要转换为FedNdarray格式
            batch_size=args.batch_size,
            epochs=args.num_rounds,
            verbose=1,
            aggregate_freq=1
        )
        
        # 评估模型
        logger.info("开始模型评估...")
        
        # 线性评估
        total_accuracy = 0
        total_ari = 0
        total_nmi = 0
        
        for i, device in enumerate(devices):
            test_dataset = fed_data[device]['test']
            test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
            
            # 获取设备上的模型
            device_model = orchestra_model.device_y[device]
            
            # 线性评估
            accuracy, features, labels = linear_evaluation(
                device_model, test_loader, torch.device('cpu')
            )
            
            # 聚类评估
            ari, nmi, cluster_labels = clustering_evaluation(
                features, labels, num_clusters=10
            )
            
            logger.info(f"客户端 {i} - 线性评估准确率: {accuracy:.4f}, ARI: {ari:.4f}, NMI: {nmi:.4f}")
            
            total_accuracy += accuracy
            total_ari += ari
            total_nmi += nmi
        
        # 平均结果
        avg_accuracy = total_accuracy / len(devices)
        avg_ari = total_ari / len(devices)
        avg_nmi = total_nmi / len(devices)
        
        logger.info(f"平均结果 - 线性评估准确率: {avg_accuracy:.4f}, ARI: {avg_ari:.4f}, NMI: {avg_nmi:.4f}")
        
        # 保存结果
        results = {
            'accuracy': avg_accuracy,
            'ari': avg_ari,
            'nmi': avg_nmi,
            'history': history
        }
        
        # 保存模型
        if args.save_model:
            model_path = os.path.join(args.output_dir, 'orchestra_model.pth')
            orchestra_model.save_model(model_path)
            logger.info(f"模型已保存到: {model_path}")
        
        return results
        
    finally:
        # 关闭SecretFlow
        sf.shutdown()


def main():
    parser = argparse.ArgumentParser(description='Orchestra CIFAR-10 联邦学习实验')
    
    # 数据参数
    parser.add_argument('--data_dir', type=str, default='./data', help='数据目录')
    parser.add_argument('--num_clients', type=int, default=4, help='客户端数量')
    parser.add_argument('--alpha', type=float, default=0.1, help='Dirichlet分布参数')
    
    # 训练参数
    parser.add_argument('--num_rounds', type=int, default=100, help='联邦学习轮数')
    parser.add_argument('--batch_size', type=int, default=16, help='批次大小')
    parser.add_argument('--learning_rate', type=float, default=0.003, help='学习率')
    
    # Orchestra参数
    parser.add_argument('--temperature', type=float, default=0.1, help='温度参数')
    parser.add_argument('--cluster_weight', type=float, default=1.0, help='聚类损失权重')
    parser.add_argument('--contrastive_weight', type=float, default=1.0, help='对比损失权重')
    parser.add_argument('--deg_weight', type=float, default=1.0, help='抗退化损失权重')
    parser.add_argument('--num_local_clusters', type=int, default=16, help='本地聚类数量')
    parser.add_argument('--num_global_clusters', type=int, default=128, help='全局聚类数量')
    parser.add_argument('--memory_size', type=int, default=128, help='投影内存大小')
    parser.add_argument('--ema_decay', type=float, default=0.996, help='EMA衰减率')
    
    # 其他参数
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--output_dir', type=str, default='./output', help='输出目录')
    parser.add_argument('--save_model', action='store_true', help='是否保存模型')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 运行实验
    results = run_orchestra_experiment(args)
    
    print("\n=== 实验结果 ===")
    print(f"线性评估准确率: {results['accuracy']:.4f}")
    print(f"调整兰德指数 (ARI): {results['ari']:.4f}")
    print(f"标准化互信息 (NMI): {results['nmi']:.4f}")


if __name__ == '__main__':
    main()