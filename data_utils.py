#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CIFAR-10数据处理工具
包含数据加载、预处理、联邦分割等功能
"""

import os
import pickle
import numpy as np
from typing import Dict, List, Tuple, Optional

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from PIL import Image


class CIFAR10FederatedDataset(Dataset):
    """CIFAR-10联邦学习数据集"""
    
    def __init__(self, 
                 data: np.ndarray, 
                 targets: np.ndarray, 
                 transform=None, 
                 return_two_views: bool = True,
                 return_rotation: bool = True):
        """
        Args:
            data: 图像数据，形状为 (N, C, H, W) 或 (N, H, W, C)
            targets: 标签数据
            transform: 数据变换
            return_two_views: 是否返回两个增强视图（用于对比学习）
            return_rotation: 是否返回旋转视图（用于抗退化）
        """
        self.data = data
        self.targets = targets
        self.transform = transform
        self.return_two_views = return_two_views
        self.return_rotation = return_rotation
        
        # 确保数据格式正确
        if len(self.data.shape) == 4 and self.data.shape[1] == 3:
            # 数据已经是 (N, C, H, W) 格式
            pass
        elif len(self.data.shape) == 4 and self.data.shape[3] == 3:
            # 数据是 (N, H, W, C) 格式，需要转换
            self.data = np.transpose(self.data, (0, 3, 1, 2))
        else:
            raise ValueError(f"不支持的数据形状: {self.data.shape}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img = self.data[idx]
        target = self.targets[idx]
        
        # 转换为PIL图像
        if isinstance(img, np.ndarray):
            # 如果是 (C, H, W) 格式，转换为 (H, W, C)
            if img.shape[0] == 3:
                img = np.transpose(img, (1, 2, 0))
            
            # 确保数据范围在 [0, 255]
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            else:
                img = img.astype(np.uint8)
            
            img = Image.fromarray(img)
        
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
            # 如果没有变换，直接返回tensor
            if isinstance(img, Image.Image):
                img = transforms.ToTensor()(img)
            return img, target


def load_cifar10_raw(data_dir: str, download: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """加载原始CIFAR-10数据"""
    
    # 使用torchvision加载数据
    train_dataset = datasets.CIFAR10(data_dir, train=True, download=download)
    test_dataset = datasets.CIFAR10(data_dir, train=False, download=download)
    
    # 提取数据和标签
    x_train = train_dataset.data  # (50000, 32, 32, 3)
    y_train = np.array(train_dataset.targets)
    x_test = test_dataset.data    # (10000, 32, 32, 3)
    y_test = np.array(test_dataset.targets)
    
    return x_train, y_train, x_test, y_test


def load_cifar10_from_pickle(data_dir: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """从pickle文件加载CIFAR-10数据（原始格式）"""
    
    def unpickle(file):
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict
    
    # 加载训练数据
    x_train = []
    y_train = []
    
    for i in range(1, 6):
        batch_file = os.path.join(data_dir, 'cifar-10-batches-py', f'data_batch_{i}')
        batch_dict = unpickle(batch_file)
        x_train.append(batch_dict[b'data'])
        y_train.extend(batch_dict[b'labels'])
    
    x_train = np.concatenate(x_train, axis=0)
    y_train = np.array(y_train)
    
    # 加载测试数据
    test_file = os.path.join(data_dir, 'cifar-10-batches-py', 'test_batch')
    test_dict = unpickle(test_file)
    x_test = test_dict[b'data']
    y_test = np.array(test_dict[b'labels'])
    
    # 重塑数据：(N, 3072) -> (N, 3, 32, 32) -> (N, 32, 32, 3)
    x_train = x_train.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    x_test = x_test.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    
    return x_train, y_train, x_test, y_test


def create_dirichlet_split(labels: np.ndarray, 
                          num_clients: int, 
                          alpha: float, 
                          min_samples_per_client: int = 10) -> List[np.ndarray]:
    """使用Dirichlet分布创建非IID数据分割"""
    
    num_classes = len(np.unique(labels))
    
    # 为每个类别生成Dirichlet分布
    label_distribution = np.random.dirichlet([alpha] * num_clients, num_classes)
    
    # 获取每个类别的样本索引
    class_idcs = [np.argwhere(labels == i).flatten() for i in range(num_classes)]
    
    # 初始化客户端索引列表
    client_idcs = [[] for _ in range(num_clients)]
    
    # 为每个类别分配样本到客户端
    for c, fracs in zip(class_idcs, label_distribution):
        # 计算每个客户端应该获得的样本数量
        client_samples = (fracs * len(c)).astype(int)
        
        # 确保所有样本都被分配
        client_samples[-1] = len(c) - client_samples[:-1].sum()
        
        # 随机打乱类别内的样本
        np.random.shuffle(c)
        
        # 分配样本到客户端
        start_idx = 0
        for i, num_samples in enumerate(client_samples):
            if num_samples > 0:
                client_idcs[i].extend(c[start_idx:start_idx + num_samples])
                start_idx += num_samples
    
    # 转换为numpy数组并确保每个客户端至少有最小样本数
    client_idcs = [np.array(idcs) for idcs in client_idcs]
    
    # 检查并调整样本数量
    for i, idcs in enumerate(client_idcs):
        if len(idcs) < min_samples_per_client:
            print(f"警告: 客户端 {i} 只有 {len(idcs)} 个样本，少于最小要求 {min_samples_per_client}")
    
    return client_idcs


def create_uniform_split(num_samples: int, num_clients: int) -> List[np.ndarray]:
    """创建均匀数据分割"""
    
    samples_per_client = num_samples // num_clients
    client_idcs = []
    
    for i in range(num_clients):
        start_idx = i * samples_per_client
        if i == num_clients - 1:
            # 最后一个客户端获得剩余的所有样本
            end_idx = num_samples
        else:
            end_idx = (i + 1) * samples_per_client
        
        client_idcs.append(np.arange(start_idx, end_idx))
    
    return client_idcs


def create_federated_cifar10(data_dir: str,
                             num_clients: int = 4,
                             alpha: float = 0.1,
                             test_split_method: str = 'uniform',
                             min_samples_per_client: int = 10,
                             seed: int = 42) -> Dict[str, Dict[str, np.ndarray]]:
    """创建联邦CIFAR-10数据集"""
    
    # 设置随机种子
    np.random.seed(seed)
    
    # 加载原始数据
    try:
        x_train, y_train, x_test, y_test = load_cifar10_raw(data_dir, download=True)
    except Exception as e:
        print(f"使用torchvision加载失败，尝试从pickle文件加载: {e}")
        x_train, y_train, x_test, y_test = load_cifar10_from_pickle(data_dir)
    
    print(f"原始数据形状: 训练集 {x_train.shape}, 测试集 {x_test.shape}")
    print(f"标签范围: 训练集 {y_train.min()}-{y_train.max()}, 测试集 {y_test.min()}-{y_test.max()}")
    
    # 数据预处理：转换为float32并归一化到[0,1]
    x_train = x_train.astype(np.float32) / 255.0
    x_test = x_test.astype(np.float32) / 255.0
    
    # 分割训练数据（非IID）
    print(f"使用Dirichlet分布分割训练数据，alpha={alpha}")
    train_client_idcs = create_dirichlet_split(
        y_train, num_clients, alpha, min_samples_per_client
    )
    
    # 分割测试数据
    if test_split_method == 'uniform':
        print("均匀分割测试数据")
        test_client_idcs = create_uniform_split(len(x_test), num_clients)
    elif test_split_method == 'dirichlet':
        print(f"使用Dirichlet分布分割测试数据，alpha={alpha}")
        test_client_idcs = create_dirichlet_split(
            y_test, num_clients, alpha, min_samples_per_client
        )
    else:
        raise ValueError(f"不支持的测试数据分割方法: {test_split_method}")
    
    # 构建客户端数据字典
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
        
        # 打印客户端数据统计
        train_class_counts = np.bincount(y_train[train_idcs], minlength=10)
        test_class_counts = np.bincount(y_test[test_idcs], minlength=10)
        
        print(f"客户端 {i}:")
        print(f"  训练样本: {len(train_idcs)}, 测试样本: {len(test_idcs)}")
        print(f"  训练类别分布: {train_class_counts}")
        print(f"  测试类别分布: {test_class_counts}")
    
    return client_data


def get_cifar10_transforms(config: Dict) -> Tuple[transforms.Compose, transforms.Compose]:
    """获取CIFAR-10数据变换"""
    
    # 训练时的数据增强
    train_transforms = [
        transforms.RandomCrop(config.get('crop_size', 32), 
                            padding=config.get('crop_padding', 4)),
        transforms.RandomHorizontalFlip(p=config.get('horizontal_flip_prob', 0.5)),
    ]
    
    # 颜色抖动
    if config.get('use_data_augmentation', True):
        color_jitter_params = config.get('color_jitter', {
            'brightness': 0.4,
            'contrast': 0.4,
            'saturation': 0.4,
            'hue': 0.1
        })
        train_transforms.append(transforms.ColorJitter(**color_jitter_params))
    
    # 转换为tensor和归一化
    train_transforms.extend([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=config.get('normalize_mean', [0.4914, 0.4822, 0.4465]),
            std=config.get('normalize_std', [0.2023, 0.1994, 0.2010])
        )
    ])
    
    train_transform = transforms.Compose(train_transforms)
    
    # 测试时的变换（只有归一化）
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=config.get('normalize_mean', [0.4914, 0.4822, 0.4465]),
            std=config.get('normalize_std', [0.2023, 0.1994, 0.2010])
        )
    ])
    
    return train_transform, test_transform


def create_data_loaders(client_data: Dict[str, Dict[str, np.ndarray]],
                       config: Dict,
                       client_id: str,
                       return_two_views: bool = True,
                       return_rotation: bool = True) -> Tuple[DataLoader, DataLoader]:
    """为指定客户端创建数据加载器"""
    
    # 获取数据变换
    train_transform, test_transform = get_cifar10_transforms(config)
    
    # 创建数据集
    train_dataset = CIFAR10FederatedDataset(
        client_data[client_id]['x_train'],
        client_data[client_id]['y_train'],
        transform=train_transform,
        return_two_views=return_two_views,
        return_rotation=return_rotation
    )
    
    test_dataset = CIFAR10FederatedDataset(
        client_data[client_id]['x_test'],
        client_data[client_id]['y_test'],
        transform=test_transform,
        return_two_views=False,
        return_rotation=False
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.get('batch_size', 16),
        shuffle=True,
        num_workers=config.get('num_workers', 4),
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.get('batch_size', 16),
        shuffle=False,
        num_workers=config.get('num_workers', 4),
        pin_memory=True
    )
    
    return train_loader, test_loader


def analyze_data_distribution(client_data: Dict[str, Dict[str, np.ndarray]], 
                            num_classes: int = 10) -> Dict:
    """分析联邦数据分布"""
    
    analysis = {
        'num_clients': len(client_data),
        'total_train_samples': 0,
        'total_test_samples': 0,
        'client_stats': {},
        'class_distribution': np.zeros(num_classes),
        'heterogeneity_metrics': {}
    }
    
    client_class_distributions = []
    
    for client_id, data in client_data.items():
        train_samples = len(data['y_train'])
        test_samples = len(data['y_test'])
        
        train_class_dist = np.bincount(data['y_train'], minlength=num_classes)
        test_class_dist = np.bincount(data['y_test'], minlength=num_classes)
        
        analysis['total_train_samples'] += train_samples
        analysis['total_test_samples'] += test_samples
        analysis['class_distribution'] += train_class_dist
        
        analysis['client_stats'][client_id] = {
            'train_samples': train_samples,
            'test_samples': test_samples,
            'train_class_dist': train_class_dist,
            'test_class_dist': test_class_dist,
            'dominant_class': np.argmax(train_class_dist),
            'num_classes_present': np.sum(train_class_dist > 0)
        }
        
        # 归一化的类别分布（用于计算异构性）
        normalized_dist = train_class_dist / train_class_dist.sum() if train_class_dist.sum() > 0 else train_class_dist
        client_class_distributions.append(normalized_dist)
    
    # 计算异构性指标
    client_class_distributions = np.array(client_class_distributions)
    
    # 计算KL散度（相对于均匀分布）
    uniform_dist = np.ones(num_classes) / num_classes
    kl_divergences = []
    
    for dist in client_class_distributions:
        if np.sum(dist) > 0:
            # 避免log(0)
            dist_smooth = dist + 1e-10
            kl_div = np.sum(dist_smooth * np.log(dist_smooth / uniform_dist))
            kl_divergences.append(kl_div)
    
    analysis['heterogeneity_metrics'] = {
        'mean_kl_divergence': np.mean(kl_divergences),
        'std_kl_divergence': np.std(kl_divergences),
        'max_kl_divergence': np.max(kl_divergences),
        'min_kl_divergence': np.min(kl_divergences)
    }
    
    return analysis


if __name__ == '__main__':
    # 测试数据加载和分割
    from config import get_config
    
    config = get_config('small')
    
    print("创建联邦CIFAR-10数据集...")
    client_data = create_federated_cifar10(
        data_dir='./data',
        num_clients=config['num_clients'],
        alpha=config['alpha'],
        seed=config['seed']
    )
    
    print("\n分析数据分布...")
    analysis = analyze_data_distribution(client_data)
    
    print(f"\n总训练样本: {analysis['total_train_samples']}")
    print(f"总测试样本: {analysis['total_test_samples']}")
    print(f"平均KL散度: {analysis['heterogeneity_metrics']['mean_kl_divergence']:.4f}")
    
    print("\n测试数据加载器...")
    train_loader, test_loader = create_data_loaders(
        client_data, config, 'client_0'
    )
    
    # 测试一个批次
    for batch_idx, (data, target) in enumerate(train_loader):
        print(f"批次 {batch_idx}: 数据形状 {[d.shape for d in data]}, 标签形状 {[t.shape for t in target]}")
        break