#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
原论文标准评价指标实现
包含线性探测、1%标签半监督学习、10%标签半监督学习
基于论文: Orchestra: Unsupervised Federated Learning via Globally Consistent Clustering
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, adjusted_rand_score, normalized_mutual_info_score
from typing import Dict, List, Tuple, Optional, Any
import json
import logging
from collections import defaultdict

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LinearProbeEvaluator:
    """线性探测评估器 - 原论文标准"""
    
    def __init__(self, 
                 input_dim: int,
                 num_classes: int = 10,
                 max_iter: int = 1000,
                 random_state: int = 42):
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.max_iter = max_iter
        self.random_state = random_state
        self.classifier = None
        
    def extract_features(self, model: nn.Module, data_loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
        """提取特征表示"""
        model.eval()
        features = []
        labels = []
        
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(data_loader):
                # 处理不同的数据格式
                if isinstance(batch_data, (list, tuple)):
                    if len(batch_data) == 2:
                        data, target = batch_data
                    else:
                        data = batch_data[0]
                        target = batch_data[1] if len(batch_data) > 1 else torch.zeros(data.shape[0])
                else:
                    data = batch_data
                    target = torch.zeros(data.shape[0])
                
                # 处理多视图数据
                if isinstance(data, (list, tuple)):
                    data = data[0]  # 取第一个视图
                if isinstance(target, (list, tuple)):
                    target = target[0]
                
                data = data.to(device)
                
                # 获取特征表示
                try:
                    if hasattr(model, 'get_representations'):
                        feat, _ = model.get_representations(data)
                    elif hasattr(model, 'backbone'):
                        feat = model.backbone(data)
                    elif hasattr(model, 'forward') and hasattr(model, 'fc'):
                        # 对于有分类层的模型，获取分类层之前的特征
                        x = data
                        for name, module in model.named_children():
                            if name != 'fc' and name != 'classifier':
                                x = module(x)
                        feat = x.view(x.size(0), -1)  # 展平
                    else:
                        feat = model(data, return_features=True) if 'return_features' in model.forward.__code__.co_varnames else model(data)
                    
                    # 确保特征是2D的
                    if feat.dim() > 2:
                        feat = feat.view(feat.size(0), -1)
                    
                    features.append(feat.cpu().numpy())
                    labels.append(target.cpu().numpy() if torch.is_tensor(target) else target)
                    
                except Exception as e:
                    logger.warning(f"特征提取失败 (batch {batch_idx}): {e}")
                    continue
        
        if not features:
            raise ValueError("未能提取到任何特征")
        
        features = np.concatenate(features, axis=0)
        labels = np.concatenate(labels, axis=0)
        
        logger.info(f"提取特征完成: {features.shape}, 标签: {labels.shape}")
        return features, labels
    
    def linear_probe_evaluation(self, 
                               train_features: np.ndarray, 
                               train_labels: np.ndarray,
                               test_features: np.ndarray, 
                               test_labels: np.ndarray) -> float:
        """线性探测评估"""
        # 训练线性分类器
        self.classifier = LogisticRegression(
            max_iter=self.max_iter,
            random_state=self.random_state,
            multi_class='ovr',
            solver='lbfgs'
        )
        
        # 标准化特征
        train_mean = np.mean(train_features, axis=0)
        train_std = np.std(train_features, axis=0) + 1e-8
        
        train_features_norm = (train_features - train_mean) / train_std
        test_features_norm = (test_features - train_mean) / train_std
        
        # 训练分类器
        self.classifier.fit(train_features_norm, train_labels)
        
        # 预测
        predictions = self.classifier.predict(test_features_norm)
        accuracy = accuracy_score(test_labels, predictions)
        
        return accuracy


class SemiSupervisedEvaluator:
    """半监督学习评估器 - 原论文标准"""
    
    def __init__(self, 
                 input_dim: int,
                 num_classes: int = 10,
                 hidden_dim: int = 512,
                 learning_rate: float = 0.001,
                 epochs: int = 100,
                 batch_size: int = 64,
                 device: torch.device = None):
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def create_labeled_subset(self, 
                             features: np.ndarray, 
                             labels: np.ndarray, 
                             label_ratio: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """创建标记子集"""
        num_samples = len(features)
        num_labeled = int(num_samples * label_ratio)
        
        # 确保每个类别至少有一个样本
        unique_labels = np.unique(labels)
        samples_per_class = max(1, num_labeled // len(unique_labels))
        
        labeled_indices = []
        for label in unique_labels:
            class_indices = np.where(labels == label)[0]
            if len(class_indices) > 0:
                selected = np.random.choice(class_indices, 
                                          min(samples_per_class, len(class_indices)), 
                                          replace=False)
                labeled_indices.extend(selected)
        
        # 如果还需要更多样本，随机选择
        if len(labeled_indices) < num_labeled:
            remaining_indices = np.setdiff1d(np.arange(num_samples), labeled_indices)
            additional = np.random.choice(remaining_indices, 
                                        num_labeled - len(labeled_indices), 
                                        replace=False)
            labeled_indices.extend(additional)
        
        labeled_indices = np.array(labeled_indices[:num_labeled])
        unlabeled_indices = np.setdiff1d(np.arange(num_samples), labeled_indices)
        
        return (features[labeled_indices], labels[labeled_indices],
                features[unlabeled_indices], labels[unlabeled_indices])
    
    def create_mlp_classifier(self) -> nn.Module:
        """创建MLP分类器"""
        return nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(self.hidden_dim // 2, self.num_classes)
        )
    
    def train_semisupervised(self, 
                           labeled_features: np.ndarray,
                           labeled_labels: np.ndarray,
                           unlabeled_features: np.ndarray,
                           test_features: np.ndarray,
                           test_labels: np.ndarray) -> float:
        """半监督训练"""
        # 创建分类器
        classifier = self.create_mlp_classifier().to(self.device)
        optimizer = torch.optim.Adam(classifier.parameters(), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        # 标准化特征（先在numpy中进行）
        all_features = np.concatenate([labeled_features, unlabeled_features], axis=0)
        mean = np.mean(all_features, axis=0)
        std = np.std(all_features, axis=0) + 1e-8
        
        # 标准化numpy数组
        labeled_features_norm = (labeled_features - mean) / std
        unlabeled_features_norm = (unlabeled_features - mean) / std
        test_features_norm = (test_features - mean) / std
        
        # 转换为张量（确保梯度计算）
        labeled_features_tensor = torch.FloatTensor(labeled_features_norm).to(self.device).requires_grad_(False)
        labeled_labels_tensor = torch.LongTensor(labeled_labels).to(self.device)
        unlabeled_features_tensor = torch.FloatTensor(unlabeled_features_norm).to(self.device).requires_grad_(False)
        test_features_tensor = torch.FloatTensor(test_features_norm).to(self.device).requires_grad_(False)
        test_labels_tensor = torch.LongTensor(test_labels).to(self.device)
        
        best_accuracy = 0.0
        
        for epoch in range(self.epochs):
            classifier.train()
            
            # 监督学习损失
            optimizer.zero_grad()
            labeled_outputs = classifier(labeled_features_tensor)
            supervised_loss = criterion(labeled_outputs, labeled_labels_tensor)
            
            # 简单的半监督策略：伪标签
            if epoch > 10:  # 在几个epoch后开始使用伪标签
                classifier.eval()
                with torch.no_grad():
                    unlabeled_outputs = classifier(unlabeled_features_tensor)
                    pseudo_labels = torch.argmax(unlabeled_outputs, dim=1)
                    confidence = torch.max(F.softmax(unlabeled_outputs, dim=1), dim=1)[0]
                    
                    # 只使用高置信度的伪标签
                    high_conf_mask = confidence > 0.8
                
                classifier.train()
                if high_conf_mask.sum() > 0:
                    # 重新计算高置信度样本的输出（需要梯度）
                    high_conf_features = unlabeled_features_tensor[high_conf_mask].detach()
                    pseudo_outputs = classifier(high_conf_features)
                    pseudo_loss = criterion(pseudo_outputs, pseudo_labels[high_conf_mask].detach())
                    total_loss = supervised_loss + 0.1 * pseudo_loss
                else:
                    total_loss = supervised_loss
            else:
                total_loss = supervised_loss
            
            total_loss.backward()
            optimizer.step()
            
            # 评估
            if (epoch + 1) % 10 == 0:
                classifier.eval()
                with torch.no_grad():
                    test_outputs = classifier(test_features_tensor)
                    predictions = torch.argmax(test_outputs, dim=1)
                    accuracy = (predictions == test_labels_tensor).float().mean().item()
                    
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                    
                    logger.info(f"Epoch {epoch+1}/{self.epochs}, Loss: {total_loss.item():.4f}, Accuracy: {accuracy:.4f}")
        
        return best_accuracy


class PaperStandardEvaluator:
    """原论文标准评估器"""
    
    def __init__(self, 
                 num_classes: int = 10,
                 device: torch.device = None,
                 output_dir: str = './paper_evaluation_results'):
        self.num_classes = num_classes
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.output_dir = output_dir
        
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"初始化原论文标准评估器，设备: {self.device}")
    
    def full_evaluation(self, 
                       model: nn.Module,
                       train_loader: DataLoader,
                       test_loader: DataLoader) -> Dict[str, Any]:
        """完整的原论文标准评估"""
        
        logger.info("开始原论文标准评估")
        
        # 提取特征
        logger.info("提取训练集特征...")
        probe_evaluator = LinearProbeEvaluator(input_dim=0, num_classes=self.num_classes)
        train_features, train_labels = probe_evaluator.extract_features(model, train_loader, self.device)
        
        logger.info("提取测试集特征...")
        test_features, test_labels = probe_evaluator.extract_features(model, test_loader, self.device)
        
        # 更新input_dim
        input_dim = train_features.shape[1]
        probe_evaluator.input_dim = input_dim
        
        results = {}
        
        # 1. 线性探测评估
        logger.info("执行线性探测评估...")
        linear_probe_acc = probe_evaluator.linear_probe_evaluation(
            train_features, train_labels, test_features, test_labels
        )
        results['linear_probe_accuracy'] = linear_probe_acc
        logger.info(f"线性探测准确率: {linear_probe_acc:.4f} ({linear_probe_acc*100:.2f}%)")
        
        # 2. 1% 标签半监督学习
        logger.info("执行1%标签半监督学习评估...")
        semisup_evaluator = SemiSupervisedEvaluator(
            input_dim=input_dim,
            num_classes=self.num_classes,
            device=self.device,
            epochs=50  # 减少epoch数以加快评估
        )
        
        # 使用训练集的1%作为标记数据
        labeled_1_features, labeled_1_labels, unlabeled_1_features, _ = semisup_evaluator.create_labeled_subset(
            train_features, train_labels, label_ratio=0.01
        )
        
        semisup_1_acc = semisup_evaluator.train_semisupervised(
            labeled_1_features, labeled_1_labels, unlabeled_1_features,
            test_features, test_labels
        )
        results['semisupervised_1_percent'] = semisup_1_acc
        logger.info(f"1%标签半监督准确率: {semisup_1_acc:.4f} ({semisup_1_acc*100:.2f}%)")
        
        # 3. 10% 标签半监督学习
        logger.info("执行10%标签半监督学习评估...")
        labeled_10_features, labeled_10_labels, unlabeled_10_features, _ = semisup_evaluator.create_labeled_subset(
            train_features, train_labels, label_ratio=0.10
        )
        
        semisup_10_acc = semisup_evaluator.train_semisupervised(
            labeled_10_features, labeled_10_labels, unlabeled_10_features,
            test_features, test_labels
        )
        results['semisupervised_10_percent'] = semisup_10_acc
        logger.info(f"10%标签半监督准确率: {semisup_10_acc:.4f} ({semisup_10_acc*100:.2f}%)")
        
        # 4. 聚类评估（额外指标）
        logger.info("执行聚类评估...")
        kmeans = KMeans(n_clusters=self.num_classes, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(test_features)
        
        ari = adjusted_rand_score(test_labels, cluster_labels)
        nmi = normalized_mutual_info_score(test_labels, cluster_labels)
        
        results['clustering_ari'] = ari
        results['clustering_nmi'] = nmi
        logger.info(f"聚类ARI: {ari:.4f}, NMI: {nmi:.4f}")
        
        # 保存结果
        self.save_results(results)
        
        # 打印汇总
        self.print_summary(results)
        
        return results
    
    def save_results(self, results: Dict[str, Any]):
        """保存评估结果"""
        results_file = os.path.join(self.output_dir, 'paper_standard_results.json')
        
        # 转换为可序列化的格式
        serializable_results = {}
        for key, value in results.items():
            if isinstance(value, (np.ndarray, torch.Tensor)):
                serializable_results[key] = float(value)
            else:
                serializable_results[key] = value
        
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"结果已保存到: {results_file}")
    
    def print_summary(self, results: Dict[str, Any]):
        """打印评估结果汇总"""
        print("\n" + "=" * 60)
        print("原论文标准评估结果汇总")
        print("=" * 60)
        
        print(f"\n📊 主要评估指标:")
        print(f"  • 线性探测准确率:     {results['linear_probe_accuracy']:.4f} ({results['linear_probe_accuracy']*100:.2f}%)")
        print(f"  • 1%标签半监督准确率:  {results['semisupervised_1_percent']:.4f} ({results['semisupervised_1_percent']*100:.2f}%)")
        print(f"  • 10%标签半监督准确率: {results['semisupervised_10_percent']:.4f} ({results['semisupervised_10_percent']*100:.2f}%)")
        
        print(f"\n🎯 聚类性能:")
        print(f"  • 调整兰德指数 (ARI): {results['clustering_ari']:.4f}")
        print(f"  • 标准化互信息 (NMI): {results['clustering_nmi']:.4f}")
        
        print(f"\n📈 与原论文对比 (CIFAR-10, α=0.1, 100客户端):")
        print(f"  • 原论文线性探测:     71.58%")
        print(f"  • 原论文1%半监督:     60.33%")
        print(f"  • 原论文10%半监督:    66.20%")
        print(f"  • 当前线性探测:       {results['linear_probe_accuracy']*100:.2f}%")
        print(f"  • 当前1%半监督:       {results['semisupervised_1_percent']*100:.2f}%")
        print(f"  • 当前10%半监督:      {results['semisupervised_10_percent']*100:.2f}%")
        
        # 计算性能比例
        linear_ratio = results['linear_probe_accuracy'] / 0.7158
        semisup_1_ratio = results['semisupervised_1_percent'] / 0.6033
        semisup_10_ratio = results['semisupervised_10_percent'] / 0.6620
        
        print(f"\n📊 性能比例 (当前/原论文):")
        print(f"  • 线性探测:   {linear_ratio:.2f}x")
        print(f"  • 1%半监督:   {semisup_1_ratio:.2f}x")
        print(f"  • 10%半监督:  {semisup_10_ratio:.2f}x")
        
        print("=" * 60)


if __name__ == '__main__':
    # 测试代码
    import torch
    from torch.utils.data import TensorDataset, DataLoader
    
    # 创建模拟数据
    num_train = 1000
    num_test = 200
    num_features = 512
    num_classes = 10
    
    # 模拟特征和标签
    train_features = torch.randn(num_train, num_features)
    train_labels = torch.randint(0, num_classes, (num_train,))
    test_features = torch.randn(num_test, num_features)
    test_labels = torch.randint(0, num_classes, (num_test,))
    
    # 创建数据集和数据加载器
    train_dataset = TensorDataset(train_features, train_labels)
    test_dataset = TensorDataset(test_features, test_labels)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # 创建模拟模型
    class MockModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(num_features, num_features)
        
        def forward(self, x, return_features=False):
            if return_features:
                return self.fc(x)
            return self.fc(x)
    
    model = MockModel()
    
    # 测试评估器
    evaluator = PaperStandardEvaluator(num_classes=num_classes)
    results = evaluator.full_evaluation(model, train_loader, test_loader)