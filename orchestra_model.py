#!/usr/bin/env python3
"""
Orchestra: Unsupervised Federated Learning via Globally Consistent Clustering
基于SecretFlow框架的Orchestra算法实现

论文参考: Orchestra: Unsupervised Federated Learning via Globally Consistent Clustering
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import logging
from abc import ABC, abstractmethod

class ContrastiveEncoder(nn.Module):
    """对比学习编码器网络"""
    
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int, dropout_rate: float = 0.2):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        layers = []
        prev_dim = input_dim
        
        # 构建编码器层
        for i, hidden_dim in enumerate(hidden_dims):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        # 输出层
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.encoder = nn.Sequential(*layers)
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        # 编码
        encoded = self.encoder(x)
        # L2归一化
        return F.normalize(encoded, dim=1, p=2)

class ClusteringHead(nn.Module):
    """聚类头网络"""
    
    def __init__(self, input_dim: int, num_clusters: int, temperature: float = 1.0):
        super().__init__()
        self.num_clusters = num_clusters
        self.temperature = temperature
        
        # 聚类中心参数
        self.cluster_centers = nn.Parameter(torch.randn(num_clusters, input_dim))
        
        # 初始化聚类中心
        nn.init.xavier_uniform_(self.cluster_centers)
    
    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """计算聚类分配概率"""
        # 归一化聚类中心
        normalized_centers = F.normalize(self.cluster_centers, dim=1, p=2)
        
        # 计算相似度（余弦相似度）
        similarities = torch.mm(embeddings, normalized_centers.t())
        
        # 应用温度参数并计算softmax
        cluster_probs = F.softmax(similarities / self.temperature, dim=1)
        
        return cluster_probs
    
    def get_cluster_centers(self) -> torch.Tensor:
        """获取归一化的聚类中心"""
        return F.normalize(self.cluster_centers, dim=1, p=2)

class OrchestraModel(nn.Module):
    """Orchestra完整模型"""
    
    def __init__(self, 
                 input_dim: int,
                 hidden_dims: List[int] = [512, 256],
                 embedding_dim: int = 128,
                 num_clusters: int = 10,
                 dropout_rate: float = 0.2,
                 temperature: float = 0.5):
        super().__init__()
        
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.num_clusters = num_clusters
        self.temperature = temperature
        
        # 对比学习编码器
        self.encoder = ContrastiveEncoder(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=embedding_dim,
            dropout_rate=dropout_rate
        )
        
        # 聚类头
        self.cluster_head = ClusteringHead(
            input_dim=embedding_dim,
            num_clusters=num_clusters,
            temperature=temperature
        )
        
        # 投影头（用于对比学习）
        self.projection_head = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embedding_dim, embedding_dim)
        )
    
    def forward(self, x: torch.Tensor, return_projections: bool = False) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """前向传播"""
        # 编码
        embeddings = self.encoder(x)
        
        # 聚类
        cluster_probs = self.cluster_head(embeddings)
        
        # 投影（用于对比学习）
        projections = None
        if return_projections:
            projections = F.normalize(self.projection_head(embeddings), dim=1, p=2)
        
        return embeddings, cluster_probs, projections
    
    def get_cluster_assignments(self, x: torch.Tensor) -> torch.Tensor:
        """获取聚类分配"""
        with torch.no_grad():
            _, cluster_probs, _ = self.forward(x)
            return torch.argmax(cluster_probs, dim=1)

class OrchestraLoss(nn.Module):
    """Orchestra损失函数"""
    
    def __init__(self, 
                 contrastive_weight: float = 1.0,
                 clustering_weight: float = 1.0,
                 consistency_weight: float = 1.0,
                 temperature: float = 0.5):
        super().__init__()
        
        self.contrastive_weight = contrastive_weight
        self.clustering_weight = clustering_weight
        self.consistency_weight = consistency_weight
        self.temperature = temperature
    
    def contrastive_loss(self, projections1: torch.Tensor, projections2: torch.Tensor) -> torch.Tensor:
        """对比学习损失（InfoNCE）"""
        batch_size = projections1.size(0)
        
        # 计算相似度矩阵
        sim_matrix = torch.mm(projections1, projections2.t()) / self.temperature
        
        # 创建正样本标签（对角线）
        labels = torch.arange(batch_size, device=projections1.device)
        
        # 计算InfoNCE损失
        loss_12 = F.cross_entropy(sim_matrix, labels)
        loss_21 = F.cross_entropy(sim_matrix.t(), labels)
        
        return (loss_12 + loss_21) / 2
    
    def clustering_entropy_loss(self, cluster_probs: torch.Tensor) -> torch.Tensor:
        """聚类熵损失（鼓励确定性分配）"""
        # 计算每个样本的熵
        entropy = -torch.sum(cluster_probs * torch.log(cluster_probs + 1e-8), dim=1)
        return torch.mean(entropy)
    
    def cluster_balance_loss(self, cluster_probs: torch.Tensor) -> torch.Tensor:
        """聚类平衡损失（鼓励均匀分布）"""
        # 计算每个聚类的平均概率
        cluster_means = torch.mean(cluster_probs, dim=0)
        
        # 计算与均匀分布的KL散度
        uniform_dist = torch.ones_like(cluster_means) / cluster_means.size(0)
        kl_div = F.kl_div(torch.log(cluster_means + 1e-8), uniform_dist, reduction='batchmean')
        
        return kl_div
    
    def global_consistency_loss(self, cluster_probs_list: List[torch.Tensor]) -> torch.Tensor:
        """全局一致性损失"""
        if len(cluster_probs_list) < 2:
            return torch.tensor(0.0, device=cluster_probs_list[0].device)
        
        total_loss = 0.0
        count = 0
        
        # 计算所有客户端对之间的一致性损失
        for i in range(len(cluster_probs_list)):
            for j in range(i + 1, len(cluster_probs_list)):
                # 计算聚类分布的平均值
                mean_i = torch.mean(cluster_probs_list[i], dim=0)
                mean_j = torch.mean(cluster_probs_list[j], dim=0)
                
                # 计算KL散度
                kl_div = F.kl_div(
                    torch.log(mean_i + 1e-8),
                    mean_j,
                    reduction='batchmean'
                )
                
                total_loss += kl_div
                count += 1
        
        return total_loss / count if count > 0 else torch.tensor(0.0, device=cluster_probs_list[0].device)
    
    def forward(self, 
                projections_list: List[torch.Tensor],
                cluster_probs_list: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """计算总损失"""
        losses = {}
        total_loss = 0.0
        
        # 1. 对比学习损失
        if len(projections_list) >= 2:
            contrastive_loss = 0.0
            count = 0
            for i in range(len(projections_list)):
                for j in range(i + 1, len(projections_list)):
                    contrastive_loss += self.contrastive_loss(projections_list[i], projections_list[j])
                    count += 1
            
            if count > 0:
                contrastive_loss /= count
                losses['contrastive'] = contrastive_loss
                total_loss += self.contrastive_weight * contrastive_loss
        
        # 2. 聚类损失
        clustering_loss = 0.0
        for cluster_probs in cluster_probs_list:
            # 熵损失
            entropy_loss = self.clustering_entropy_loss(cluster_probs)
            # 平衡损失
            balance_loss = self.cluster_balance_loss(cluster_probs)
            clustering_loss += entropy_loss + balance_loss
        
        clustering_loss /= len(cluster_probs_list)
        losses['clustering'] = clustering_loss
        total_loss += self.clustering_weight * clustering_loss
        
        # 3. 全局一致性损失
        consistency_loss = self.global_consistency_loss(cluster_probs_list)
        losses['consistency'] = consistency_loss
        total_loss += self.consistency_weight * consistency_loss
        
        losses['total'] = total_loss
        return losses

class OrchestraTrainer:
    """Orchestra训练器"""
    
    def __init__(self, 
                 model: OrchestraModel,
                 loss_fn: OrchestraLoss,
                 optimizer: torch.optim.Optimizer,
                 device: torch.device = None):
        
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 移动模型到设备
        self.model.to(self.device)
        
        # 训练历史
        self.training_history = {
            'total_loss': [],
            'contrastive_loss': [],
            'clustering_loss': [],
            'consistency_loss': []
        }
    
    def train_step(self, data_batches: List[torch.Tensor]) -> Dict[str, float]:
        """单步训练"""
        self.model.train()
        self.optimizer.zero_grad()
        
        # 移动数据到设备
        data_batches = [batch.to(self.device) for batch in data_batches]
        
        # 前向传播
        projections_list = []
        cluster_probs_list = []
        
        for batch in data_batches:
            embeddings, cluster_probs, projections = self.model(batch, return_projections=True)
            projections_list.append(projections)
            cluster_probs_list.append(cluster_probs)
        
        # 计算损失
        losses = self.loss_fn(projections_list, cluster_probs_list)
        
        # 反向传播
        losses['total'].backward()
        self.optimizer.step()
        
        # 记录损失
        loss_values = {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in losses.items()}
        
        # 更新训练历史
        for key in self.training_history.keys():
            if key in loss_values:
                self.training_history[key].append(loss_values[key])
        
        return loss_values
    
    def evaluate(self, test_data: torch.Tensor, true_labels: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """评估模型"""
        self.model.eval()
        
        with torch.no_grad():
            test_data = test_data.to(self.device)
            embeddings, cluster_probs, _ = self.model(test_data)
            predicted_clusters = torch.argmax(cluster_probs, dim=1)
        
        results = {
            'num_samples': len(test_data),
            'num_clusters_used': len(torch.unique(predicted_clusters)),
            'cluster_entropy': torch.mean(-torch.sum(cluster_probs * torch.log(cluster_probs + 1e-8), dim=1)).item()
        }
        
        if true_labels is not None:
            try:
                from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
                
                true_labels_np = true_labels.cpu().numpy()
                predicted_clusters_np = predicted_clusters.cpu().numpy()
                embeddings_np = embeddings.cpu().numpy()
                
                # 聚类评估指标
                ari = adjusted_rand_score(true_labels_np, predicted_clusters_np)
                nmi = normalized_mutual_info_score(true_labels_np, predicted_clusters_np)
                
                # 轮廓系数
                if len(np.unique(predicted_clusters_np)) > 1:
                    silhouette = silhouette_score(embeddings_np, predicted_clusters_np)
                else:
                    silhouette = -1.0
                
                results.update({
                    'adjusted_rand_score': ari,
                    'normalized_mutual_info': nmi,
                    'silhouette_score': silhouette
                })
            except ImportError:
                logging.warning("sklearn not available, skipping clustering metrics")
        
        return results
    
    def get_embeddings(self, data: torch.Tensor) -> torch.Tensor:
        """获取数据的嵌入表示"""
        self.model.eval()
        
        with torch.no_grad():
            data = data.to(self.device)
            embeddings, _, _ = self.model(data)
        
        return embeddings.cpu()
    
    def get_cluster_assignments(self, data: torch.Tensor) -> torch.Tensor:
        """获取聚类分配"""
        self.model.eval()
        
        with torch.no_grad():
            data = data.to(self.device)
            assignments = self.model.get_cluster_assignments(data)
        
        return assignments.cpu()