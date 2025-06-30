# -*- coding: utf-8 -*-
"""
Orchestra联邦学习策略实现
基于论文: Orchestra: Unsupervised Federated Learning via Globally Consistent Clustering
"""

import copy
from typing import Tuple, List, Optional
import logging

import numpy as np
import torch
import torch.nn.functional as F

from secretflow_fl.ml.nn.fl.backend.torch.fl_base import BaseTorchModel
from secretflow_fl.ml.nn.fl.strategy_dispatcher import register_strategy


class OrchestraStrategy(BaseTorchModel):
    """
    Orchestra联邦学习策略：实现无监督聚类的联邦学习
    
    核心特性：
    1. 无监督学习：不需要标签数据
    2. 全局一致性聚类：确保各客户端聚类结果一致
    3. 对比学习：通过对比损失学习特征表示
    4. 聚类损失：通过聚类中心学习数据分布
    """
    
    def __init__(self, builder_base, **kwargs):
        # 正确传递builder_base参数给BaseTorchModel
        logging.warning(f"OrchestraStrategy.__init__ called. builder_base type: {type(builder_base)}, kwargs keys: {list(kwargs.keys())}")
        super().__init__(builder_base=builder_base, **kwargs)
        self.logger = logging.getLogger(__name__)
        
        # Orchestra特定参数
        self.temperature = kwargs.get('temperature', 0.5)  # 对比学习温度参数
        self.cluster_weight = kwargs.get('cluster_weight', 1.0)  # 聚类损失权重
        self.contrastive_weight = kwargs.get('contrastive_weight', 1.0)  # 对比损失权重
        
        # 全局聚类中心（由服务器维护）
        self.global_cluster_centers = None
        self.num_clusters = kwargs.get('num_clusters', 10)
        
    def train_step(
        self,
        weights: np.ndarray,
        cur_steps: int,
        train_steps: int,
        **kwargs,
    ) -> Tuple[np.ndarray, int]:
        """
        Orchestra训练步骤
        
        Args:
            weights: 全局权重（包含模型参数和聚类中心）
            cur_steps: 当前训练步数
            train_steps: 本地训练步数
            kwargs: 策略特定参数
        Returns:
            训练后的参数和样本数量
        """
        assert self.model is not None, "Model cannot be none"
        self.model.train()
        
        # 刷新数据迭代器
        refresh_data = kwargs.get("refresh_data", False)
        if refresh_data:
            self._reset_data_iter()
            
        # 应用全局权重
        if weights is not None:
            self.set_weights(weights)
            
        # 获取全局聚类中心
        self.global_cluster_centers = kwargs.get('global_cluster_centers', None)
        
        num_sample = 0
        total_loss = 0.0
        total_cluster_loss = 0.0
        total_contrastive_loss = 0.0
        
        self.logger.info(f"开始本地训练，训练步数: {train_steps}")
        
        for step in range(train_steps):
            try:
                x, y, s_w = self.next_batch()
                num_sample += x.shape[0]
                
                # Orchestra特定的前向传播
                # 获取特征表示和投影
                if hasattr(self.model, 'forward_orchestra'):
                    features, projections = self.model.forward_orchestra(x)
                else:
                    # 兼容性处理：如果模型没有orchestra特定方法
                    output = self.model(x)
                    if isinstance(output, tuple) and len(output) == 2:
                        features, projections = output
                    else:
                        features = output
                        projections = features  # 使用相同的特征作为投影
                
                # 计算Orchestra损失
                cluster_loss = self._compute_cluster_loss(features)
                contrastive_loss = self._compute_contrastive_loss(projections)
                
                # 总损失
                loss = (self.cluster_weight * cluster_loss + 
                       self.contrastive_weight * contrastive_loss)
                
                # 反向传播
                if self.model.automatic_optimization:
                    self.model.backward_step(loss)
                else:
                    # 手动优化
                    loss.backward()
                    if hasattr(self.model, 'optimizer'):
                        self.model.optimizer.step()
                        self.model.optimizer.zero_grad()
                
                total_loss += loss.item()
                total_cluster_loss += cluster_loss.item()
                total_contrastive_loss += contrastive_loss.item()
                
            except Exception as e:
                self.logger.error(f"训练步骤 {step} 出错: {str(e)}")
                continue
        
        # 计算平均损失
        avg_loss = total_loss / train_steps if train_steps > 0 else 0.0
        avg_cluster_loss = total_cluster_loss / train_steps if train_steps > 0 else 0.0
        avg_contrastive_loss = total_contrastive_loss / train_steps if train_steps > 0 else 0.0
        
        # 记录训练日志
        logs = {
            "train-loss": avg_loss,
            "cluster-loss": avg_cluster_loss,
            "contrastive-loss": avg_contrastive_loss,
            "num_samples": num_sample
        }
        
        self.logs = self.transform_metrics(logs)
        self.wrapped_metrics.extend(self.wrap_local_metrics())
        self.epoch_logs = copy.deepcopy(self.logs)
        
        self.logger.info(f"本地训练完成，平均损失: {avg_loss:.4f}, 样本数: {num_sample}")
        
        # 返回更新后的模型权重
        model_weights = self.get_weights(return_numpy=True)
        return model_weights, num_sample
    
    def _compute_cluster_loss(self, features: torch.Tensor) -> torch.Tensor:
        """
        计算聚类损失
        
        Args:
            features: 特征表示 [batch_size, feature_dim]
        Returns:
            聚类损失
        """
        if self.global_cluster_centers is None:
            # 如果没有全局聚类中心，使用随机初始化
            feature_dim = features.shape[-1]
            self.global_cluster_centers = torch.randn(
                self.num_clusters, feature_dim, 
                device=features.device
            )
        
        # 确保聚类中心在正确的设备上
        if isinstance(self.global_cluster_centers, np.ndarray):
            cluster_centers = torch.from_numpy(self.global_cluster_centers).to(features.device)
        else:
            cluster_centers = self.global_cluster_centers.to(features.device)
        
        # 计算特征到聚类中心的距离
        # features: [batch_size, feature_dim]
        # cluster_centers: [num_clusters, feature_dim]
        distances = torch.cdist(features, cluster_centers)  # [batch_size, num_clusters]
        
        # 软分配：使用softmax计算分配概率
        assignments = F.softmax(-distances / self.temperature, dim=1)
        
        # 聚类损失：最小化特征到最近聚类中心的距离
        min_distances, _ = torch.min(distances, dim=1)
        cluster_loss = torch.mean(min_distances)
        
        return cluster_loss
    
    def _compute_contrastive_loss(self, projections: torch.Tensor) -> torch.Tensor:
        """
        计算对比学习损失
        
        Args:
            projections: 投影特征 [batch_size, projection_dim]
        Returns:
            对比损失
        """
        batch_size = projections.shape[0]
        
        if batch_size < 2:
            return torch.tensor(0.0, device=projections.device)
        
        # 归一化投影
        projections = F.normalize(projections, dim=1)
        
        # 计算相似度矩阵
        similarity_matrix = torch.matmul(projections, projections.T) / self.temperature
        
        # 创建正样本掩码（这里简化处理，实际应该基于数据增强）
        # 假设batch中相邻的样本是正样本对
        positive_mask = torch.zeros_like(similarity_matrix, dtype=torch.bool)
        for i in range(0, batch_size - 1, 2):
            if i + 1 < batch_size:
                positive_mask[i, i + 1] = True
                positive_mask[i + 1, i] = True
        
        # 负样本掩码
        negative_mask = ~positive_mask
        # 移除对角线（自己与自己的相似度）
        negative_mask.fill_diagonal_(False)
        
        # 如果没有正样本对，返回零损失
        if not positive_mask.any():
            return torch.tensor(0.0, device=projections.device)
        
        # 计算对比损失
        positive_similarities = similarity_matrix[positive_mask]
        negative_similarities = similarity_matrix[negative_mask].view(batch_size, -1)
        
        # InfoNCE损失
        logits = torch.cat([positive_similarities.unsqueeze(1), negative_similarities], dim=1)
        labels = torch.zeros(positive_similarities.shape[0], dtype=torch.long, device=projections.device)
        
        contrastive_loss = F.cross_entropy(logits, labels)
        
        return contrastive_loss
    
    def apply_weights(self, weights, **kwargs):
        """
        应用全局权重到本地模型
        
        Args:
            weights: 全局权重
            kwargs: 额外参数，可能包含聚类中心
        """
        if weights is not None:
            self.set_weights(weights)
        
        # 更新全局聚类中心
        if 'global_cluster_centers' in kwargs:
            self.global_cluster_centers = kwargs['global_cluster_centers']
    
    def get_cluster_assignments(self, data_loader=None) -> np.ndarray:
        """
        获取数据的聚类分配
        
        Args:
            data_loader: 数据加载器，如果为None则使用训练数据
        Returns:
            聚类分配数组
        """
        self.model.eval()
        assignments = []
        
        if data_loader is None:
            data_loader = self.train_data_loader
        
        with torch.no_grad():
            for batch in data_loader:
                if isinstance(batch, (list, tuple)):
                    x = batch[0]
                else:
                    x = batch
                
                # 获取特征
                if hasattr(self.model, 'forward_orchestra'):
                    features, _ = self.model.forward_orchestra(x)
                else:
                    features = self.model(x)
                    if isinstance(features, tuple):
                        features = features[0]
                
                # 计算到聚类中心的距离
                if self.global_cluster_centers is not None:
                    if isinstance(self.global_cluster_centers, np.ndarray):
                        cluster_centers = torch.from_numpy(self.global_cluster_centers).to(features.device)
                    else:
                        cluster_centers = self.global_cluster_centers.to(features.device)
                    
                    distances = torch.cdist(features, cluster_centers)
                    batch_assignments = torch.argmin(distances, dim=1)
                    assignments.append(batch_assignments.cpu().numpy())
        
        return np.concatenate(assignments) if assignments else np.array([])


@register_strategy(strategy_name="orchestra", backend="torch")
class PYUOrchestraStrategy(OrchestraStrategy):
    """
    Orchestra策略的PYU包装类
    用于在SecretFlow的PYU设备上运行
    """
    pass


# 为了向后兼容，也注册一个简化版本
@register_strategy(strategy_name="orchestra_simple", backend="torch")
class PYUOrchestraSimpleStrategy(BaseTorchModel):
    """
    简化版Orchestra策略
    适用于快速原型开发
    """
    
    def __init__(self, builder_base, **kwargs):
        # 正确传递builder_base参数给BaseTorchModel
        super().__init__(builder_base=builder_base, **kwargs)
    
    def train_step(
        self,
        weights: np.ndarray,
        cur_steps: int,
        train_steps: int,
        **kwargs,
    ) -> Tuple[np.ndarray, int]:
        """简化的训练步骤"""
        assert self.model is not None, "Model cannot be none"
        self.model.train()
        
        if weights is not None:
            self.set_weights(weights)
            
        num_sample = 0
        total_loss = 0.0
        
        for step in range(train_steps):
            x, y, s_w = self.next_batch()
            num_sample += x.shape[0]
            
            # 简单的前向传播
            loss = self.model.training_step((x, y), cur_steps + step, sample_weight=s_w)
            
            if self.model.automatic_optimization:
                self.model.backward_step(loss)
                
            total_loss += loss.item()
        
        avg_loss = total_loss / train_steps if train_steps > 0 else 0.0
        self.logs = {"train-loss": avg_loss}
        
        model_weights = self.get_weights(return_numpy=True)
        return model_weights, num_sample
    
    def apply_weights(self, weights, **kwargs):
        """应用权重"""
        if weights is not None:
            self.set_weights(weights)