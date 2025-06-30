#!/usr/bin/env python3
"""
Orchestra联邦学习实现
基于SecretFlow框架的分布式Orchestra算法
"""

import secretflow as sf
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass
from secretflow.device import PYU, SPU
from secretflow.data import FedNdarray, PartitionWay
from secretflow_fl.ml.nn import FLModel
from secretflow_fl.ml.nn.core.torch import TorchModel, metric_wrapper, optim_wrapper
from secretflow_fl.ml.nn.callbacks.history import History
from secretflow_fl.ml.nn.fl.backend.torch.strategy import PYUFedAvgW

from orchestra_model import OrchestraModel, OrchestraLoss, OrchestraTrainer
# 导入自定义Orchestra策略
from orchestra_strategy import OrchestraStrategy, PYUOrchestraStrategy

@dataclass
class OrchestraConfig:
    """Orchestra配置类"""
    input_dim: int
    hidden_dims: List[int] = None
    embedding_dim: int = 128
    num_clusters: int = 10
    dropout_rate: float = 0.2
    temperature: float = 0.5
    
    # 损失权重
    contrastive_weight: float = 1.0
    clustering_weight: float = 1.0
    consistency_weight: float = 1.0
    
    # 训练参数
    learning_rate: float = 0.001
    batch_size: int = 256
    num_epochs: int = 100
    
    # 联邦学习参数
    aggregation_strategy: str = 'fedavg'  # 'fedavg', 'fedprox'
    communication_rounds: int = 10
    local_epochs: int = 5
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [512, 256]

def create_orchestra_model_builder(config: OrchestraConfig):
    """创建Orchestra模型构建器函数"""
    def model_builder():
        return OrchestraModel(
            input_dim=config.input_dim,
            hidden_dims=config.hidden_dims,
            embedding_dim=config.embedding_dim,
            num_clusters=config.num_clusters,
            dropout_rate=config.dropout_rate,
            temperature=config.temperature
        )
    return model_builder

def create_orchestra_loss_builder(config: OrchestraConfig):
    """创建Orchestra损失函数构建器"""
    def loss_builder():
        return OrchestraLoss(
            contrastive_weight=config.contrastive_weight,
            clustering_weight=config.clustering_weight,
            consistency_weight=config.consistency_weight,
            temperature=config.temperature
        )
    return loss_builder

def create_orchestra_torch_model(config: OrchestraConfig):
    """创建Orchestra TorchModel实例"""
    model_builder = create_orchestra_model_builder(config)
    loss_builder = create_orchestra_loss_builder(config)
    optimizer = optim_wrapper(torch.optim.Adam, lr=config.learning_rate)
    
    return TorchModel(
        model_fn=model_builder,
        loss_fn=loss_builder,
        optim_fn=optimizer,
        metrics=[]
    )

def create_simple_orchestra_model(config: OrchestraConfig):
    """创建简单的Orchestra模型函数"""
    def model_fn():
        from orchestra_model import OrchestraModel
        return OrchestraModel(
            input_dim=config.input_dim,
            hidden_dims=config.hidden_dims,
            embedding_dim=config.embedding_dim,
            num_clusters=config.num_clusters,
            dropout_rate=config.dropout_rate,
            temperature=config.temperature
        )
    return model_fn

class FederatedOrchestraTrainer:
    """联邦Orchestra训练器"""
    
    def __init__(self, 
                 config: OrchestraConfig,
                 parties: List[str],
                 spu_config: Optional[Dict] = None):
        
        self.config = config
        self.parties = parties
        self.num_parties = len(parties)
        
        # 初始化SecretFlow
        try:
            sf.init(parties=parties, address='local')
        except Exception as e:
            # 如果已经初始化过，忽略错误
            pass
        
        # 创建设备
        self.devices = {party: PYU(party) for party in parties}
        
        # 创建SPU设备（如果需要）
        if spu_config:
            self.spu = SPU(spu_config)
        else:
            self.spu = None
        
        # 创建联邦模型
        self.fed_model = None
        self.training_history = History()
        
        # 日志设置
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def setup_model(self) -> FLModel:
        """设置联邦模型"""
        # 创建Orchestra TorchModel
        torch_model = create_orchestra_torch_model(self.config)
        
        # 创建联邦模型，使用Orchestra策略和聚合器
        self.fed_model = FLModel(
            device_list=list(self.devices.values()),
            model=torch_model,
            strategy="orchestra",
            aggregator=PYUFedAvgW(),
            backend='torch',
            random_seed=42,
            model_init_params={}
        )
        
        return self.fed_model
    
    def prepare_data(self, 
                     data_dict: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> Dict[str, FedNdarray]:
        """准备联邦数据"""
        fed_data = {}
        
        for party, (x_data, y_data) in data_dict.items():
            if party not in self.devices:
                raise ValueError(f"Party {party} not in devices")
            
            # 将数据移动到对应设备
            device = self.devices[party]
            
            try:
                # 确保数据是正确的numpy格式
                x_data = np.array(x_data, dtype=np.float32)
                y_data = np.array(y_data, dtype=np.int64)
                
                # 创建联邦数据 - 直接使用numpy数组
                fed_x = FedNdarray(
                    partitions={device: device(lambda x=x_data: x)},
                    partition_way=PartitionWay.VERTICAL
                )
                
                fed_y = FedNdarray(
                    partitions={device: device(lambda y=y_data: y)},
                    partition_way=PartitionWay.VERTICAL
                )
                
                fed_data[party] = (fed_x, fed_y)
                
            except Exception as e:
                self.logger.error(f"准备{party}数据时出错: {e}")
                raise
        
        return fed_data
    
    def custom_loss_function(self, y_true, y_pred, sample_weight=None):
        """自定义损失函数"""
        # y_pred包含 (embeddings, cluster_probs, projections)
        embeddings, cluster_probs, projections = y_pred
        
        # 这里需要实现Orchestra特定的损失计算
        # 由于联邦学习的限制，我们简化损失计算
        
        # 聚类熵损失
        entropy_loss = -torch.sum(cluster_probs * torch.log(cluster_probs + 1e-8), dim=1)
        entropy_loss = torch.mean(entropy_loss)
        
        # 聚类平衡损失
        cluster_means = torch.mean(cluster_probs, dim=0)
        uniform_dist = torch.ones_like(cluster_means) / cluster_means.size(0)
        balance_loss = torch.nn.functional.kl_div(
            torch.log(cluster_means + 1e-8), 
            uniform_dist, 
            reduction='batchmean'
        )
        
        return entropy_loss + balance_loss
    
    def train(self, 
              fed_data: Dict[str, FedNdarray],
              validation_data: Optional[Dict[str, FedNdarray]] = None) -> History:
        """训练联邦Orchestra模型"""
        
        if self.fed_model is None:
            self.setup_model()
        
        self.logger.info(f"开始联邦Orchestra训练，共{self.config.communication_rounds}轮通信")
        
        # 准备训练数据
        train_x_list = []
        train_y_list = []
        
        for party in self.parties:
            if party in fed_data:
                x_data, y_data = fed_data[party]
                train_x_list.append(x_data)
                train_y_list.append(y_data)
        
        # 使用第一个参与方的数据（简化处理）
        # 在实际联邦学习中，数据不会合并，而是分布式训练
        fed_train_x = train_x_list[0] if train_x_list else None
        fed_train_y = train_y_list[0] if train_y_list else None
        
        if fed_train_x is None or fed_train_y is None:
            raise ValueError("没有可用的训练数据")
        
        # 训练参数
        train_params = {
            'x': fed_train_x,
            'y': fed_train_y,
            'batch_size': self.config.batch_size,
            'epochs': self.config.local_epochs,
            'verbose': 1,
            'validation_data': validation_data,
            'shuffle': True
        }
        
        # 执行联邦训练
        for round_idx in range(self.config.communication_rounds):
            self.logger.info(f"通信轮次 {round_idx + 1}/{self.config.communication_rounds}")
            
            # 本地训练
            history = self.fed_model.fit(**train_params)
            
            # 记录训练历史
            try:
                if history and hasattr(history, 'history') and history.history:
                    self.training_history.merge(history)
                else:
                    self.logger.warning("本轮训练没有产生有效的训练历史")
            except Exception as e:
                self.logger.warning(f"合并训练历史时出错: {e}")
            
            # 评估（如果有验证数据）
            if validation_data:
                val_metrics = self.evaluate(validation_data)
                self.logger.info(f"验证指标: {val_metrics}")
        
        return self.training_history
    
    def evaluate(self, test_data: Dict[str, FedNdarray]) -> Dict[str, float]:
        """评估模型"""
        if self.fed_model is None:
            raise ValueError("模型未训练")
        
        # 准备测试数据
        test_x_list = []
        test_y_list = []
        
        for party in self.parties:
            if party in test_data:
                x_data, y_data = test_data[party]
                test_x_list.append(x_data)
                test_y_list.append(y_data)
        
        # 使用第一个参与方的测试数据（简化处理）
        fed_test_x = test_x_list[0] if test_x_list else None
        fed_test_y = test_y_list[0] if test_y_list else None
        
        if fed_test_x is None or fed_test_y is None:
            raise ValueError("没有可用的测试数据")
        
        # 评估
        metrics = self.fed_model.evaluate(
            x=fed_test_x,
            y=fed_test_y,
            batch_size=self.config.batch_size
        )
        
        return metrics
    
    def predict(self, data: FedNdarray) -> FedNdarray:
        """预测"""
        if self.fed_model is None:
            raise ValueError("模型未训练")
        
        return self.fed_model.predict(data, batch_size=self.config.batch_size)
    
    def get_model_weights(self) -> Dict[str, Any]:
        """获取模型权重"""
        if self.fed_model is None:
            raise ValueError("模型未训练")
        
        return self.fed_model.get_weights()
    
    def save_model(self, filepath: str):
        """保存模型"""
        if self.fed_model is None:
            raise ValueError("模型未训练")
        
        self.fed_model.save_model(filepath)
        self.logger.info(f"模型已保存到: {filepath}")
    
    def load_model(self, filepath: str):
        """加载模型"""
        if self.fed_model is None:
            self.setup_model()
        
        self.fed_model.load_model(filepath)
        self.logger.info(f"模型已从 {filepath} 加载")

class OrchestraDataProcessor:
    """Orchestra数据处理器"""
    
    @staticmethod
    def create_federated_split(data: np.ndarray, 
                              labels: np.ndarray, 
                              num_parties: int,
                              split_strategy: str = 'iid') -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """创建联邦数据分割"""
        
        num_samples = len(data)
        party_data = {}
        
        if split_strategy == 'iid':
            # IID分割
            indices = np.random.permutation(num_samples)
            samples_per_party = num_samples // num_parties
            
            for i in range(num_parties):
                party_name = f'party_{i}'
                start_idx = i * samples_per_party
                
                if i == num_parties - 1:
                    end_idx = num_samples
                else:
                    end_idx = (i + 1) * samples_per_party
                
                party_indices = indices[start_idx:end_idx]
                party_data[party_name] = (data[party_indices], labels[party_indices])
        
        elif split_strategy == 'non_iid':
            # Non-IID分割（按标签分割）
            unique_labels = np.unique(labels)
            labels_per_party = len(unique_labels) // num_parties
            
            for i in range(num_parties):
                party_name = f'party_{i}'
                start_label = i * labels_per_party
                
                if i == num_parties - 1:
                    end_label = len(unique_labels)
                else:
                    end_label = (i + 1) * labels_per_party
                
                party_labels = unique_labels[start_label:end_label]
                party_indices = np.isin(labels, party_labels)
                
                party_data[party_name] = (data[party_indices], labels[party_indices])
        
        else:
            raise ValueError(f"未知的分割策略: {split_strategy}")
        
        return party_data
    
    @staticmethod
    def preprocess_cifar_data(data: np.ndarray) -> np.ndarray:
        """预处理CIFAR数据"""
        # 归一化到[-1, 1]
        data = data.astype(np.float32) / 255.0
        data = (data - 0.5) / 0.5
        
        # 展平为向量
        if len(data.shape) > 2:
            data = data.reshape(data.shape[0], -1)
        
        return data
    
    @staticmethod
    def add_data_augmentation(data: np.ndarray, 
                            augmentation_factor: float = 0.1) -> np.ndarray:
        """添加数据增强"""
        noise = np.random.normal(0, augmentation_factor, data.shape)
        return data + noise.astype(data.dtype)

def create_orchestra_experiment(config: OrchestraConfig, 
                              parties: List[str]) -> FederatedOrchestraTrainer:
    """创建Orchestra实验"""
    trainer = FederatedOrchestraTrainer(config=config, parties=parties)
    return trainer

# 示例使用
if __name__ == "__main__":
    # 配置
    config = OrchestraConfig(
        input_dim=3072,  # CIFAR-10/100: 32*32*3
        hidden_dims=[1024, 512, 256],
        embedding_dim=128,
        num_clusters=10,
        learning_rate=0.001,
        batch_size=256,
        communication_rounds=20,
        local_epochs=5
    )
    
    # 参与方
    parties = ['alice', 'bob', 'charlie']
    
    # 创建训练器
    trainer = create_orchestra_experiment(config, parties)
    
    print("Orchestra联邦学习训练器已创建")
    print(f"参与方: {parties}")
    print(f"配置: {config}")