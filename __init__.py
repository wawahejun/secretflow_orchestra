#!/usr/bin/env python3
"""
Orchestra: Unsupervised Federated Learning via Globally Consistent Clustering
基于SecretFlow框架的Orchestra算法实现

论文参考: Orchestra: Unsupervised Federated Learning via Globally Consistent Clustering
"""

from .orchestra_model import (
    ContrastiveEncoder,
    ClusteringHead,
    OrchestraModel,
    OrchestraLoss,
    OrchestraTrainer
)

from .federated_orchestra import (
    OrchestraConfig,
    FederatedOrchestraModel,
    FederatedOrchestraTrainer,
    OrchestraDataProcessor
)

from .cifar_experiments import (
    CIFAROrchestralExperiment,
    run_cifar_experiments
)

__version__ = "1.0.0"
__author__ = "SecretFlow Orchestra Team"

__all__ = [
    # 核心模型
    "ContrastiveEncoder",
    "ClusteringHead", 
    "OrchestraModel",
    "OrchestraLoss",
    "OrchestraTrainer",
    
    # 联邦学习组件
    "OrchestraConfig",
    "FederatedOrchestraModel",
    "FederatedOrchestraTrainer",
    "OrchestraDataProcessor",
    
    # 实验组件
    "CIFAROrchestralExperiment",
    "run_cifar_experiments",
]