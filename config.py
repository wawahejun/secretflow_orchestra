#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Orchestra CIFAR-10实验配置文件
基于论文原始实现的参数设置
"""

# 基础配置
config_dict = {
    # 数据集配置
    'dataset': 'CIFAR10',
    'data_dir': './data',
    'num_classes': 10,
    
    # 联邦学习配置
    'num_clients': 100,  # 论文中使用100个客户端
    'num_rounds': 100,   # 联邦学习轮数
    'clients_per_round': 10,  # 每轮参与的客户端数量
    'alpha': 0.1,        # Dirichlet分布参数，控制数据异构性
    
    # 训练配置
    'train_mode': 'orchestra',
    'model_type': 'res18',  # ResNet-18
    'batch_size': 16,       # 本地批次大小
    'local_epochs': 10,     # 本地训练轮数
    'learning_rate': 0.003, # 学习率
    'weight_decay': 1e-4,   # 权重衰减
    'momentum': 0.9,        # SGD动量
    'optimizer': 'adam',    # 优化器类型
    
    # Orchestra特定参数
    'temperature': 0.1,           # 温度参数
    'cluster_weight': 1.0,        # 聚类损失权重
    'contrastive_weight': 1.0,    # 对比损失权重
    'deg_weight': 1.0,            # 抗退化损失权重
    'num_local_clusters': 16,     # 本地聚类数量
    'num_global_clusters': 128,   # 全局聚类数量
    'memory_size': 128,           # 投影内存大小
    'ema_decay': 0.996,           # EMA衰减率
    
    # 数据增强配置
    'use_data_augmentation': True,
    'crop_size': 32,
    'crop_padding': 4,
    'horizontal_flip_prob': 0.5,
    'color_jitter': {
        'brightness': 0.4,
        'contrast': 0.4,
        'saturation': 0.4,
        'hue': 0.1
    },
    
    # 归一化参数（CIFAR-10）
    'normalize_mean': [0.4914, 0.4822, 0.4465],
    'normalize_std': [0.2023, 0.1994, 0.2010],
    
    # 设备配置
    'device': 'cuda',  # 优先使用CUDA加速
    'num_workers': 4,
    
    # 日志和保存配置
    'log_interval': 10,
    'save_interval': 20,
    'output_dir': './output',
    'save_model': True,
    'verbose': True,
    
    # 随机种子
    'seed': 42,
}

# 评估配置
eval_dict = {
    # 线性评估配置
    'linear_eval': {
        'batch_size': 256,
        'learning_rate': 0.1,
        'epochs': 100,
        'weight_decay': 0.0,
        'momentum': 0.9,
        'optimizer': 'sgd',
        'scheduler': 'cosine',
    },
    
    # 聚类评估配置
    'clustering_eval': {
        'num_clusters': 10,
        'algorithm': 'kmeans',
        'random_state': 42,
        'max_iter': 300,
    },
    
    # 评估指标
    'metrics': [
        'accuracy',      # 线性评估准确率
        'ari',          # 调整兰德指数
        'nmi',          # 标准化互信息
        'silhouette',   # 轮廓系数
    ],
}

# 不同实验设置的配置变体
experiment_configs = {
    # 基础实验
    'base': config_dict.copy(),
    
    # 小规模实验（用于快速测试）
    'small': {
        **config_dict,
        'num_clients': 4,
        'num_rounds': 5,  # 减少轮数以便快速验证
        'clients_per_round': 4,
        'local_epochs': 5,
        'batch_size': 32,
    },
    
    # 大规模实验
    'large': {
        **config_dict,
        'num_clients': 200,
        'num_rounds': 200,
        'clients_per_round': 20,
        'local_epochs': 15,
    },
    
    # 高异构性实验
    'high_heterogeneity': {
        **config_dict,
        'alpha': 0.01,  # 更低的alpha值表示更高的异构性
    },
    
    # 低异构性实验
    'low_heterogeneity': {
        **config_dict,
        'alpha': 1.0,   # 更高的alpha值表示更低的异构性
    },
    
    # 不同聚类数量的实验
    'more_clusters': {
        **config_dict,
        'num_local_clusters': 32,
        'num_global_clusters': 256,
    },
    
    'fewer_clusters': {
        **config_dict,
        'num_local_clusters': 8,
        'num_global_clusters': 64,
    },
}

# 超参数搜索空间
hyperparameter_search_space = {
    'learning_rate': [0.001, 0.003, 0.01, 0.03],
    'temperature': [0.05, 0.1, 0.2, 0.5],
    'cluster_weight': [0.5, 1.0, 2.0],
    'contrastive_weight': [0.5, 1.0, 2.0],
    'deg_weight': [0.5, 1.0, 2.0],
    'num_local_clusters': [8, 16, 32],
    'num_global_clusters': [64, 128, 256],
    'ema_decay': [0.99, 0.996, 0.999],
    'batch_size': [16, 32, 64],
    'local_epochs': [5, 10, 15],
}


def get_config(experiment_type='base'):
    """获取指定实验类型的配置"""
    if experiment_type in experiment_configs:
        return experiment_configs[experiment_type].copy()
    else:
        raise ValueError(f"未知的实验类型: {experiment_type}")


def get_eval_config():
    """获取评估配置"""
    return eval_dict.copy()


def update_config(config, **kwargs):
    """更新配置参数"""
    config.update(kwargs)
    return config


def print_config(config):
    """打印配置信息"""
    print("=== 实验配置 ===")
    for key, value in config.items():
        if isinstance(value, dict):
            print(f"{key}:")
            for sub_key, sub_value in value.items():
                print(f"  {sub_key}: {sub_value}")
        else:
            print(f"{key}: {value}")
    print("=" * 20)


if __name__ == '__main__':
    # 测试配置
    base_config = get_config('base')
    print_config(base_config)
    
    print("\n")
    
    small_config = get_config('small')
    print_config(small_config)