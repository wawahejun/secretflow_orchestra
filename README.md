# Orchestra: Unsupervised Federated Learning via Globally Consistent Clustering

本项目在SecretFlow框架下实现了Orchestra论文的算法，并在CIFAR-10数据集上复现了实验。

## 项目结构

```
secretflow_orchestra/
├── README.md                    # 项目说明文档
├── config.py                    # 实验配置参数
├── data_utils.py               # 数据处理工具
├── models.py                   # Orchestra模型实现
├── evaluation.py               # 模型评估工具
├── run_experiment.py           # 简化实验脚本
├── secretflow_builtin_orchestra_experiment.py  # SecretFlow集成实验
└── data/                       # 数据目录
    └── cifar-10-batches-py/    # CIFAR-10数据集
```

## 核心功能

### 1. Orchestra模型实现
- **ResNet骨干网络**: 支持ResNet-18/34/50，针对CIFAR-10优化
- **投影网络**: 多层感知机，用于特征投影
- **目标网络**: 使用EMA更新的目标网络
- **Sinkhorn-Knopp算法**: 实现等大小聚类
- **旋转预测**: 抗退化机制
- **对比学习**: 实例级对比损失
- **聚类损失**: 本地和全局聚类一致性

### 2. 联邦学习支持
- **非IID数据分割**: 使用Dirichlet分布
- **联邦平均**: FedAvg算法
- **客户端选择**: 支持随机选择
- **模型聚合**: 参数平均和聚类中心同步

### 3. 评估工具
- **线性评估**: 使用逻辑回归评估特征质量
- **聚类评估**: K-means聚类，计算ARI、NMI等指标
- **可视化**: 混淆矩阵、特征分布、训练曲线

## 快速开始

### 1. 环境要求

```bash
# Python 3.8+
pip install torch torchvision
pip install numpy scipy scikit-learn
pip install matplotlib seaborn
pip install secretflow  # 如果使用SecretFlow集成
```

### 2. 数据准备

CIFAR-10数据集已包含在`data/`目录中。如需重新下载：

```python
import torchvision
torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
```

### 3. 运行实验

#### SecretFlow集成实验

```bash
python secretflow_builtin_orchestra_experiment.py
```

## 配置说明

### 预定义配置

- **small**: 小规模实验，适合快速测试
  - 2个客户端，5轮训练
  - 本地聚类数: 20，全局聚类数: 10
  - 内存大小: 1024

- **medium**: 中等规模实验
  - 5个客户端，10轮训练
  - 本地聚类数: 50，全局聚类数: 10
  - 内存大小: 2048

- **large**: 大规模实验
  - 10个客户端，20轮训练
  - 本地聚类数: 100，全局聚类数: 10
  - 内存大小: 4096

### 关键参数

```python
config = {
    # 基础配置
    'num_clients': 5,           # 客户端数量
    'num_rounds': 10,           # 联邦学习轮数
    'local_epochs': 5,          # 本地训练轮数
    'batch_size': 64,           # 批大小
    'learning_rate': 0.001,     # 学习率
    
    # Orchestra参数
    'temperature': 0.1,         # 对比学习温度
    'clustering_weight': 1.0,   # 聚类损失权重
    'contrastive_weight': 1.0,  # 对比损失权重
    'rotation_weight': 0.5,     # 旋转损失权重
    'num_local_clusters': 50,   # 本地聚类数
    'num_global_clusters': 10,  # 全局聚类数
    'memory_size': 2048,        # 投影内存大小
    'ema_decay': 0.99,          # EMA衰减率
    
    # 数据配置
    'alpha': 0.5,               # Dirichlet分布参数（越小越异构）
    'seed': 42,                 # 随机种子
}
```

## 实验结果

### 预期性能指标

在CIFAR-10数据集上，Orchestra模型的预期性能：

- **线性评估准确率**: 70-80%
- **聚类ARI**: 0.3-0.5
- **聚类NMI**: 0.4-0.6

### 结果分析

1. **对比学习效果**: 通过实例级对比损失学习有意义的特征表示
2. **聚类一致性**: 本地和全局聚类保持一致，避免客户端漂移
3. **抗退化机制**: 旋转预测任务防止表示退化
4. **联邦学习**: 在保护隐私的同时实现全局一致的聚类

## 文件说明

### 核心模块

1. **models.py**: Orchestra模型的完整实现
   - `OrchestraModel`: 主模型类
   - `ResNet`: 骨干网络
   - `SinkhornKnopp`: 等大小聚类算法
   - `ProjectionMLP`: 投影网络

2. **data_utils.py**: 数据处理工具
   - `CIFAR10FederatedDataset`: 联邦数据集类
   - `create_federated_cifar10`: 创建联邦数据分割
   - `create_dirichlet_split`: 非IID数据分割

3. **evaluation.py**: 评估工具
   - `LinearEvaluator`: 线性评估
   - `ClusteringEvaluator`: 聚类评估
   - `VisualizationUtils`: 可视化工具




## 许可证

本项目遵循Apache 2.0许可证。详见LICENSE文件。

## 联系方式

如有问题或建议，请提交Issue或联系项目维护者。