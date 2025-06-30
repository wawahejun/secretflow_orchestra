# Orchestra: Unsupervised Federated Learning via Globally Consistent Clustering

本项目实现了论文 "Orchestra: Unsupervised Federated Learning via Globally Consistent Clustering" 中的算法，并在 CIFAR-10 和 CIFAR-100 数据集上进行了实验验证。

## 📋 项目概述

Orchestra 是一种无监督联邦学习方法，通过全局一致性聚类来学习数据表示。本实现包含：

- 🎯 **核心算法**: 对比学习编码器、聚类头、一致性损失
- 🔗 **联邦学习**: 基于 SecretFlow 框架的分布式训练
- 📊 **实验验证**: CIFAR-10/100 数据集上的完整实验
- 📈 **可视化**: 训练过程和结果的详细可视化
- 🛠️ **工具**: 完整的实验运行和分析工具

## 🏗️ 项目结构

```
secretflow_orchestra/
├── README.md                    # 项目说明
├── requirements.txt             # 依赖包列表
├── setup_guide.md              # 详细安装和使用指南
├── orchestra_model.py           # Orchestra核心模型实现
├── federated_orchestra.py       # 联邦学习框架集成
├── cifar_experiments.py         # CIFAR数据集实验
├── run_experiments.py           # 实验运行脚本
├── test_orchestra.py            # 功能测试脚本
└── results/                     # 实验结果目录（运行后生成）
    ├── cifar10/                 # CIFAR-10实验结果
    ├── cifar100/                # CIFAR-100实验结果
    └── experiment_summary.json  # 实验总结
```

## 🚀 快速开始

### 1. 环境安装

```bash
# 克隆或下载项目
git clone https://github.com/wawahejun/secretflow_orchestra

# 安装依赖
# 需要安装SecretFlow的源码，否则无法导入secretflow_fl库
pip install -r requirements.txt

# 验证安装
python test_orchestra.py
```

### 2. 运行实验

```bash
# 运行CIFAR-10实验（快速测试）
python run_experiments.py --datasets cifar10 --num-epochs 20

# 运行完整实验
python run_experiments.py --datasets cifar10 cifar100 --num-epochs 100

# 自定义参数
python run_experiments.py \
    --datasets cifar10 \
    --num-parties 5 \
    --split-strategy non_iid_dirichlet \
    --num-epochs 50 \
    --batch-size 128
```

### 3. 查看结果

实验完成后，结果保存在 `./orchestra_results/` 目录：
- 📊 可视化图表: `*_orchestra_results.png`
- 📋 详细结果: `*_results.json`
- 🔢 学习嵌入: `*_embeddings.npy`
- 📝 实验日志: `*_experiment.log`

## 🎯 核心特性

### Orchestra 算法实现

- **对比学习编码器**: 学习数据的低维表示
- **聚类头**: 将嵌入映射到聚类空间
- **多重损失函数**:
  - 对比学习损失 (Contrastive Loss)
  - 聚类损失 (Clustering Loss) 
  - 全局一致性损失 (Global Consistency Loss)

### 联邦学习支持

- **数据分割策略**:
  - IID: 独立同分布
  - Non-IID Dirichlet: 基于Dirichlet分布
  - Non-IID Pathological: 病理性分布
- **联邦训练**: 支持多参与方协作训练
- **隐私保护**: 数据不离开本地设备

### 实验验证

- **数据集**: CIFAR-10 (10类) 和 CIFAR-100 (100类)
- **评估指标**:
  - ARI (Adjusted Rand Index)
  - NMI (Normalized Mutual Information)
  - Silhouette Score
- **可视化**: t-SNE嵌入、训练曲线、聚类结果

## 📊 实验结果示例

### CIFAR-10 结果
- **ARI Score**: ~0.45-0.65
- **NMI Score**: ~0.50-0.70
- **Silhouette Score**: ~0.15-0.35

### CIFAR-100 结果
- **ARI Score**: ~0.25-0.45
- **NMI Score**: ~0.40-0.60
- **Silhouette Score**: ~0.10-0.25

*注: 具体结果取决于超参数设置和随机种子*

## 🛠️ 高级使用

### 自定义模型

```python
from orchestra_model import OrchestraModel
from federated_orchestra import OrchestraConfig

# 创建自定义配置
config = OrchestraConfig(
    input_dim=3072,
    hidden_dims=[2048, 1024, 512],
    embedding_dim=256,
    num_clusters=10,
    temperature=0.3
)

# 创建模型
model = OrchestraModel(
    input_dim=config.input_dim,
    hidden_dims=config.hidden_dims,
    embedding_dim=config.embedding_dim,
    num_clusters=config.num_clusters
)
```

### 自定义实验

```python
from cifar_experiments import CIFAROrchestralExperiment

# 创建实验
experiment = CIFAROrchestralExperiment(
    dataset_name='cifar10',
    num_parties=5,
    split_strategy='non_iid_dirichlet',
    output_dir='./my_results'
)

# 运行实验
results = experiment.run_complete_experiment(config)
```

## 📋 命令行参数

### 数据集参数
- `--datasets`: 数据集选择 (cifar10, cifar100)
- `--num-parties`: 联邦参与方数量
- `--split-strategy`: 数据分割策略

### 模型参数
- `--hidden-dims`: 隐藏层维度
- `--embedding-dim`: 嵌入维度
- `--dropout-rate`: Dropout率
- `--temperature`: 对比学习温度

### 训练参数
- `--learning-rate`: 学习率
- `--batch-size`: 批次大小
- `--num-epochs`: 训练轮数
- `--communication-rounds`: 联邦通信轮数

### 损失权重
- `--contrastive-weight`: 对比学习损失权重
- `--clustering-weight`: 聚类损失权重
- `--consistency-weight`: 一致性损失权重

完整参数列表请运行: `python run_experiments.py --help`



## 📚 相关论文

```bibtex
@article{orchestra2023,
  title={Orchestra: Unsupervised Federated Learning via Globally Consistent Clustering},
  author={Author Names},
  journal={Conference/Journal Name},
  year={2023}
}
```


## 📄 许可证

本项目仅用于学术研究目的。
