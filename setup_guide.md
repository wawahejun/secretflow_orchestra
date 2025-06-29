# Orchestra SecretFlow 实现安装和使用指南

## 概述

本项目实现了论文 "Orchestra: Unsupervised Federated Learning via Globally Consistent Clustering" 中的算法，并在 CIFAR-10 和 CIFAR-100 数据集上进行了实验验证。

## 安装步骤

### 1. 环境要求

- Python 3.8+
- CUDA 11.0+ (可选，用于GPU加速)
- 至少 8GB RAM
- 至少 10GB 可用磁盘空间

### 2. 安装依赖

```bash
# 创建虚拟环境（推荐）
python -m venv orchestra_env
source orchestra_env/bin/activate  # Linux/Mac
# 或
orchestra_env\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt

# 如果需要安装SecretFlow（可选）
pip install secretflow
```

### 3. 验证安装

```bash
python -c "import torch; print('PyTorch版本:', torch.__version__)"
python -c "import torchvision; print('TorchVision版本:', torchvision.__version__)"
python -c "import sklearn; print('Scikit-learn版本:', sklearn.__version__)"
```

## 快速开始

### 1. 运行基本实验

```bash
# 运行CIFAR-10实验
python run_experiments.py --datasets cifar10 --num-epochs 50

# 运行CIFAR-100实验
python run_experiments.py --datasets cifar100 --num-epochs 50

# 同时运行两个数据集
python run_experiments.py --datasets cifar10 cifar100 --num-epochs 100
```

### 2. 自定义实验参数

```bash
# 调整联邦学习参数
python run_experiments.py \
    --datasets cifar10 \
    --num-parties 5 \
    --split-strategy non_iid_dirichlet \
    --communication-rounds 30 \
    --local-epochs 3

# 调整模型参数
python run_experiments.py \
    --datasets cifar10 \
    --hidden-dims 2048 1024 512 \
    --embedding-dim 256 \
    --dropout-rate 0.3 \
    --temperature 0.3

# 调整训练参数
python run_experiments.py \
    --datasets cifar10 \
    --learning-rate 0.0005 \
    --batch-size 128 \
    --num-epochs 200
```

### 3. 实验结果

实验完成后，结果将保存在 `./orchestra_results/` 目录下：

```
orchestra_results/
├── experiment_summary.json          # 实验总结
├── orchestra_cifar10_20231201_143022/  # CIFAR-10实验结果
│   ├── experiment_config.json       # 实验配置
│   ├── experiment_metadata.json     # 实验元数据
│   ├── cifar10_results.json         # 详细结果
│   ├── cifar10_embeddings.npy       # 学习到的嵌入
│   ├── cifar10_clusters.npy         # 聚类分配
│   ├── cifar10_orchestra_results.png # 结果可视化
│   └── cifar10_experiment.log       # 实验日志
└── orchestra_cifar100_20231201_143022/ # CIFAR-100实验结果
    └── ...
```

## 高级使用

### 1. 编程接口

```python
from cifar_experiments import CIFAROrchestralExperiment
from federated_orchestra import OrchestraConfig

# 创建配置
config = OrchestraConfig(
    input_dim=3072,
    hidden_dims=[1024, 512, 256],
    embedding_dim=128,
    num_clusters=10,
    num_epochs=100
)

# 创建实验
experiment = CIFAROrchestralExperiment(
    dataset_name='cifar10',
    num_parties=3,
    split_strategy='iid'
)

# 运行实验
centralized_results, federated_results = experiment.run_complete_experiment(config)
```

### 2. 自定义数据分割

```python
from federated_orchestra import OrchestraDataProcessor

# 创建自定义数据分割
federated_data = OrchestraDataProcessor.create_federated_split(
    data=train_data,
    labels=train_labels,
    num_parties=5,
    split_strategy='non_iid_pathological',
    alpha=0.5  # Dirichlet分布参数
)
```

### 3. 模型自定义

```python
from orchestra_model import OrchestraModel, OrchestraLoss

# 创建自定义模型
model = OrchestraModel(
    input_dim=3072,
    hidden_dims=[2048, 1024, 512, 256],
    embedding_dim=256,
    num_clusters=10,
    dropout_rate=0.2
)

# 创建自定义损失函数
loss_fn = OrchestraLoss(
    contrastive_weight=1.0,
    clustering_weight=1.5,
    consistency_weight=0.8,
    temperature=0.4
)
```

## 实验参数说明

### 数据集参数
- `--datasets`: 选择数据集 (cifar10, cifar100)
- `--num-parties`: 联邦学习参与方数量
- `--split-strategy`: 数据分割策略
  - `iid`: 独立同分布
  - `non_iid_dirichlet`: 非独立同分布（Dirichlet分布）
  - `non_iid_pathological`: 病理性非独立同分布

### 模型参数
- `--hidden-dims`: 隐藏层维度列表
- `--embedding-dim`: 嵌入维度
- `--dropout-rate`: Dropout率
- `--temperature`: 对比学习温度参数

### 训练参数
- `--learning-rate`: 学习率
- `--batch-size`: 批次大小
- `--num-epochs`: 训练轮数
- `--communication-rounds`: 联邦通信轮数
- `--local-epochs`: 本地训练轮数

### 损失权重
- `--contrastive-weight`: 对比学习损失权重
- `--clustering-weight`: 聚类损失权重
- `--consistency-weight`: 一致性损失权重

## 性能优化建议

### 1. GPU加速

```bash
# 确保CUDA可用
python -c "import torch; print('CUDA可用:', torch.cuda.is_available())"

# 使用GPU运行
python run_experiments.py --device cuda --datasets cifar10
```

### 2. 内存优化

```bash
# 减少批次大小
python run_experiments.py --batch-size 64 --datasets cifar10

# 减少隐藏层维度
python run_experiments.py --hidden-dims 512 256 128 --datasets cifar10
```

### 3. 快速测试

```bash
# 快速测试（少量轮数）
python run_experiments.py --num-epochs 10 --communication-rounds 5 --datasets cifar10
```

## 故障排除

### 1. 内存不足

```bash
# 减少批次大小和模型大小
python run_experiments.py --batch-size 32 --hidden-dims 256 128 --datasets cifar10
```

### 2. CUDA错误

```bash
# 强制使用CPU
python run_experiments.py --device cpu --datasets cifar10
```

### 3. 依赖问题

```bash
# 重新安装依赖
pip install --upgrade -r requirements.txt
```

### 4. 数据下载问题

如果CIFAR数据集下载失败，可以手动下载：

```python
import torchvision

# 手动下载CIFAR-10
torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
torchvision.datasets.CIFAR10(root='./data', train=False, download=True)

# 手动下载CIFAR-100
torchvision.datasets.CIFAR100(root='./data', train=True, download=True)
torchvision.datasets.CIFAR100(root='./data', train=False, download=True)
```

## 结果解释

### 评估指标

1. **ARI (Adjusted Rand Index)**: 聚类质量指标，范围[-1, 1]，越高越好
2. **NMI (Normalized Mutual Information)**: 聚类与真实标签的互信息，范围[0, 1]，越高越好
3. **Silhouette Score**: 聚类内聚性指标，范围[-1, 1]，越高越好

### 可视化结果

- **训练损失曲线**: 显示各种损失随训练进行的变化
- **聚类性能指标**: 显示评估指标随训练进行的变化
- **t-SNE嵌入可视化**: 显示学习到的嵌入的2D投影
- **聚类混淆矩阵**: 显示预测聚类与真实标签的对应关系

## 扩展和自定义

### 1. 添加新数据集

在 `cifar_experiments.py` 中添加新的数据集支持：

```python
def load_custom_data(self):
    # 实现自定义数据加载逻辑
    pass
```

### 2. 自定义模型架构

在 `orchestra_model.py` 中修改模型结构：

```python
class CustomOrchestraModel(OrchestraModel):
    def __init__(self, ...):
        # 自定义模型初始化
        pass
```

### 3. 添加新的评估指标

在训练器中添加自定义评估函数：

```python
def custom_evaluation_metric(predictions, labels):
    # 实现自定义评估逻辑
    return metric_value
```

## 联系和支持

如果遇到问题或需要帮助，请：

1. 检查实验日志文件
2. 查看错误信息和堆栈跟踪
3. 尝试减少实验规模进行测试
4. 确保所有依赖正确安装

## 许可证

本项目仅用于学术研究目的。