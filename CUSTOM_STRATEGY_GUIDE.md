# Orchestra自定义策略实现指南

本指南详细介绍如何使用**方式1（项目内直接注册）**来实现和使用自定义的Orchestra联邦学习策略。

## 📋 目录

1. [概述](#概述)
2. [方式1详细步骤](#方式1详细步骤)
3. [文件结构](#文件结构)
4. [核心代码解析](#核心代码解析)
5. [运行演示](#运行演示)
6. [自定义扩展](#自定义扩展)
7. [常见问题](#常见问题)

## 🎯 概述

**方式1**是在项目内直接注册自定义策略的方法，具有以下优势：

✅ **无需修改SecretFlow源码**  
✅ **开发和调试简单**  
✅ **易于版本控制**  
✅ **支持快速迭代**  
✅ **完全自主控制**  

## 🚀 方式1详细步骤

### 步骤1：创建自定义策略文件

创建 `orchestra_strategy.py` 文件，实现自定义策略：

```python
# orchestra_strategy.py
from secretflow_fl.ml.nn.fl.backend.torch.fl_base import BaseTorchModel
from secretflow_fl.ml.nn.fl.strategy_dispatcher import register_strategy

class OrchestraStrategy(BaseTorchModel):
    """Orchestra联邦学习策略"""
    
    def train_step(self, weights, cur_steps, train_steps, **kwargs):
        """实现Orchestra特定的训练逻辑"""
        # 1. 应用全局权重
        if weights is not None:
            self.set_weights(weights)
        
        # 2. Orchestra训练逻辑
        for step in range(train_steps):
            # 获取数据
            x, y, s_w = self.next_batch()
            
            # Orchestra前向传播
            features, projections = self.model.forward_orchestra(x)
            
            # 计算Orchestra损失
            cluster_loss = self._compute_cluster_loss(features)
            contrastive_loss = self._compute_contrastive_loss(projections)
            loss = cluster_loss + contrastive_loss
            
            # 反向传播
            self.model.backward_step(loss)
        
        # 3. 返回更新后的权重
        return self.get_weights(return_numpy=True), num_sample

# 关键：使用装饰器注册策略
@register_strategy(strategy_name="orchestra", backend="torch")
class PYUOrchestraStrategy(OrchestraStrategy):
    pass
```

### 步骤2：在主模块中导入策略

在 `federated_orchestra.py` 中导入策略以触发注册：

```python
# federated_orchestra.py
from orchestra_strategy import PYUOrchestraStrategy  # 触发策略注册

class FederatedOrchestraTrainer:
    def setup_model(self):
        self.fed_model = FLModel(
            device_list=list(self.devices.values()),
            model=create_model,
            strategy="orchestra",  # 使用自定义策略
            backend='torch',
            # 传递策略特定参数
            strategy_config={
                'temperature': self.config.temperature,
                'num_clusters': self.config.num_clusters
            }
        )
```

### 步骤3：运行实验

```python
# 导入策略以触发注册
from orchestra_strategy import PYUOrchestraStrategy

# 创建训练器并运行
trainer = FederatedOrchestraTrainer(config, devices)
results = trainer.train(federated_data)
```

## 📁 文件结构

```
secretflow_orchestra/
├── orchestra_strategy.py          # 🆕 自定义策略实现
├── federated_orchestra.py         # 修改：导入自定义策略
├── demo_custom_strategy.py        # 🆕 演示脚本
├── CUSTOM_STRATEGY_GUIDE.md       # 🆕 本指南
├── orchestra_model.py             # Orchestra模型定义
├── cifar_experiments.py           # CIFAR实验脚本
└── ...
```

## 🔍 核心代码解析

### 1. 策略注册机制

```python
@register_strategy(strategy_name="orchestra", backend="torch")
class PYUOrchestraStrategy(OrchestraStrategy):
    pass
```

**工作原理：**
- `@register_strategy` 装饰器将策略注册到全局调度器
- 策略名称格式：`{strategy_name}_{backend}`
- 导入模块时自动触发注册

### 2. 策略调度

```python
# SecretFlow内部调度逻辑
from secretflow_fl.ml.nn.fl.strategy_dispatcher import dispatch_strategy

# 当FLModel指定strategy="orchestra"时
strategy_instance = dispatch_strategy("orchestra", "torch", *args, **kwargs)
```

### 3. Orchestra特定逻辑

```python
def train_step(self, weights, cur_steps, train_steps, **kwargs):
    # 1. 聚类损失：最小化特征到聚类中心的距离
    cluster_loss = self._compute_cluster_loss(features)
    
    # 2. 对比损失：学习判别性特征表示
    contrastive_loss = self._compute_contrastive_loss(projections)
    
    # 3. 总损失
    total_loss = cluster_weight * cluster_loss + contrastive_weight * contrastive_loss
```

## 🎮 运行演示

### 快速开始

```bash
# 1. 确保环境正确
cd /home/wawahejun/sf/secretflow_orchestra

# 2. 运行演示
python demo_custom_strategy.py
```

### 演示内容

1. **策略注册验证**：检查策略是否成功注册
2. **联邦训练演示**：使用自定义策略进行训练
3. **参数传递演示**：展示如何传递Orchestra特定参数
4. **结果分析**：显示训练结果和性能指标

### 预期输出

```
=== 策略注册演示 ===
已注册的策略:
  - fed_avg_w_torch
  - orchestra_torch          # ✅ 我们的策略
  - orchestra_simple_torch   # ✅ 简化版策略

=== Orchestra自定义策略演示 ===
1. 设置SecretFlow集群...
2. 加载演示数据...
3. 创建Orchestra配置...
...
✅ 演示成功完成！
```

## 🛠 自定义扩展

### 1. 添加新的损失函数

```python
class AdvancedOrchestraStrategy(OrchestraStrategy):
    def _compute_regularization_loss(self, weights):
        """添加正则化损失"""
        l2_loss = sum(torch.norm(w) ** 2 for w in weights)
        return 0.001 * l2_loss
    
    def train_step(self, weights, cur_steps, train_steps, **kwargs):
        # 调用父类方法
        model_weights, num_sample = super().train_step(
            weights, cur_steps, train_steps, **kwargs
        )
        
        # 添加额外的正则化
        reg_loss = self._compute_regularization_loss(model_weights)
        
        return model_weights, num_sample

@register_strategy(strategy_name="orchestra_advanced", backend="torch")
class PYUAdvancedOrchestraStrategy(AdvancedOrchestraStrategy):
    pass
```

### 2. 自定义聚合策略

```python
class OrchestraWithCustomAggregation(OrchestraStrategy):
    def aggregate_weights(self, weights_list, sample_counts):
        """自定义权重聚合"""
        # 实现基于聚类质量的加权聚合
        quality_weights = self._compute_cluster_quality(weights_list)
        
        aggregated = None
        total_quality = sum(quality_weights)
        
        for weights, quality in zip(weights_list, quality_weights):
            weight_ratio = quality / total_quality
            if aggregated is None:
                aggregated = [w * weight_ratio for w in weights]
            else:
                for i, w in enumerate(weights):
                    aggregated[i] += w * weight_ratio
        
        return aggregated
```

### 3. 添加差分隐私

```python
class PrivateOrchestraStrategy(OrchestraStrategy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.noise_scale = kwargs.get('noise_scale', 0.1)
    
    def train_step(self, weights, cur_steps, train_steps, **kwargs):
        model_weights, num_sample = super().train_step(
            weights, cur_steps, train_steps, **kwargs
        )
        
        # 添加差分隐私噪声
        if self.noise_scale > 0:
            for i, w in enumerate(model_weights):
                noise = np.random.normal(0, self.noise_scale, w.shape)
                model_weights[i] = w + noise
        
        return model_weights, num_sample
```

## ❓ 常见问题

### Q1: 策略没有被识别？

**A:** 确保导入了策略模块：
```python
# 必须导入以触发注册
from orchestra_strategy import PYUOrchestraStrategy
```

### Q2: 如何传递自定义参数？

**A:** 通过 `strategy_config` 或 `kwargs` 传递：
```python
FLModel(
    strategy="orchestra",
    strategy_config={'temperature': 0.5}
)

# 或在训练时传递
fed_model.fit(
    x=data, y=labels,
    strategy_kwargs={'custom_param': value}
)
```

### Q3: 如何调试策略？

**A:** 添加日志和断点：
```python
class OrchestraStrategy(BaseTorchModel):
    def train_step(self, weights, cur_steps, train_steps, **kwargs):
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"开始训练步骤，当前步数: {cur_steps}")
        
        # 添加断点
        import pdb; pdb.set_trace()
        
        # 你的训练逻辑
        ...
```

### Q4: 如何验证策略正确性？

**A:** 创建单元测试：
```python
def test_orchestra_strategy():
    strategy = OrchestraStrategy(model_builder, random_seed=42)
    
    # 测试训练步骤
    weights, num_samples = strategy.train_step(
        weights=None, cur_steps=0, train_steps=1
    )
    
    assert weights is not None
    assert num_samples > 0
```

### Q5: 性能优化建议？

**A:** 
1. **批处理优化**：增大batch size
2. **内存管理**：及时释放不需要的张量
3. **计算优化**：使用GPU加速
4. **通信优化**：减少不必要的数据传输

```python
class OptimizedOrchestraStrategy(OrchestraStrategy):
    def train_step(self, weights, cur_steps, train_steps, **kwargs):
        # 使用torch.no_grad()减少内存使用
        with torch.no_grad():
            # 非训练相关的计算
            pass
        
        # 清理GPU缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return super().train_step(weights, cur_steps, train_steps, **kwargs)
```

## 🎉 总结

使用**方式1**实现自定义Orchestra策略的关键步骤：

1. ✅ 创建策略类继承 `BaseTorchModel`
2. ✅ 实现 `train_step` 和 `apply_weights` 方法
3. ✅ 使用 `@register_strategy` 装饰器注册
4. ✅ 在主模块中导入策略触发注册
5. ✅ 在 `FLModel` 中指定策略名称

这种方式让您能够：
- 🚀 快速开发和测试新策略
- 🔧 完全控制策略行为
- 📦 保持代码模块化
- 🔄 支持持续迭代改进

现在您可以开始实现自己的Orchestra联邦学习策略了！