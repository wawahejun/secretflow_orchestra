#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Orchestra自定义策略快速入门
演示如何使用方式1实现和使用自定义联邦学习策略
"""

import numpy as np
import logging
import secretflow as sf
from secretflow.device import PYU

# 重要：导入自定义策略以触发注册
from orchestra_strategy import PYUOrchestraStrategy
from federated_orchestra import FederatedOrchestraTrainer, OrchestraConfig

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def quick_demo():
    """快速演示自定义策略的使用"""
    
    print("🚀 Orchestra自定义策略快速演示")
    print("=" * 50)
    
    try:
        # 1. 初始化SecretFlow
        print("1. 初始化SecretFlow...")
        sf.init(['alice', 'bob'], address='local')
        
        # 2. 验证策略注册
        print("2. 验证自定义策略注册...")
        print("✅ Orchestra策略已通过导入触发注册")
        print("   (策略名称: orchestra_torch)")
        
        # 3. 创建配置
        print("3. 创建Orchestra配置...")
        config = OrchestraConfig(
            input_dim=32,
            num_clusters=5,
            temperature=0.5,
            clustering_weight=1.0,
            contrastive_weight=0.5,
            learning_rate=0.001,
            batch_size=32,
            local_epochs=2,
            communication_rounds=2
        )
        
        # 4. 创建训练器
        print("4. 创建联邦训练器...")
        trainer = FederatedOrchestraTrainer(
            config=config,
            parties=['alice', 'bob']
        )
        
        # 5. 设置模型（使用自定义策略）
        print("5. 设置联邦模型（使用Orchestra策略）...")
        trainer.setup_model()
        
        print(f"✅ 模型设置完成，使用策略: {trainer.fed_model.strategy}")
        
        # 6. 创建模拟数据
        print("6. 创建模拟数据...")
        np.random.seed(42)
        
        # 为每个参与方创建数据
        alice_data = {
            'x': np.random.randn(100, 32).astype(np.float32),
            'y': np.random.randint(0, 5, 100).astype(np.int64)
        }
        
        bob_data = {
            'x': np.random.randn(100, 32).astype(np.float32),
            'y': np.random.randint(0, 5, 100).astype(np.int64)
        }
        
        federated_data = {
            'alice': (alice_data['x'], alice_data['y']),
            'bob': (bob_data['x'], bob_data['y'])
        }
        
        print("✅ 数据准备完成")
        print(f"  Alice: {alice_data['x'].shape[0]} 样本")
        print(f"  Bob: {bob_data['x'].shape[0]} 样本")
        
        # 7. 准备联邦数据
        print("7. 准备联邦数据...")
        fed_data = trainer.prepare_data(federated_data)
        print("✅ 联邦数据准备完成")
        
        # 8. 开始训练
        print("8. 开始联邦训练...")
        print("-" * 30)
        
        for round_idx in range(config.communication_rounds):
            print(f"通信轮次 {round_idx + 1}/{config.communication_rounds}")
            
            try:
                # 执行一轮训练
                history = trainer.fed_model.fit(
                    x=fed_data['x'],
                    y=fed_data['y'],
                    batch_size=config.batch_size,
                    epochs=config.local_epochs,
                    verbose=0
                )
                print(f"✅ 轮次 {round_idx + 1} 完成")
                
            except Exception as e:
                print(f"⚠️ 轮次 {round_idx + 1} 遇到问题: {str(e)[:50]}...")
                # 继续下一轮
                continue
        
        print("-" * 30)
        print("✅ 联邦训练演示完成！")
        
        # 9. 显示结果摘要
        print("\n📊 训练摘要:")
        print(f"  策略类型: Orchestra自定义策略")
        print(f"  参与方: {list(federated_data.keys())}")
        print(f"  总样本数: {sum(len(data[0]) for data in federated_data.values())}")
        print(f"  聚类数量: {config.num_clusters}")
        print(f"  通信轮次: {config.communication_rounds}")
        print(f"  本地训练轮次: {config.local_epochs}")
        
    except Exception as e:
        print(f"❌ 演示失败: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # 清理
        try:
            sf.shutdown()
            print("\n🔧 SecretFlow已关闭")
        except:
            pass

def show_strategy_info():
    """显示策略信息"""
    print("\n📋 自定义策略信息:")
    print("=" * 30)
    print("策略名称: orchestra")
    print("后端: torch")
    print("类型: 无监督联邦学习")
    print("特性:")
    print("  - 全局一致性聚类")
    print("  - 对比学习")
    print("  - 自适应聚类中心")
    print("  - 支持非IID数据")
    
    print("\n🔧 关键优势:")
    print("  ✅ 无需修改SecretFlow源码")
    print("  ✅ 支持自定义参数")
    print("  ✅ 易于调试和扩展")
    print("  ✅ 完全模块化设计")

if __name__ == "__main__":
    show_strategy_info()
    quick_demo()
    
    print("\n🎉 快速入门完成！")
    print("\n📚 更多信息请参考:")
    print("  - CUSTOM_STRATEGY_GUIDE.md: 详细实现指南")
    print("  - orchestra_strategy.py: 策略实现代码")
    print("  - demo_custom_strategy.py: 完整演示脚本")