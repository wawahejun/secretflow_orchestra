#!/usr/bin/env python3
"""
Orchestra实现测试脚本
验证模型、损失函数和训练器的基本功能
"""

import os
import sys
import torch
import numpy as np
from typing import Dict, Any
import unittest
import tempfile
import shutil

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from orchestra_model import (
    ContrastiveEncoder, 
    ClusteringHead, 
    OrchestraModel, 
    OrchestraLoss, 
    OrchestraTrainer
)
from federated_orchestra import (
    OrchestraConfig,
    FederatedOrchestraModel,
    OrchestraDataProcessor
)

class TestOrchestraComponents(unittest.TestCase):
    """Orchestra组件测试类"""
    
    def setUp(self):
        """测试设置"""
        self.device = torch.device('cpu')  # 使用CPU进行测试
        self.batch_size = 32
        self.input_dim = 100
        self.embedding_dim = 64
        self.num_clusters = 10
        self.hidden_dims = [256, 128]
        
        # 创建测试数据
        self.test_data = torch.randn(self.batch_size, self.input_dim)
        self.test_labels = torch.randint(0, self.num_clusters, (self.batch_size,))
        
        # 创建临时目录
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """测试清理"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_contrastive_encoder(self):
        """测试对比学习编码器"""
        print("\n测试对比学习编码器...")
        
        encoder = ContrastiveEncoder(
            input_dim=self.input_dim,
            hidden_dims=self.hidden_dims,
            output_dim=self.embedding_dim,
            dropout_rate=0.2
        )
        
        # 前向传播测试
        embeddings = encoder(self.test_data)
        
        # 检查输出形状
        expected_shape = (self.batch_size, self.embedding_dim)
        self.assertEqual(embeddings.shape, expected_shape, 
                        f"编码器输出形状错误: 期望 {expected_shape}, 实际 {embeddings.shape}")
        
        # 检查输出是否为有限值
        self.assertTrue(torch.isfinite(embeddings).all(), "编码器输出包含非有限值")
        
        print(f"✓ 编码器输出形状: {embeddings.shape}")
        print(f"✓ 编码器输出范围: [{embeddings.min():.4f}, {embeddings.max():.4f}]")
    
    def test_clustering_head(self):
        """测试聚类头"""
        print("\n测试聚类头...")
        
        clustering_head = ClusteringHead(
            input_dim=self.embedding_dim,
            num_clusters=self.num_clusters,
            temperature=0.5
        )
        
        # 创建测试嵌入
        embeddings = torch.randn(self.batch_size, self.embedding_dim)
        
        # 前向传播测试
        cluster_probs = clustering_head(embeddings)
        
        # 检查输出形状
        expected_probs_shape = (self.batch_size, self.num_clusters)
        
        self.assertEqual(cluster_probs.shape, expected_probs_shape,
                        f"聚类概率形状错误: 期望 {expected_probs_shape}, 实际 {cluster_probs.shape}")
        
        # 检查概率和为1
        prob_sums = cluster_probs.sum(dim=1)
        self.assertTrue(torch.allclose(prob_sums, torch.ones_like(prob_sums), atol=1e-6),
                       "聚类概率和不为1")
        
        # 检查概率范围
        self.assertTrue((cluster_probs >= 0).all() and (cluster_probs <= 1).all(),
                       "聚类概率超出[0,1]范围")
        
        print(f"✓ 聚类概率形状: {cluster_probs.shape}")
        print(f"✓ 概率和范围: [{prob_sums.min():.6f}, {prob_sums.max():.6f}]")
    
    def test_orchestra_model(self):
        """测试完整Orchestra模型"""
        print("\n测试完整Orchestra模型...")
        
        model = OrchestraModel(
            input_dim=self.input_dim,
            hidden_dims=self.hidden_dims,
            embedding_dim=self.embedding_dim,
            num_clusters=self.num_clusters,
            dropout_rate=0.2,
            temperature=0.5
        )
        
        # 前向传播测试
        outputs = model(self.test_data)
        
        # 解包输出
        embeddings, cluster_probs, projections = outputs
        
        # 检查输出形状
        self.assertEqual(embeddings.shape, (self.batch_size, self.embedding_dim))
        self.assertEqual(cluster_probs.shape, (self.batch_size, self.num_clusters))
        self.assertIsNone(projections)  # 默认不返回投影
        
        print(f"✓ 嵌入形状: {embeddings.shape}")
        print(f"✓ 聚类概率形状: {cluster_probs.shape}")
        print(f"✓ 投影: {projections}")
    
    def test_orchestra_loss(self):
        """测试Orchestra损失函数"""
        print("\n测试Orchestra损失函数...")
        
        loss_fn = OrchestraLoss(
            contrastive_weight=1.0,
            clustering_weight=1.0,
            consistency_weight=0.5,
            temperature=0.5
        )
        
        # 创建测试数据
        data_batches = [self.test_data[:16], self.test_data[16:]]
        
        # 创建模型输出
        model = OrchestraModel(
            input_dim=self.input_dim,
            hidden_dims=self.hidden_dims,
            embedding_dim=self.embedding_dim,
            num_clusters=self.num_clusters
        )
        
        outputs_list = [model(batch) for batch in data_batches]
        
        # 提取projections和cluster_probs
        projections_list = []
        cluster_probs_list = []
        
        for outputs in outputs_list:
            embeddings, cluster_probs, projections = outputs
            # 如果没有projections，使用embeddings作为projections
            if projections is None:
                projections = embeddings
            projections_list.append(projections)
            cluster_probs_list.append(cluster_probs)
        
        # 计算损失
        losses = loss_fn(projections_list, cluster_probs_list)
        
        # 检查损失键
        expected_keys = {'total', 'contrastive', 'clustering', 'consistency'}
        self.assertEqual(set(losses.keys()), expected_keys,
                        f"损失键错误: 期望 {expected_keys}, 实际 {set(losses.keys())}")
        
        # 检查损失值
        for key, value in losses.items():
            self.assertIsInstance(value, torch.Tensor, f"{key}损失不是tensor")
            self.assertEqual(value.dim(), 0, f"{key}损失不是标量")
            self.assertTrue(torch.isfinite(value), f"{key}损失不是有限值")
            if key != 'total':  # total可能为负（由于consistency loss）
                self.assertTrue(value >= 0, f"{key}损失为负值: {value}")
        
        print(f"✓ 损失组件: {list(losses.keys())}")
        for key, value in losses.items():
            print(f"✓ {key}损失: {value.item():.6f}")
    
    def test_orchestra_trainer(self):
        """测试Orchestra训练器"""
        print("\n测试Orchestra训练器...")
        
        # 创建模型和损失函数
        model = OrchestraModel(
            input_dim=self.input_dim,
            hidden_dims=self.hidden_dims,
            embedding_dim=self.embedding_dim,
            num_clusters=self.num_clusters
        )
        
        loss_fn = OrchestraLoss(
            contrastive_weight=1.0,
            clustering_weight=1.0,
            consistency_weight=0.5
        )
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        trainer = OrchestraTrainer(model, loss_fn, optimizer, self.device)
        
        # 测试训练步骤
        data_batches = [self.test_data[:16], self.test_data[16:]]
        
        # 记录初始损失
        initial_losses = trainer.train_step(data_batches)
        
        # 再次训练
        second_losses = trainer.train_step(data_batches)
        
        # 检查损失结构
        expected_keys = {'total', 'contrastive', 'clustering', 'consistency'}
        self.assertEqual(set(initial_losses.keys()), expected_keys)
        self.assertEqual(set(second_losses.keys()), expected_keys)
        
        print(f"✓ 初始损失: {initial_losses['total']:.6f}")
        print(f"✓ 第二次损失: {second_losses['total']:.6f}")
        
        # 测试评估
        eval_results = trainer.evaluate(self.test_data, self.test_labels)
        
        # 检查评估结果
        expected_eval_keys = {
            'adjusted_rand_score', 'normalized_mutual_info', 
            'silhouette_score', 'num_clusters_used', 'cluster_entropy'
        }
        self.assertTrue(expected_eval_keys.issubset(set(eval_results.keys())),
                      f"评估结果缺少必要键: {expected_eval_keys - set(eval_results.keys())}")
        
        print(f"✓ 评估指标: {list(eval_results.keys())}")
        for key, value in eval_results.items():
            if isinstance(value, (int, float)):
                print(f"✓ {key}: {value:.4f}")
    
    def test_data_processor(self):
        """测试数据处理器"""
        print("\n测试数据处理器...")
        
        # 创建测试数据
        num_samples = 1000
        data = np.random.randn(num_samples, self.input_dim)
        labels = np.random.randint(0, self.num_clusters, num_samples)
        
        # 测试联邦数据分割
        num_parties = 3
        federated_data = OrchestraDataProcessor.create_federated_split(
            data=data,
            labels=labels,
            num_parties=num_parties,
            split_strategy='iid'
        )
        
        # 检查分割结果
        self.assertEqual(len(federated_data), num_parties,
                        f"参与方数量错误: 期望 {num_parties}, 实际 {len(federated_data)}")
        
        total_samples = 0
        for party, (party_data, party_labels) in federated_data.items():
            self.assertIsInstance(party_data, np.ndarray, f"{party}数据不是numpy数组")
            self.assertIsInstance(party_labels, np.ndarray, f"{party}标签不是numpy数组")
            self.assertEqual(party_data.shape[0], party_labels.shape[0],
                           f"{party}数据和标签数量不匹配")
            self.assertEqual(party_data.shape[1], self.input_dim,
                           f"{party}数据维度错误")
            total_samples += len(party_data)
        
        self.assertEqual(total_samples, num_samples,
                        f"总样本数不匹配: 期望 {num_samples}, 实际 {total_samples}")
        
        print(f"✓ 联邦数据分割成功")
        print(f"✓ 参与方数量: {len(federated_data)}")
        for party, (party_data, party_labels) in federated_data.items():
            print(f"✓ {party}: {len(party_data)} 样本")
    
    def test_orchestra_config(self):
        """测试Orchestra配置"""
        print("\n测试Orchestra配置...")
        
        config = OrchestraConfig(
            input_dim=self.input_dim,
            hidden_dims=self.hidden_dims,
            embedding_dim=self.embedding_dim,
            num_clusters=self.num_clusters
        )
        
        # 检查配置属性
        self.assertEqual(config.input_dim, self.input_dim)
        self.assertEqual(config.hidden_dims, self.hidden_dims)
        self.assertEqual(config.embedding_dim, self.embedding_dim)
        self.assertEqual(config.num_clusters, self.num_clusters)
        
        # 检查默认值
        self.assertIsInstance(config.dropout_rate, float)
        self.assertIsInstance(config.temperature, float)
        self.assertIsInstance(config.learning_rate, float)
        
        print(f"✓ 配置创建成功")
        print(f"✓ 输入维度: {config.input_dim}")
        print(f"✓ 隐藏层维度: {config.hidden_dims}")
        print(f"✓ 嵌入维度: {config.embedding_dim}")
        print(f"✓ 聚类数: {config.num_clusters}")

def run_basic_functionality_test():
    """运行基本功能测试"""
    print("="*80)
    print("Orchestra 基本功能测试")
    print("="*80)
    
    # 检查依赖
    try:
        import torch
        import numpy as np
        import sklearn
        print(f"✓ PyTorch版本: {torch.__version__}")
        print(f"✓ NumPy版本: {np.__version__}")
        print(f"✓ Scikit-learn版本: {sklearn.__version__}")
    except ImportError as e:
        print(f"❌ 依赖检查失败: {e}")
        return False
    
    # 检查CUDA
    if torch.cuda.is_available():
        print(f"✓ CUDA可用: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠ CUDA不可用，将使用CPU")
    
    # 运行单元测试
    print("\n开始单元测试...")
    
    # 创建测试套件
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestOrchestraComponents)
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # 打印结果
    print("\n" + "="*80)
    print("测试结果总结")
    print("="*80)
    print(f"运行测试: {result.testsRun}")
    print(f"失败: {len(result.failures)}")
    print(f"错误: {len(result.errors)}")
    
    if result.failures:
        print("\n失败的测试:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print("\n错误的测试:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
    
    success = len(result.failures) == 0 and len(result.errors) == 0
    
    if success:
        print("\n🎉 所有测试通过！Orchestra实现基本功能正常。")
    else:
        print("\n❌ 部分测试失败，请检查实现。")
    
    return success

def run_integration_test():
    """运行集成测试"""
    print("\n" + "="*80)
    print("Orchestra 集成测试")
    print("="*80)
    
    try:
        # 创建小规模测试数据
        print("创建测试数据...")
        num_samples = 200
        input_dim = 50
        num_clusters = 5
        
        data = np.random.randn(num_samples, input_dim)
        labels = np.random.randint(0, num_clusters, num_samples)
        
        # 创建配置
        config = OrchestraConfig(
            input_dim=input_dim,
            hidden_dims=[128, 64],
            embedding_dim=32,
            num_clusters=num_clusters,
            num_epochs=5,  # 少量轮数用于测试
            batch_size=32
        )
        
        # 创建模型
        print("创建模型...")
        model = OrchestraModel(
            input_dim=config.input_dim,
            hidden_dims=config.hidden_dims,
            embedding_dim=config.embedding_dim,
            num_clusters=config.num_clusters
        )
        
        # 创建损失函数和优化器
        loss_fn = OrchestraLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        
        # 创建训练器
        device = torch.device('cpu')
        trainer = OrchestraTrainer(model, loss_fn, optimizer, device)
        
        # 简单训练循环
        print("开始训练...")
        data_tensor = torch.tensor(data, dtype=torch.float32)
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        
        for epoch in range(config.num_epochs):
            # 分割数据模拟多个客户端
            mid = len(data_tensor) // 2
            data_batches = [data_tensor[:mid], data_tensor[mid:]]
            
            losses = trainer.train_step(data_batches)
            
            if epoch % 2 == 0:
                print(f"Epoch {epoch}: Loss = {losses['total']:.4f}")
        
        # 评估
        print("评估模型...")
        eval_results = trainer.evaluate(data_tensor, labels_tensor)
        
        print("评估结果:")
        for key, value in eval_results.items():
            if isinstance(value, (int, float)):
                print(f"  {key}: {value:.4f}")
        
        print("\n🎉 集成测试成功！")
        return True
        
    except Exception as e:
        print(f"\n❌ 集成测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Orchestra 实现测试")
    print("这个脚本将测试Orchestra实现的各个组件")
    
    # 运行基本功能测试
    basic_success = run_basic_functionality_test()
    
    # 运行集成测试
    integration_success = run_integration_test()
    
    # 总结
    print("\n" + "="*80)
    print("最终测试结果")
    print("="*80)
    print(f"基本功能测试: {'✓ 通过' if basic_success else '❌ 失败'}")
    print(f"集成测试: {'✓ 通过' if integration_success else '❌ 失败'}")
    
    if basic_success and integration_success:
        print("\n🎉 所有测试通过！Orchestra实现可以正常使用。")
        print("\n下一步可以运行:")
        print("  python run_experiments.py --datasets cifar10 --num-epochs 10")
    else:
        print("\n❌ 部分测试失败，请检查实现或依赖。")
        sys.exit(1)