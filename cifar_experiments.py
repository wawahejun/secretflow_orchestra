#!/usr/bin/env python3
"""
CIFAR数据集上的Orchestra实验
复现论文在CIFAR-10和CIFAR-100上的实验结果
"""

import os
import sys
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import logging
import json
from datetime import datetime
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from orchestra_model import OrchestraModel, OrchestraLoss, OrchestraTrainer
from federated_orchestra import (
    OrchestraConfig, 
    FederatedOrchestraTrainer, 
    OrchestraDataProcessor
)

class CIFAROrchestralExperiment:
    """CIFAR数据集Orchestra实验类"""
    
    def __init__(self, 
                 dataset_name: str = 'cifar10',
                 num_parties: int = 3,
                 split_strategy: str = 'iid',
                 output_dir: str = './orchestra_results'):
        
        self.dataset_name = dataset_name.lower()
        self.num_parties = num_parties
        self.split_strategy = split_strategy
        self.output_dir = output_dir
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 设置日志
        self.setup_logging()
        
        # 数据集参数
        if self.dataset_name == 'cifar10':
            self.num_classes = 10
            self.class_names = [
                'airplane', 'automobile', 'bird', 'cat', 'deer',
                'dog', 'frog', 'horse', 'ship', 'truck'
            ]
        elif self.dataset_name == 'cifar100':
            self.num_classes = 100
            self.class_names = None  # 太多类别，不列出
        else:
            raise ValueError(f"不支持的数据集: {dataset_name}")
        
        self.input_dim = 32 * 32 * 3  # CIFAR图像维度
        
    def setup_logging(self):
        """设置日志"""
        log_file = os.path.join(self.output_dir, f'{self.dataset_name}_experiment.log')
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        self.logger = logging.getLogger(__name__)
    
    def load_cifar_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """加载CIFAR数据集"""
        self.logger.info(f"加载{self.dataset_name.upper()}数据集...")
        
        # 数据预处理
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        # 加载数据集
        if self.dataset_name == 'cifar10':
            train_dataset = torchvision.datasets.CIFAR10(
                root='./data', train=True, download=True, transform=transform
            )
            test_dataset = torchvision.datasets.CIFAR10(
                root='./data', train=False, download=True, transform=transform
            )
        else:  # cifar100
            train_dataset = torchvision.datasets.CIFAR100(
                root='./data', train=True, download=True, transform=transform
            )
            test_dataset = torchvision.datasets.CIFAR100(
                root='./data', train=False, download=True, transform=transform
            )
        
        # 转换为numpy数组
        def dataset_to_numpy(dataset):
            loader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset), shuffle=False)
            data, labels = next(iter(loader))
            return data.numpy(), labels.numpy()
        
        train_data, train_labels = dataset_to_numpy(train_dataset)
        test_data, test_labels = dataset_to_numpy(test_dataset)
        
        # 展平图像数据
        train_data = train_data.reshape(train_data.shape[0], -1)
        test_data = test_data.reshape(test_data.shape[0], -1)
        
        self.logger.info(f"数据加载完成:")
        self.logger.info(f"  训练集: {train_data.shape}, 标签: {train_labels.shape}")
        self.logger.info(f"  测试集: {test_data.shape}, 标签: {test_labels.shape}")
        self.logger.info(f"  类别数: {self.num_classes}")
        
        return train_data, train_labels, test_data, test_labels
    
    def create_federated_data(self, 
                            train_data: np.ndarray, 
                            train_labels: np.ndarray) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """创建联邦数据分割"""
        self.logger.info(f"创建联邦数据分割 ({self.split_strategy})...")
        
        federated_data = OrchestraDataProcessor.create_federated_split(
            data=train_data,
            labels=train_labels,
            num_parties=self.num_parties,
            split_strategy=self.split_strategy
        )
        
        # 打印数据分布信息
        for party, (data, labels) in federated_data.items():
            unique_labels, counts = np.unique(labels, return_counts=True)
            self.logger.info(f"  {party}: {len(data)} 样本, {len(unique_labels)} 类别")
            if len(unique_labels) <= 20:  # 只显示少量类别的分布
                label_dist = dict(zip(unique_labels, counts))
                self.logger.info(f"    标签分布: {label_dist}")
        
        return federated_data
    
    def run_centralized_experiment(self, 
                                 train_data: np.ndarray,
                                 train_labels: np.ndarray,
                                 test_data: np.ndarray,
                                 test_labels: np.ndarray,
                                 config: OrchestraConfig) -> Dict:
        """运行中心化Orchestra实验（作为基线）"""
        self.logger.info("开始中心化Orchestra实验...")
        
        # 创建模型
        model = OrchestraModel(
            input_dim=config.input_dim,
            hidden_dims=config.hidden_dims,
            embedding_dim=config.embedding_dim,
            num_clusters=config.num_clusters,
            dropout_rate=config.dropout_rate,
            temperature=config.temperature
        )
        
        # 创建损失函数
        loss_fn = OrchestraLoss(
            contrastive_weight=config.contrastive_weight,
            clustering_weight=config.clustering_weight,
            consistency_weight=config.consistency_weight,
            temperature=config.temperature
        )
        
        # 创建优化器
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        
        # 创建训练器
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        trainer = OrchestraTrainer(model, loss_fn, optimizer, device)
        
        # 训练
        train_dataset = torch.utils.data.TensorDataset(
            torch.tensor(train_data, dtype=torch.float32),
            torch.tensor(train_labels, dtype=torch.long)
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=config.batch_size, shuffle=True
        )
        
        training_history = {
            'epoch': [],
            'total_loss': [],
            'contrastive_loss': [],
            'clustering_loss': [],
            'consistency_loss': [],
            'ari_score': [],
            'nmi_score': [],
            'silhouette_score': []
        }
        
        for epoch in range(config.num_epochs):
            epoch_losses = []
            
            for batch_idx, (batch_data, _) in enumerate(train_loader):
                # 为了模拟多个客户端，我们将批次分割
                batch_size = batch_data.size(0)
                split_size = batch_size // 2
                
                if split_size > 0:
                    data_batches = [
                        batch_data[:split_size],
                        batch_data[split_size:]
                    ]
                else:
                    data_batches = [batch_data]
                
                losses = trainer.train_step(data_batches)
                epoch_losses.append(losses)
            
            # 计算平均损失
            avg_losses = {
                key: np.mean([loss[key] for loss in epoch_losses])
                for key in epoch_losses[0].keys()
            }
            
            # 记录训练历史
            training_history['epoch'].append(epoch)
            for key, value in avg_losses.items():
                if key in training_history:
                    training_history[key].append(value)
            
            # 每10轮评估一次
            if (epoch + 1) % 10 == 0:
                test_tensor = torch.tensor(test_data, dtype=torch.float32)
                test_labels_tensor = torch.tensor(test_labels, dtype=torch.long)
                
                eval_results = trainer.evaluate(test_tensor, test_labels_tensor)
                
                training_history['ari_score'].append(eval_results.get('adjusted_rand_score', 0))
                training_history['nmi_score'].append(eval_results.get('normalized_mutual_info', 0))
                training_history['silhouette_score'].append(eval_results.get('silhouette_score', 0))
                
                self.logger.info(
                    f"Epoch {epoch+1}/{config.num_epochs} - "
                    f"Loss: {avg_losses['total']:.4f}, "
                    f"ARI: {eval_results.get('adjusted_rand_score', 0):.4f}, "
                    f"NMI: {eval_results.get('normalized_mutual_info', 0):.4f}"
                )
        
        # 最终评估
        test_tensor = torch.tensor(test_data, dtype=torch.float32)
        test_labels_tensor = torch.tensor(test_labels, dtype=torch.long)
        final_results = trainer.evaluate(test_tensor, test_labels_tensor)
        
        # 获取嵌入和聚类分配
        embeddings = trainer.get_embeddings(test_tensor)
        cluster_assignments = trainer.get_cluster_assignments(test_tensor)
        
        return {
            'training_history': training_history,
            'final_results': final_results,
            'embeddings': embeddings.numpy(),
            'cluster_assignments': cluster_assignments.numpy(),
            'model_state': model.state_dict()
        }
    
    def run_federated_experiment(self,
                               federated_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
                               test_data: np.ndarray,
                               test_labels: np.ndarray,
                               config: OrchestraConfig) -> Dict:
        """运行联邦Orchestra实验"""
        self.logger.info("开始联邦Orchestra实验...")
        
        # 创建参与方列表
        parties = list(federated_data.keys())
        
        # 创建联邦训练器
        fed_trainer = FederatedOrchestraTrainer(config=config, parties=parties)
        
        # 准备联邦数据
        try:
            fed_data_dict = fed_trainer.prepare_data(federated_data)
            
            # 训练
            training_history = fed_trainer.train(fed_data_dict)
            
            # 评估
            test_fed_data = {
                'test': (test_data, test_labels)
            }
            test_fed_dict = fed_trainer.prepare_data(test_fed_data)
            final_results = fed_trainer.evaluate(test_fed_dict)
            
            # 确保训练历史有正确的结构
            history_dict = training_history.history if hasattr(training_history, 'history') else {}
            if not history_dict:
                history_dict = {
                    'epoch': [],
                    'total_loss': [],
                    'contrastive_loss': [],
                    'clustering_loss': [],
                    'consistency_loss': [],
                    'ari_score': [],
                    'nmi_score': [],
                    'silhouette_score': []
                }
            
            return {
                'training_history': history_dict,
                'final_results': final_results,
                'model_weights': fed_trainer.get_model_weights()
            }
        
        except Exception as e:
            self.logger.error(f"联邦实验失败: {e}")
            # 如果联邦实验失败，返回空但结构正确的结果
            return {
                'training_history': {
                    'epoch': [],
                    'total_loss': [],
                    'contrastive_loss': [],
                    'clustering_loss': [],
                    'consistency_loss': [],
                    'ari_score': [],
                    'nmi_score': [],
                    'silhouette_score': []
                },
                'final_results': {},
                'error': str(e)
            }
    
    def visualize_results(self, 
                        centralized_results: Dict,
                        federated_results: Dict,
                        test_labels: np.ndarray):
        """可视化实验结果"""
        self.logger.info("生成可视化结果...")
        
        # 创建图表
        fig = plt.figure(figsize=(20, 15))
        
        # 1. 训练损失曲线
        if 'training_history' in centralized_results:
            history = centralized_results['training_history']
            
            plt.subplot(3, 4, 1)
            if ('epoch' in history and len(history['epoch']) > 0 and
                'total_loss' in history and len(history['total_loss']) > 0):
                plt.plot(history['epoch'], history['total_loss'], label='Total Loss')
                if 'contrastive_loss' in history and len(history['contrastive_loss']) == len(history['epoch']):
                    plt.plot(history['epoch'], history['contrastive_loss'], label='Contrastive')
                if 'clustering_loss' in history and len(history['clustering_loss']) == len(history['epoch']):
                    plt.plot(history['epoch'], history['clustering_loss'], label='Clustering')
                if 'consistency_loss' in history and len(history['consistency_loss']) == len(history['epoch']):
                    plt.plot(history['epoch'], history['consistency_loss'], label='Consistency')
                plt.title('Training Losses (Centralized)')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.legend()
                plt.grid(True)
            else:
                plt.text(0.5, 0.5, 'No training history available', 
                        ha='center', va='center', transform=plt.gca().transAxes)
                plt.title('Training Losses (Centralized)')
            
            # 2. 聚类性能指标
            plt.subplot(3, 4, 2)
            if ('ari_score' in history and len(history['ari_score']) > 0 and
                'nmi_score' in history and len(history['nmi_score']) > 0 and
                'silhouette_score' in history and len(history['silhouette_score']) > 0):
                eval_epochs = list(range(10, len(history['ari_score']) * 10 + 1, 10))
                plt.plot(eval_epochs, history['ari_score'], 'o-', label='ARI')
                plt.plot(eval_epochs, history['nmi_score'], 's-', label='NMI')
                plt.plot(eval_epochs, history['silhouette_score'], '^-', label='Silhouette')
                plt.title('Clustering Metrics (Centralized)')
                plt.xlabel('Epoch')
                plt.ylabel('Score')
                plt.legend()
                plt.grid(True)
            else:
                plt.text(0.5, 0.5, 'No clustering metrics available', 
                        ha='center', va='center', transform=plt.gca().transAxes)
                plt.title('Clustering Metrics (Centralized)')
        
        # 3. 嵌入可视化 (t-SNE)
        if 'embeddings' in centralized_results:
            embeddings = centralized_results['embeddings']
            
            # 使用t-SNE降维
            if embeddings.shape[0] > 1000:  # 如果样本太多，随机采样
                indices = np.random.choice(embeddings.shape[0], 1000, replace=False)
                embeddings_sample = embeddings[indices]
                labels_sample = test_labels[indices]
            else:
                embeddings_sample = embeddings
                labels_sample = test_labels
            
            tsne = TSNE(n_components=2, random_state=42, perplexity=30)
            embeddings_2d = tsne.fit_transform(embeddings_sample)
            
            plt.subplot(3, 4, 3)
            scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                                c=labels_sample, cmap='tab10', alpha=0.7)
            plt.title('t-SNE Embeddings (True Labels)')
            plt.colorbar(scatter)
            
            # 4. 聚类结果可视化
            if 'cluster_assignments' in centralized_results:
                cluster_assignments = centralized_results['cluster_assignments']
                if len(cluster_assignments) > 1000:
                    cluster_sample = cluster_assignments[indices]
                else:
                    cluster_sample = cluster_assignments
                
                plt.subplot(3, 4, 4)
                scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                                    c=cluster_sample, cmap='tab10', alpha=0.7)
                plt.title('t-SNE Embeddings (Predicted Clusters)')
                plt.colorbar(scatter)
        
        # 5. 聚类混淆矩阵
        if 'cluster_assignments' in centralized_results:
            from sklearn.metrics import confusion_matrix
            
            cluster_assignments = centralized_results['cluster_assignments']
            
            # 计算混淆矩阵（需要匹配聚类标签和真实标签）
            cm = confusion_matrix(test_labels, cluster_assignments)
            
            plt.subplot(3, 4, 5)
            sns.heatmap(cm, annot=False, cmap='Blues', fmt='d')
            plt.title('Clustering Confusion Matrix')
            plt.xlabel('Predicted Cluster')
            plt.ylabel('True Label')
        
        # 6. 类别分布
        plt.subplot(3, 4, 6)
        unique_labels, counts = np.unique(test_labels, return_counts=True)
        plt.bar(unique_labels, counts)
        plt.title('True Label Distribution')
        plt.xlabel('Label')
        plt.ylabel('Count')
        
        if 'cluster_assignments' in centralized_results:
            cluster_assignments = centralized_results['cluster_assignments']
            unique_clusters, cluster_counts = np.unique(cluster_assignments, return_counts=True)
            
            plt.subplot(3, 4, 7)
            plt.bar(unique_clusters, cluster_counts)
            plt.title('Predicted Cluster Distribution')
            plt.xlabel('Cluster')
            plt.ylabel('Count')
        
        # 7. 性能对比（如果有联邦结果）
        if 'final_results' in centralized_results and 'final_results' in federated_results:
            cent_results = centralized_results['final_results']
            fed_results = federated_results['final_results']
            
            metrics = ['adjusted_rand_score', 'normalized_mutual_info', 'silhouette_score']
            cent_values = [cent_results.get(m, 0) for m in metrics]
            fed_values = [fed_results.get(m, 0) for m in metrics]
            
            plt.subplot(3, 4, 8)
            x = np.arange(len(metrics))
            width = 0.35
            
            plt.bar(x - width/2, cent_values, width, label='Centralized', alpha=0.8)
            plt.bar(x + width/2, fed_values, width, label='Federated', alpha=0.8)
            
            plt.title('Performance Comparison')
            plt.xlabel('Metrics')
            plt.ylabel('Score')
            plt.xticks(x, ['ARI', 'NMI', 'Silhouette'])
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # 8. 损失分解
        if 'training_history' in centralized_results:
            history = centralized_results['training_history']
            
            plt.subplot(3, 4, 9)
            if ('epoch' in history and len(history['epoch']) > 0):
                if 'contrastive_loss' in history and len(history['contrastive_loss']) == len(history['epoch']):
                    plt.plot(history['epoch'], history['contrastive_loss'], label='Contrastive')
                if 'clustering_loss' in history and len(history['clustering_loss']) == len(history['epoch']):
                    plt.plot(history['epoch'], history['clustering_loss'], label='Clustering')
                if 'consistency_loss' in history and len(history['consistency_loss']) == len(history['epoch']):
                    plt.plot(history['epoch'], history['consistency_loss'], label='Consistency')
                plt.title('Loss Components')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.legend()
                plt.grid(True)
            else:
                plt.text(0.5, 0.5, 'No loss components available', 
                        ha='center', va='center', transform=plt.gca().transAxes)
                plt.title('Loss Components')
        
        # 9-12. 预留空间用于其他可视化
        
        plt.tight_layout()
        
        # 保存图表
        plot_path = os.path.join(self.output_dir, f'{self.dataset_name}_orchestra_results.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        self.logger.info(f"结果图表已保存: {plot_path}")
        
        plt.show()
    
    def save_results(self, 
                   centralized_results: Dict,
                   federated_results: Dict,
                   config: OrchestraConfig):
        """保存实验结果"""
        results = {
            'experiment_info': {
                'dataset': self.dataset_name,
                'num_parties': self.num_parties,
                'split_strategy': self.split_strategy,
                'timestamp': datetime.now().isoformat(),
                'config': config.__dict__
            },
            'centralized_results': centralized_results,
            'federated_results': federated_results
        }
        
        # 移除不能序列化的对象
        if 'model_state' in results['centralized_results']:
            del results['centralized_results']['model_state']
        
        if 'embeddings' in results['centralized_results']:
            # 保存嵌入到单独文件
            embeddings_path = os.path.join(self.output_dir, f'{self.dataset_name}_embeddings.npy')
            np.save(embeddings_path, results['centralized_results']['embeddings'])
            del results['centralized_results']['embeddings']
        
        if 'cluster_assignments' in results['centralized_results']:
            # 保存聚类分配到单独文件
            clusters_path = os.path.join(self.output_dir, f'{self.dataset_name}_clusters.npy')
            np.save(clusters_path, results['centralized_results']['cluster_assignments'])
            del results['centralized_results']['cluster_assignments']
        
        # 保存JSON结果
        results_path = os.path.join(self.output_dir, f'{self.dataset_name}_results.json')
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        self.logger.info(f"实验结果已保存: {results_path}")
    
    def run_complete_experiment(self, config: OrchestraConfig) -> Tuple[Dict, Dict]:
        """运行完整实验"""
        self.logger.info(f"开始{self.dataset_name.upper()} Orchestra完整实验")
        self.logger.info(f"配置: {config.__dict__}")
        
        # 1. 加载数据
        train_data, train_labels, test_data, test_labels = self.load_cifar_data()
        
        # 2. 创建联邦数据分割
        federated_data = self.create_federated_data(train_data, train_labels)
        
        # 3. 运行中心化实验
        centralized_results = self.run_centralized_experiment(
            train_data, train_labels, test_data, test_labels, config
        )
        
        # 4. 运行联邦实验
        federated_results = self.run_federated_experiment(
            federated_data, test_data, test_labels, config
        )
        
        # 5. 可视化结果
        self.visualize_results(centralized_results, federated_results, test_labels)
        
        # 6. 保存结果
        self.save_results(centralized_results, federated_results, config)
        
        # 7. 打印总结
        self.print_experiment_summary(centralized_results, federated_results)
        
        return centralized_results, federated_results
    
    def print_experiment_summary(self, centralized_results: Dict, federated_results: Dict):
        """打印实验总结"""
        self.logger.info("\n" + "="*60)
        self.logger.info(f"{self.dataset_name.upper()} ORCHESTRA 实验总结")
        self.logger.info("="*60)
        
        # 中心化结果
        if 'final_results' in centralized_results:
            cent_results = centralized_results['final_results']
            self.logger.info("\n中心化Orchestra结果:")
            self.logger.info(f"  ARI Score: {cent_results.get('adjusted_rand_score', 'N/A'):.4f}")
            self.logger.info(f"  NMI Score: {cent_results.get('normalized_mutual_info', 'N/A'):.4f}")
            self.logger.info(f"  Silhouette Score: {cent_results.get('silhouette_score', 'N/A'):.4f}")
            self.logger.info(f"  使用聚类数: {cent_results.get('num_clusters_used', 'N/A')}")
            self.logger.info(f"  聚类熵: {cent_results.get('cluster_entropy', 'N/A'):.4f}")
        
        # 联邦结果
        if 'final_results' in federated_results and federated_results['final_results']:
            fed_results = federated_results['final_results']
            self.logger.info("\n联邦Orchestra结果:")
            for key, value in fed_results.items():
                self.logger.info(f"  {key}: {value}")
        elif 'error' in federated_results:
            self.logger.info(f"\n联邦实验失败: {federated_results['error']}")
        
        self.logger.info("\n实验配置:")
        self.logger.info(f"  数据集: {self.dataset_name.upper()}")
        self.logger.info(f"  参与方数量: {self.num_parties}")
        self.logger.info(f"  数据分割策略: {self.split_strategy}")
        self.logger.info(f"  输出目录: {self.output_dir}")
        
        self.logger.info("\n" + "="*60)

# 主实验运行函数
def run_cifar_experiments(datasets: List[str] = ['cifar10', 'cifar100'],
                        num_parties: int = 3,
                        split_strategy: str = 'iid',
                        output_base_dir: str = './orchestra_results'):
    """运行CIFAR实验"""
    
    results = {}
    
    for dataset in datasets:
        print(f"\n{'='*80}")
        print(f"开始 {dataset.upper()} 实验")
        print(f"{'='*80}")
        
        # 创建数据集特定的输出目录
        output_dir = os.path.join(output_base_dir, dataset)
        
        # 创建实验
        experiment = CIFAROrchestralExperiment(
            dataset_name=dataset,
            num_parties=num_parties,
            split_strategy=split_strategy,
            output_dir=output_dir
        )
        
        # 配置
        config = OrchestraConfig(
            input_dim=3072,  # 32*32*3
            hidden_dims=[1024, 512, 256],
            embedding_dim=128,
            num_clusters=10 if dataset == 'cifar10' else 100,
            dropout_rate=0.2,
            temperature=0.5,
            learning_rate=0.001,
            batch_size=256,
            num_epochs=100,
            communication_rounds=20,
            local_epochs=5
        )
        
        # 运行实验
        try:
            centralized_results, federated_results = experiment.run_complete_experiment(config)
            results[dataset] = {
                'centralized': centralized_results,
                'federated': federated_results
            }
        except Exception as e:
            print(f"实验失败: {e}")
            results[dataset] = {'error': str(e)}
    
    return results

if __name__ == "__main__":
    # 运行完整实验
    results = run_cifar_experiments(
        datasets=['cifar10', 'cifar100'],
        num_parties=3,
        split_strategy='iid'
    )
    
    print("\n所有实验完成！")
    for dataset, result in results.items():
        if 'error' in result:
            print(f"{dataset}: 失败 - {result['error']}")
        else:
            print(f"{dataset}: 成功完成")