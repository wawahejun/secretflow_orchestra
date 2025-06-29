#!/usr/bin/env python3
"""
Orchestra 演示脚本
展示如何使用Orchestra实现进行基本的无监督聚类任务
"""

import os
import sys
import torch
import numpy as np

# 设置matplotlib后端（解决Colab环境问题）
# 清除可能冲突的环境变量
if 'MPLBACKEND' in os.environ:
    del os.environ['MPLBACKEND']

import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs, make_circles, make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import warnings
warnings.filterwarnings('ignore')

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from orchestra_model import OrchestraModel, OrchestraLoss, OrchestraTrainer
from federated_orchestra import OrchestraConfig, OrchestraDataProcessor

def create_synthetic_datasets():
    """创建合成数据集用于演示"""
    datasets = {}
    
    # 1. 简单的高斯聚类
    X1, y1 = make_blobs(n_samples=800, centers=4, n_features=2, 
                       random_state=42, cluster_std=1.5)
    datasets['blobs'] = (X1, y1, '高斯聚类')
    
    # 2. 圆形聚类
    X2, y2 = make_circles(n_samples=800, noise=0.1, factor=0.3, random_state=42)
    datasets['circles'] = (X2, y2, '圆形聚类')
    
    # 3. 月牙形聚类
    X3, y3 = make_moons(n_samples=800, noise=0.1, random_state=42)
    datasets['moons'] = (X3, y3, '月牙形聚类')
    
    # 4. 高维数据（模拟真实场景）
    X4, y4 = make_blobs(n_samples=1000, centers=5, n_features=50, 
                       random_state=42, cluster_std=2.0)
    datasets['high_dim'] = (X4, y4, '高维聚类')
    
    return datasets

def visualize_2d_data(X, y_true, y_pred, title, save_path=None):
    """可视化2D数据的聚类结果"""
    if X.shape[1] != 2:
        print(f"跳过可视化 {title}（不是2D数据）")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 真实标签
    scatter1 = ax1.scatter(X[:, 0], X[:, 1], c=y_true, cmap='tab10', alpha=0.7)
    ax1.set_title(f'{title} - 真实标签')
    ax1.grid(True, alpha=0.3)
    
    # 预测标签
    scatter2 = ax2.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='tab10', alpha=0.7)
    ax2.set_title(f'{title} - 预测聚类')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"可视化结果已保存: {save_path}")
    
    plt.show()

def run_orchestra_clustering(X, y_true, dataset_name, config):
    """运行Orchestra聚类"""
    print(f"\n{'='*60}")
    print(f"运行 {dataset_name} 聚类")
    print(f"数据形状: {X.shape}, 真实聚类数: {len(np.unique(y_true))}")
    print(f"{'='*60}")
    
    # 数据预处理
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 转换为tensor
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y_true, dtype=torch.long)
    
    # 创建模型
    model = OrchestraModel(
        input_dim=X.shape[1],
        hidden_dims=config.hidden_dims,
        embedding_dim=config.embedding_dim,
        num_clusters=config.num_clusters,
        dropout_rate=config.dropout_rate,
        temperature=config.temperature
    )
    
    # 创建损失函数和优化器
    loss_fn = OrchestraLoss(
        contrastive_weight=config.contrastive_weight,
        clustering_weight=config.clustering_weight,
        consistency_weight=config.consistency_weight,
        temperature=config.temperature
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    
    # 创建训练器
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trainer = OrchestraTrainer(model, loss_fn, optimizer, device)
    
    print(f"使用设备: {device}")
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters())}")
    
    # 训练
    print("\n开始训练...")
    training_history = {
        'epoch': [],
        'total_loss': [],
        'contrastive_loss': [],
        'clustering_loss': [],
        'consistency_loss': [],
        'ari_score': [],
        'nmi_score': []
    }
    
    for epoch in range(config.num_epochs):
        # 模拟联邦学习：将数据分成两个批次
        mid = len(X_tensor) // 2
        data_batches = [X_tensor[:mid], X_tensor[mid:]]
        
        # 训练步骤
        losses = trainer.train_step(data_batches)
        
        # 记录损失
        training_history['epoch'].append(epoch)
        for key in ['total_loss', 'contrastive_loss', 'clustering_loss', 'consistency_loss']:
            loss_key = key.replace('_loss', '')
            training_history[key].append(losses[loss_key].item())
        
        # 每10轮评估一次
        if (epoch + 1) % 10 == 0 or epoch == config.num_epochs - 1:
            eval_results = trainer.evaluate(X_tensor, y_tensor)
            
            ari = eval_results.get('adjusted_rand_score', 0)
            nmi = eval_results.get('normalized_mutual_info', 0)
            
            training_history['ari_score'].append(ari)
            training_history['nmi_score'].append(nmi)
            
            print(f"Epoch {epoch+1:3d}/{config.num_epochs}: "
                  f"Loss={losses['total']:.4f}, "
                  f"ARI={ari:.4f}, "
                  f"NMI={nmi:.4f}")
    
    # 最终评估
    print("\n最终评估...")
    final_results = trainer.evaluate(X_tensor, y_tensor)
    
    # 获取聚类分配
    cluster_assignments = trainer.get_cluster_assignments(X_tensor).numpy()
    embeddings = trainer.get_embeddings(X_tensor).numpy()
    
    # 打印结果
    print("\n最终结果:")
    for key, value in final_results.items():
        if isinstance(value, (int, float)):
            print(f"  {key}: {value:.4f}")
    
    return {
        'training_history': training_history,
        'final_results': final_results,
        'cluster_assignments': cluster_assignments,
        'embeddings': embeddings,
        'model': model
    }

def plot_training_curves(training_history, dataset_name, save_path=None):
    """绘制训练曲线"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    epochs = training_history['epoch']
    
    # 损失曲线
    ax1.plot(epochs, training_history['total_loss'], 'b-', label='Total Loss')
    ax1.plot(epochs, training_history['contrastive_loss'], 'r-', label='Contrastive')
    ax1.plot(epochs, training_history['clustering_loss'], 'g-', label='Clustering')
    ax1.plot(epochs, training_history['consistency_loss'], 'm-', label='Consistency')
    ax1.set_title(f'{dataset_name} - 训练损失')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 聚类性能
    eval_epochs = list(range(10, len(training_history['ari_score']) * 10 + 1, 10))
    if len(eval_epochs) != len(training_history['ari_score']):
        eval_epochs = list(range(len(training_history['ari_score'])))
    
    ax2.plot(eval_epochs, training_history['ari_score'], 'o-', label='ARI Score')
    ax2.plot(eval_epochs, training_history['nmi_score'], 's-', label='NMI Score')
    ax2.set_title(f'{dataset_name} - 聚类性能')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Score')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 损失分解（饼图）
    final_losses = {
        'Contrastive': training_history['contrastive_loss'][-1],
        'Clustering': training_history['clustering_loss'][-1],
        'Consistency': training_history['consistency_loss'][-1]
    }
    
    ax3.pie(final_losses.values(), labels=final_losses.keys(), autopct='%1.1f%%')
    ax3.set_title(f'{dataset_name} - 最终损失分解')
    
    # 性能趋势
    if len(training_history['ari_score']) > 1:
        ax4.plot(eval_epochs, training_history['ari_score'], 'o-', color='blue', alpha=0.7)
        ax4.set_title(f'{dataset_name} - ARI Score 趋势')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('ARI Score')
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"训练曲线已保存: {save_path}")
    
    plt.show()

def run_federated_demo(X, y_true, dataset_name):
    """运行联邦学习演示"""
    print(f"\n{'='*60}")
    print(f"联邦学习演示 - {dataset_name}")
    print(f"{'='*60}")
    
    # 创建联邦数据分割
    federated_data = OrchestraDataProcessor.create_federated_split(
        data=X,
        labels=y_true,
        num_parties=3,
        split_strategy='iid'
    )
    
    print("联邦数据分割:")
    for party, (data, labels) in federated_data.items():
        unique_labels = np.unique(labels)
        print(f"  {party}: {len(data)} 样本, {len(unique_labels)} 类别")
    
    return federated_data

def main():
    """主演示函数"""
    print("Orchestra 无监督联邦学习演示")
    print("="*80)
    
    # 检查环境
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    print(f"PyTorch版本: {torch.__version__}")
    
    # 创建输出目录
    output_dir = './demo_results'
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建合成数据集
    print("\n创建合成数据集...")
    datasets = create_synthetic_datasets()
    
    # 为每个数据集运行演示
    all_results = {}
    
    for dataset_key, (X, y_true, description) in datasets.items():
        print(f"\n处理数据集: {description}")
        
        # 配置参数
        num_clusters = len(np.unique(y_true))
        
        if dataset_key == 'high_dim':
            # 高维数据使用更大的网络
            config = OrchestraConfig(
                input_dim=X.shape[1],
                hidden_dims=[256, 128, 64],
                embedding_dim=32,
                num_clusters=num_clusters,
                num_epochs=50,
                batch_size=64,
                learning_rate=0.001
            )
        else:
            # 2D数据使用简单网络
            config = OrchestraConfig(
                input_dim=X.shape[1],
                hidden_dims=[64, 32],
                embedding_dim=16,
                num_clusters=num_clusters,
                num_epochs=30,
                batch_size=32,
                learning_rate=0.001
            )
        
        # 运行Orchestra聚类
        results = run_orchestra_clustering(X, y_true, description, config)
        all_results[dataset_key] = results
        
        # 可视化结果（仅2D数据）
        if X.shape[1] == 2:
            vis_path = os.path.join(output_dir, f'{dataset_key}_clustering.png')
            visualize_2d_data(X, y_true, results['cluster_assignments'], 
                            description, vis_path)
        
        # 绘制训练曲线
        curve_path = os.path.join(output_dir, f'{dataset_key}_training.png')
        plot_training_curves(results['training_history'], description, curve_path)
        
        # 联邦学习演示
        federated_data = run_federated_demo(X, y_true, description)
    
    # 总结结果
    print("\n" + "="*80)
    print("演示结果总结")
    print("="*80)
    
    for dataset_key, results in all_results.items():
        dataset_name = datasets[dataset_key][2]
        final_results = results['final_results']
        
        print(f"\n{dataset_name}:")
        print(f"  ARI Score: {final_results.get('adjusted_rand_score', 0):.4f}")
        print(f"  NMI Score: {final_results.get('normalized_mutual_info', 0):.4f}")
        print(f"  Silhouette Score: {final_results.get('silhouette_score', 0):.4f}")
        print(f"  使用聚类数: {final_results.get('num_clusters_used', 'N/A')}")
    
    print(f"\n所有结果已保存到: {output_dir}")
    print("\n🎉 Orchestra演示完成！")
    
    # 使用建议
    print("\n" + "="*80)
    print("下一步建议")
    print("="*80)
    print("1. 运行完整CIFAR实验:")
    print("   python run_experiments.py --datasets cifar10 --num-epochs 20")
    print("\n2. 运行功能测试:")
    print("   python test_orchestra.py")
    print("\n3. 查看详细使用指南:")
    print("   查看 setup_guide.md 文件")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n演示被用户中断")
    except Exception as e:
        print(f"\n演示运行出错: {e}")
        import traceback
        traceback.print_exc()