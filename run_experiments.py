#!/usr/bin/env python3
"""
Orchestra实验运行脚本
提供命令行接口来运行CIFAR-10和CIFAR-100上的Orchestra实验
"""

import argparse

import sys
import json
from datetime import datetime
from typing import List, Dict
import os
# 显式设置后端（根据环境选择）
if os.environ.get('MPLBACKEND') == 'module://matplotlib_inline.backend_inline':
    os.environ['MPLBACKEND'] = 'agg'  # 替换为兼容的后端

import matplotlib.pyplot as plt

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from cifar_experiments import CIFAROrchestralExperiment, run_cifar_experiments
from federated_orchestra import OrchestraConfig

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='运行Orchestra联邦学习实验',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 数据集选择
    parser.add_argument(
        '--datasets', 
        nargs='+', 
        choices=['cifar10', 'cifar100'], 
        default=['cifar10'],
        help='要运行实验的数据集'
    )
    
    # 联邦学习参数
    parser.add_argument(
        '--num-parties', 
        type=int, 
        default=3,
        help='联邦学习参与方数量'
    )
    
    parser.add_argument(
        '--split-strategy', 
        choices=['iid', 'non_iid_dirichlet', 'non_iid_pathological'], 
        default='iid',
        help='数据分割策略'
    )
    
    # 模型参数
    parser.add_argument(
        '--hidden-dims', 
        nargs='+', 
        type=int, 
        default=[1024, 512, 256],
        help='隐藏层维度'
    )
    
    parser.add_argument(
        '--embedding-dim', 
        type=int, 
        default=128,
        help='嵌入维度'
    )
    
    parser.add_argument(
        '--dropout-rate', 
        type=float, 
        default=0.2,
        help='Dropout率'
    )
    
    parser.add_argument(
        '--temperature', 
        type=float, 
        default=0.5,
        help='对比学习温度参数'
    )
    
    # 训练参数
    parser.add_argument(
        '--learning-rate', 
        type=float, 
        default=0.001,
        help='学习率'
    )
    
    parser.add_argument(
        '--batch-size', 
        type=int, 
        default=256,
        help='批次大小'
    )
    
    parser.add_argument(
        '--num-epochs', 
        type=int, 
        default=50,
        help='训练轮数'
    )
    
    parser.add_argument(
        '--communication-rounds', 
        type=int, 
        default=20,
        help='联邦通信轮数'
    )
    
    parser.add_argument(
        '--local-epochs', 
        type=int, 
        default=5,
        help='本地训练轮数'
    )
    
    # 损失权重
    parser.add_argument(
        '--contrastive-weight', 
        type=float, 
        default=1.0,
        help='对比学习损失权重'
    )
    
    parser.add_argument(
        '--clustering-weight', 
        type=float, 
        default=1.0,
        help='聚类损失权重'
    )
    
    parser.add_argument(
        '--consistency-weight', 
        type=float, 
        default=0.5,
        help='一致性损失权重'
    )
    
    # 输出设置
    parser.add_argument(
        '--output-dir', 
        type=str, 
        default='./orchestra_results',
        help='结果输出目录'
    )
    
    parser.add_argument(
        '--experiment-name', 
        type=str, 
        default=None,
        help='实验名称（用于区分不同实验）'
    )
    
    # 其他选项
    parser.add_argument(
        '--seed', 
        type=int, 
        default=42,
        help='随机种子'
    )
    
    parser.add_argument(
        '--device', 
        choices=['auto', 'cpu', 'cuda'], 
        default='auto',
        help='计算设备'
    )
    
    parser.add_argument(
        '--verbose', 
        action='store_true',
        help='详细输出'
    )
    
    parser.add_argument(
        '--save-models', 
        action='store_true',
        help='保存训练好的模型'
    )
    
    parser.add_argument(
        '--skip-visualization', 
        action='store_true',
        help='跳过结果可视化'
    )
    
    return parser.parse_args()

def setup_experiment_environment(args):
    """设置实验环境"""
    import torch
    import numpy as np
    import random
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    
    # 设置设备
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"使用设备: {device}")
    print(f"随机种子: {args.seed}")
    
    return device

def create_experiment_config(args, dataset: str) -> OrchestraConfig:
    """创建实验配置"""
    # 根据数据集设置聚类数
    num_clusters = 10 if dataset == 'cifar10' else 100
    
    config = OrchestraConfig(
        input_dim=3072,  # 32*32*3 for CIFAR
        hidden_dims=args.hidden_dims,
        embedding_dim=args.embedding_dim,
        num_clusters=num_clusters,
        dropout_rate=args.dropout_rate,
        temperature=args.temperature,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        communication_rounds=args.communication_rounds,
        local_epochs=args.local_epochs,
        contrastive_weight=args.contrastive_weight,
        clustering_weight=args.clustering_weight,
        consistency_weight=args.consistency_weight
    )
    
    return config

def run_single_experiment(dataset: str, args, device: str) -> Dict:
    """运行单个数据集的实验"""
    print(f"\n{'='*80}")
    print(f"开始 {dataset.upper()} 实验")
    print(f"{'='*80}")
    
    # 创建实验名称
    if args.experiment_name:
        exp_name = f"{args.experiment_name}_{dataset}"
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_name = f"orchestra_{dataset}_{timestamp}"
    
    # 创建输出目录
    output_dir = os.path.join(args.output_dir, exp_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存实验配置
    config = create_experiment_config(args, dataset)
    config_path = os.path.join(output_dir, 'experiment_config.json')
    with open(config_path, 'w', encoding='utf-8') as f:
        config_dict = config.__dict__.copy()
        config_dict.update({
            'dataset': dataset,
            'num_parties': args.num_parties,
            'split_strategy': args.split_strategy,
            'seed': args.seed,
            'device': device,
            'experiment_name': exp_name,
            'timestamp': datetime.now().isoformat()
        })
        json.dump(config_dict, f, indent=2, ensure_ascii=False)
    
    print(f"实验配置已保存: {config_path}")
    
    # 创建实验对象
    experiment = CIFAROrchestralExperiment(
        dataset_name=dataset,
        num_parties=args.num_parties,
        split_strategy=args.split_strategy,
        output_dir=output_dir
    )
    
    # 运行实验
    try:
        centralized_results, federated_results = experiment.run_complete_experiment(config)
        
        # 保存额外的实验元数据
        metadata = {
            'experiment_name': exp_name,
            'dataset': dataset,
            'success': True,
            'completion_time': datetime.now().isoformat(),
            'config': config_dict
        }
        
        metadata_path = os.path.join(output_dir, 'experiment_metadata.json')
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        return {
            'success': True,
            'centralized': centralized_results,
            'federated': federated_results,
            'output_dir': output_dir,
            'experiment_name': exp_name
        }
        
    except Exception as e:
        print(f"实验失败: {e}")
        
        # 保存错误信息
        error_metadata = {
            'experiment_name': exp_name,
            'dataset': dataset,
            'success': False,
            'error': str(e),
            'completion_time': datetime.now().isoformat(),
            'config': config_dict
        }
        
        error_path = os.path.join(output_dir, 'experiment_error.json')
        with open(error_path, 'w', encoding='utf-8') as f:
            json.dump(error_metadata, f, indent=2, ensure_ascii=False)
        
        return {
            'success': False,
            'error': str(e),
            'output_dir': output_dir,
            'experiment_name': exp_name
        }

def print_experiment_summary(results: Dict[str, Dict]):
    """打印所有实验的总结"""
    print("\n" + "="*100)
    print("ORCHESTRA 实验总结")
    print("="*100)
    
    successful_experiments = 0
    failed_experiments = 0
    
    for dataset, result in results.items():
        print(f"\n{dataset.upper()}:")
        if result['success']:
            successful_experiments += 1
            print(f"  ✓ 成功完成")
            print(f"  📁 输出目录: {result['output_dir']}")
            
            # 打印关键指标
            if 'centralized' in result and 'final_results' in result['centralized']:
                final_results = result['centralized']['final_results']
                print(f"  📊 ARI Score: {final_results.get('adjusted_rand_score', 'N/A'):.4f}")
                print(f"  📊 NMI Score: {final_results.get('normalized_mutual_info', 'N/A'):.4f}")
                print(f"  📊 Silhouette Score: {final_results.get('silhouette_score', 'N/A'):.4f}")
        else:
            failed_experiments += 1
            print(f"  ✗ 失败")
            print(f"  ❌ 错误: {result['error']}")
            print(f"  📁 错误日志: {result['output_dir']}")
    
    print(f"\n总计:")
    print(f"  成功: {successful_experiments}")
    print(f"  失败: {failed_experiments}")
    print(f"  总计: {successful_experiments + failed_experiments}")
    
    if successful_experiments > 0:
        print(f"\n🎉 实验完成！查看结果目录获取详细信息。")
    
    print("="*100)

def main():
    """主函数"""
    # 解析参数
    args = parse_arguments()
    
    # 打印实验信息
    print("Orchestra 联邦学习实验")
    print(f"数据集: {', '.join(args.datasets)}")
    print(f"参与方数量: {args.num_parties}")
    print(f"分割策略: {args.split_strategy}")
    print(f"输出目录: {args.output_dir}")
    
    # 设置环境
    device = setup_experiment_environment(args)
    
    # 创建主输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 运行实验
    results = {}
    
    for dataset in args.datasets:
        result = run_single_experiment(dataset, args, device)
        results[dataset] = result
    
    # 打印总结
    print_experiment_summary(results)
    
    # 保存总体结果
    summary_path = os.path.join(args.output_dir, 'experiment_summary.json')
    summary_data = {
        'timestamp': datetime.now().isoformat(),
        'arguments': vars(args),
        'device': device,
        'results': {k: {key: v[key] for key in ['success', 'experiment_name', 'output_dir'] if key in v} 
                   for k, v in results.items()}
    }
    
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n实验总结已保存: {summary_path}")
    
    return results

if __name__ == "__main__":
    try:
        results = main()
        
        # 检查是否有失败的实验
        failed_count = sum(1 for r in results.values() if not r['success'])
        if failed_count > 0:
            sys.exit(1)  # 有失败的实验时返回错误码
        else:
            sys.exit(0)  # 所有实验成功
            
    except KeyboardInterrupt:
        print("\n实验被用户中断")
        sys.exit(130)
    except Exception as e:
        print(f"\n实验运行出错: {e}")
        sys.exit(1)