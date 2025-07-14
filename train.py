#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Orchestra联邦学习训练脚本
实现完整的训练流程，包括数据加载、模型训练、评估等
"""

import os
import sys
import time
import logging
import argparse
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# 导入自定义模块
from config import get_config, get_eval_config, print_config
from data_utils import create_federated_cifar10, create_data_loaders
from models import create_orchestra_model
from evaluation import ComprehensiveEvaluator


class OrchestraTrainer:
    """Orchestra训练器"""
    
    def __init__(self, config: Dict, device: torch.device):
        self.config = config
        self.device = device
        self.logger = self._setup_logger()
        
        # 创建输出目录
        os.makedirs(config['output_dir'], exist_ok=True)
        
        # 训练历史
        self.history = defaultdict(list)
        
        # 最佳模型状态
        self.best_model_state = None
        self.best_metric = 0.0
    
    def _setup_logger(self) -> logging.Logger:
        """设置日志"""
        logger = logging.getLogger('OrchestraTrainer')
        logger.setLevel(logging.INFO)
        
        # 创建文件处理器
        log_file = os.path.join(self.config['output_dir'], 'training.log')
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # 创建控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # 创建格式器
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # 添加处理器
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def create_model(self) -> nn.Module:
        """创建Orchestra模型"""
        model = create_orchestra_model(
            backbone_type=self.config.get('model_type', 'resnet18'),
            projection_dim=self.config.get('projection_dim', 128),
            num_local_clusters=self.config['num_local_clusters'],
            num_global_clusters=self.config['num_global_clusters'],
            memory_size=self.config['memory_size'],
            temperature=self.config['temperature'],
            ema_decay=self.config['ema_decay']
        )
        
        return model.to(self.device)
    
    def create_optimizer(self, model: nn.Module) -> optim.Optimizer:
        """创建优化器"""
        optimizer_type = self.config.get('optimizer', 'adam').lower()
        
        if optimizer_type == 'adam':
            optimizer = optim.Adam(
                model.parameters(),
                lr=self.config['learning_rate'],
                weight_decay=self.config.get('weight_decay', 1e-4)
            )
        elif optimizer_type == 'sgd':
            optimizer = optim.SGD(
                model.parameters(),
                lr=self.config['learning_rate'],
                momentum=self.config.get('momentum', 0.9),
                weight_decay=self.config.get('weight_decay', 1e-4)
            )
        else:
            raise ValueError(f"不支持的优化器类型: {optimizer_type}")
        
        return optimizer
    
    def create_scheduler(self, optimizer: optim.Optimizer) -> Optional[optim.lr_scheduler._LRScheduler]:
        """创建学习率调度器"""
        scheduler_type = self.config.get('scheduler', None)
        
        if scheduler_type is None:
            return None
        elif scheduler_type.lower() == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.config['num_rounds']
            )
        elif scheduler_type.lower() == 'step':
            return optim.lr_scheduler.StepLR(
                optimizer, 
                step_size=self.config.get('step_size', 30),
                gamma=self.config.get('gamma', 0.1)
            )
        else:
            raise ValueError(f"不支持的调度器类型: {scheduler_type}")
    
    def train_epoch(self, 
                   model: nn.Module, 
                   data_loader: DataLoader, 
                   optimizer: optim.Optimizer,
                   epoch: int) -> Dict[str, float]:
        """训练一个epoch"""
        model.train()
        
        epoch_losses = defaultdict(list)
        total_samples = 0
        
        for batch_idx, (data, target) in enumerate(data_loader):
            # 处理数据
            if isinstance(data, (list, tuple)):
                if len(data) == 3:  # (img1, img2, img3)
                    x1, x2, x3 = data
                    x1, x2, x3 = x1.to(self.device), x2.to(self.device), x3.to(self.device)
                else:  # (img1, img2)
                    x1, x2 = data
                    x1, x2 = x1.to(self.device), x2.to(self.device)
                    x3 = None
            else:
                # 单视图数据，创建两个相同的视图
                x1 = x2 = data.to(self.device)
                x3 = None
            
            # 处理标签
            if isinstance(target, (list, tuple)):
                if len(target) == 3:  # (label1, label2, rotation_label)
                    rotation_labels = target[2].to(self.device)
                else:
                    rotation_labels = None
            else:
                rotation_labels = None
            
            batch_size = x1.size(0)
            total_samples += batch_size
            
            # 前向传播
            optimizer.zero_grad()
            
            losses = model(x1, x2, x3, rotation_labels)
            
            # 计算总损失
            total_loss = 0
            for loss_name, loss_value in losses.items():
                weight_name = f'{loss_name}_weight'
                weight = self.config.get(weight_name, 1.0)
                total_loss += weight * loss_value
                epoch_losses[loss_name].append(loss_value.item())
            
            epoch_losses['total'].append(total_loss.item())
            
            # 反向传播
            total_loss.backward()
            optimizer.step()
            
            # 日志记录
            if batch_idx % self.config.get('log_interval', 10) == 0:
                self.logger.info(
                    f'Epoch {epoch}, Batch {batch_idx}/{len(data_loader)}, '
                    f'Total Loss: {total_loss.item():.4f}'
                )
        
        # 计算平均损失
        avg_losses = {}
        for loss_name, loss_values in epoch_losses.items():
            avg_losses[loss_name] = np.mean(loss_values)
        
        return avg_losses
    
    def federated_training(self, client_data: Dict[str, Dict[str, np.ndarray]]) -> nn.Module:
        """联邦学习训练"""
        
        self.logger.info("开始联邦学习训练...")
        
        # 创建全局模型
        global_model = self.create_model()
        
        # 客户端模型列表
        client_models = []
        client_optimizers = []
        client_schedulers = []
        client_loaders = []
        
        # 为每个客户端创建模型和数据加载器
        for i in range(self.config['num_clients']):
            client_id = f'client_{i}'
            
            # 创建客户端模型（复制全局模型）
            client_model = self.create_model()
            client_model.load_state_dict(global_model.state_dict())
            client_models.append(client_model)
            
            # 创建优化器和调度器
            optimizer = self.create_optimizer(client_model)
            scheduler = self.create_scheduler(optimizer)
            client_optimizers.append(optimizer)
            client_schedulers.append(scheduler)
            
            # 创建数据加载器
            train_loader, _ = create_data_loaders(
                client_data, self.config, client_id,
                return_two_views=True, return_rotation=True
            )
            client_loaders.append(train_loader)
        
        # 联邦学习轮次
        for round_idx in range(self.config['num_rounds']):
            self.logger.info(f"\n=== 联邦学习轮次 {round_idx + 1}/{self.config['num_rounds']} ===")
            
            round_start_time = time.time()
            
            # 选择参与的客户端
            participating_clients = self._select_clients(round_idx)
            
            # 客户端本地训练
            client_weights = []
            client_losses = []
            
            for client_idx in participating_clients:
                self.logger.info(f"客户端 {client_idx} 开始本地训练...")
                
                # 同步全局模型到客户端
                client_models[client_idx].load_state_dict(global_model.state_dict())
                
                # 本地训练
                local_losses = self._local_training(
                    client_models[client_idx],
                    client_loaders[client_idx],
                    client_optimizers[client_idx],
                    round_idx
                )
                
                client_losses.append(local_losses)
                client_weights.append(client_models[client_idx].state_dict())
                
                # 更新学习率
                if client_schedulers[client_idx] is not None:
                    client_schedulers[client_idx].step()
            
            # 联邦聚合
            self.logger.info("执行联邦聚合...")
            aggregated_weights = self._federated_averaging(client_weights)
            global_model.load_state_dict(aggregated_weights)
            
            # 记录训练历史
            avg_losses = self._average_client_losses(client_losses)
            for loss_name, loss_value in avg_losses.items():
                self.history[loss_name].append(loss_value)
            
            round_time = time.time() - round_start_time
            self.logger.info(f"轮次 {round_idx + 1} 完成，耗时: {round_time:.2f}秒")
            
            # 定期评估和保存
            if (round_idx + 1) % self.config.get('save_interval', 20) == 0:
                self._save_checkpoint(global_model, round_idx + 1)
            
            # 早停检查（如果配置了）
            if self.config.get('early_stopping', False):
                if self._check_early_stopping(avg_losses):
                    self.logger.info(f"早停触发，在轮次 {round_idx + 1} 停止训练")
                    break
        
        self.logger.info("联邦学习训练完成！")
        return global_model
    
    def _select_clients(self, round_idx: int) -> List[int]:
        """选择参与训练的客户端"""
        clients_per_round = self.config.get('clients_per_round', self.config['num_clients'])
        
        if clients_per_round >= self.config['num_clients']:
            return list(range(self.config['num_clients']))
        else:
            # 随机选择客户端
            np.random.seed(round_idx)  # 确保可重现性
            selected = np.random.choice(
                self.config['num_clients'], 
                clients_per_round, 
                replace=False
            )
            return selected.tolist()
    
    def _local_training(self, 
                       model: nn.Module, 
                       data_loader: DataLoader, 
                       optimizer: optim.Optimizer,
                       round_idx: int) -> Dict[str, float]:
        """客户端本地训练"""
        
        local_losses = defaultdict(list)
        
        for epoch in range(self.config['local_epochs']):
            epoch_losses = self.train_epoch(model, data_loader, optimizer, epoch)
            
            for loss_name, loss_value in epoch_losses.items():
                local_losses[loss_name].append(loss_value)
        
        # 计算平均损失
        avg_local_losses = {}
        for loss_name, loss_values in local_losses.items():
            avg_local_losses[loss_name] = np.mean(loss_values)
        
        return avg_local_losses
    
    def _federated_averaging(self, client_weights: List[Dict]) -> Dict:
        """联邦平均聚合"""
        
        # 获取第一个客户端的权重作为模板
        avg_weights = {}
        
        for key in client_weights[0].keys():
            # 计算所有客户端对应参数的平均值
            avg_weights[key] = torch.stack([
                client_weights[i][key] for i in range(len(client_weights))
            ]).mean(dim=0)
        
        return avg_weights
    
    def _average_client_losses(self, client_losses: List[Dict[str, float]]) -> Dict[str, float]:
        """计算客户端损失的平均值"""
        
        if not client_losses:
            return {}
        
        avg_losses = {}
        loss_names = client_losses[0].keys()
        
        for loss_name in loss_names:
            avg_losses[loss_name] = np.mean([
                client_loss[loss_name] for client_loss in client_losses
            ])
        
        return avg_losses
    
    def _save_checkpoint(self, model: nn.Module, round_idx: int):
        """保存检查点"""
        checkpoint = {
            'round': round_idx,
            'model_state_dict': model.state_dict(),
            'config': self.config,
            'history': dict(self.history)
        }
        
        checkpoint_path = os.path.join(
            self.config['output_dir'], 
            f'checkpoint_round_{round_idx}.pth'
        )
        
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"检查点已保存: {checkpoint_path}")
    
    def _check_early_stopping(self, current_losses: Dict[str, float]) -> bool:
        """检查是否应该早停"""
        # 简单的早停策略：如果总损失在最近几轮没有改善
        patience = self.config.get('early_stopping_patience', 10)
        min_delta = self.config.get('early_stopping_min_delta', 1e-4)
        
        if len(self.history['total']) < patience:
            return False
        
        recent_losses = self.history['total'][-patience:]
        best_recent_loss = min(recent_losses)
        current_loss = current_losses.get('total', float('inf'))
        
        return current_loss > best_recent_loss + min_delta
    
    def evaluate_model(self, 
                      model: nn.Module, 
                      client_data: Dict[str, Dict[str, np.ndarray]]) -> Dict:
        """评估模型"""
        
        self.logger.info("开始模型评估...")
        
        # 创建评估器
        evaluator = ComprehensiveEvaluator(
            num_classes=self.config['num_classes'],
            num_clusters=self.config['num_classes'],
            output_dir=os.path.join(self.config['output_dir'], 'evaluation')
        )
        
        # 合并所有客户端的测试数据
        all_test_data = []
        all_test_labels = []
        
        for client_id, data in client_data.items():
            all_test_data.append(data['x_test'])
            all_test_labels.append(data['y_test'])
        
        all_test_data = np.concatenate(all_test_data, axis=0)
        all_test_labels = np.concatenate(all_test_labels, axis=0)
        
        # 创建测试数据加载器
        from data_utils import CIFAR10FederatedDataset, get_cifar10_transforms
        
        _, test_transform = get_cifar10_transforms(self.config)
        test_dataset = CIFAR10FederatedDataset(
            all_test_data, all_test_labels, 
            transform=test_transform, 
            return_two_views=False, 
            return_rotation=False
        )
        
        test_loader = DataLoader(
            test_dataset, 
            batch_size=self.config['batch_size'], 
            shuffle=False,
            num_workers=self.config.get('num_workers', 4)
        )
        
        # 创建训练数据加载器（用于线性评估）
        all_train_data = []
        all_train_labels = []
        
        for client_id, data in client_data.items():
            all_train_data.append(data['x_train'])
            all_train_labels.append(data['y_train'])
        
        all_train_data = np.concatenate(all_train_data, axis=0)
        all_train_labels = np.concatenate(all_train_labels, axis=0)
        
        train_transform, _ = get_cifar10_transforms(self.config)
        train_dataset = CIFAR10FederatedDataset(
            all_train_data, all_train_labels, 
            transform=train_transform, 
            return_two_views=False, 
            return_rotation=False
        )
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config['batch_size'], 
            shuffle=False,
            num_workers=self.config.get('num_workers', 4)
        )
        
        # 执行评估
        results = evaluator.evaluate_model(
            model, train_loader, test_loader, self.device
        )
        
        return results
    
    def save_final_model(self, model: nn.Module):
        """保存最终模型"""
        model_path = os.path.join(self.config['output_dir'], 'final_model.pth')
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': self.config,
            'history': dict(self.history)
        }, model_path)
        
        self.logger.info(f"最终模型已保存: {model_path}")


def main():
    parser = argparse.ArgumentParser(description='Orchestra CIFAR-10 联邦学习训练')
    
    # 配置参数
    parser.add_argument('--config', type=str, default='small', 
                       help='配置类型 (base, small, large, etc.)')
    parser.add_argument('--data_dir', type=str, default='./data', help='数据目录')
    parser.add_argument('--output_dir', type=str, default='./output', help='输出目录')
    parser.add_argument('--device', type=str, default='auto', 
                       help='设备类型 (cpu, cuda, auto)')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    
    # 训练参数覆盖
    parser.add_argument('--num_rounds', type=int, help='联邦学习轮数')
    parser.add_argument('--num_clients', type=int, help='客户端数量')
    parser.add_argument('--learning_rate', type=float, help='学习率')
    parser.add_argument('--batch_size', type=int, help='批次大小')
    parser.add_argument('--alpha', type=float, help='Dirichlet分布参数')
    
    # 其他选项
    parser.add_argument('--no_evaluation', action='store_true', help='跳过模型评估')
    parser.add_argument('--resume', type=str, help='从检查点恢复训练')
    
    args = parser.parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # 设置设备
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"使用设备: {device}")
    
    # 加载配置
    config = get_config(args.config)
    
    # 覆盖配置参数
    if args.data_dir:
        config['data_dir'] = args.data_dir
    if args.output_dir:
        config['output_dir'] = args.output_dir
    if args.num_rounds:
        config['num_rounds'] = args.num_rounds
    if args.num_clients:
        config['num_clients'] = args.num_clients
    if args.learning_rate:
        config['learning_rate'] = args.learning_rate
    if args.batch_size:
        config['batch_size'] = args.batch_size
    if args.alpha:
        config['alpha'] = args.alpha
    
    config['seed'] = args.seed
    config['device'] = str(device)
    
    # 打印配置
    print_config(config)
    
    # 创建训练器
    trainer = OrchestraTrainer(config, device)
    
    try:
        # 加载数据
        trainer.logger.info("加载CIFAR-10数据集...")
        client_data = create_federated_cifar10(
            data_dir=config['data_dir'],
            num_clients=config['num_clients'],
            alpha=config['alpha'],
            seed=config['seed']
        )
        
        # 开始训练
        if args.resume:
            trainer.logger.info(f"从检查点恢复训练: {args.resume}")
            # TODO: 实现从检查点恢复的逻辑
            raise NotImplementedError("从检查点恢复功能尚未实现")
        else:
            model = trainer.federated_training(client_data)
        
        # 保存最终模型
        trainer.save_final_model(model)
        
        # 评估模型
        if not args.no_evaluation:
            evaluation_results = trainer.evaluate_model(model, client_data)
            
            # 保存评估结果
            import json
            results_file = os.path.join(config['output_dir'], 'evaluation_results.json')
            
            # 转换numpy数组为列表以便JSON序列化
            serializable_results = {}
            for key, value in evaluation_results.items():
                if isinstance(value, dict):
                    serializable_results[key] = {}
                    for sub_key, sub_value in value.items():
                        if isinstance(sub_value, np.ndarray):
                            serializable_results[key][sub_key] = sub_value.tolist()
                        elif isinstance(sub_value, (np.integer, np.floating)):
                            serializable_results[key][sub_key] = float(sub_value)
                        else:
                            serializable_results[key][sub_key] = sub_value
                else:
                    serializable_results[key] = value
            
            with open(results_file, 'w') as f:
                json.dump(serializable_results, f, indent=2)
            
            trainer.logger.info(f"评估结果已保存: {results_file}")
        
        trainer.logger.info("训练和评估完成！")
        
    except Exception as e:
        trainer.logger.error(f"训练过程中发生错误: {e}")
        raise


if __name__ == '__main__':
    main()