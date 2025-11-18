"""
模型训练模块
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from torchvision import models
import numpy as np
from typing import Dict, Tuple, Optional
import config


class VehicleClassifier(nn.Module):
    """车辆分类模型"""
    
    def __init__(self, model_type: str = "resnet18", num_classes: int = 10, 
                 pretrained: bool = True):
        """
        初始化模型
        
        Args:
            model_type: 模型类型（resnet18, resnet34, resnet50, efficientnet_b0等）
            num_classes: 分类数量
            pretrained: 是否使用预训练权重
        """
        super(VehicleClassifier, self).__init__()
        
        self.model_type = model_type
        self.num_classes = num_classes
        
        # 根据模型类型创建backbone
        if model_type.startswith("resnet"):
            if model_type == "resnet18":
                backbone = models.resnet18(pretrained=pretrained)
                num_features = 512
            elif model_type == "resnet34":
                backbone = models.resnet34(pretrained=pretrained)
                num_features = 512
            elif model_type == "resnet50":
                backbone = models.resnet50(pretrained=pretrained)
                num_features = 2048
            else:
                raise ValueError(f"不支持的ResNet类型: {model_type}")
            
            # 替换最后的全连接层
            backbone.fc = nn.Linear(num_features, num_classes)
            self.model = backbone
            
        elif model_type.startswith("efficientnet"):
            try:
                from efficientnet_pytorch import EfficientNet
                if model_type == "efficientnet_b0":
                    self.model = EfficientNet.from_pretrained('efficientnet-b0', 
                                                             num_classes=num_classes)
                elif model_type == "efficientnet_b3":
                    self.model = EfficientNet.from_pretrained('efficientnet-b3', 
                                                             num_classes=num_classes)
                else:
                    raise ValueError(f"不支持的EfficientNet类型: {model_type}")
            except ImportError:
                print("警告: efficientnet_pytorch未安装，使用ResNet18代替")
                backbone = models.resnet18(pretrained=pretrained)
                backbone.fc = nn.Linear(512, num_classes)
                self.model = backbone
                self.model_type = "resnet18"
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")
    
    def forward(self, x):
        return self.model(x)


def train_epoch(model: nn.Module, train_loader, criterion, optimizer, device):
    """训练一个epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        
        # 前向传播
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        # 统计
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = correct / total
    
    return epoch_loss, epoch_acc


def evaluate(model: nn.Module, test_loader, criterion, device):
    """评估模型"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(test_loader)
    epoch_acc = correct / total
    
    return epoch_loss, epoch_acc, all_preds, all_labels


def train_model(model: nn.Module, train_loader, test_loader,
                learning_rate: float = 0.001, epochs: int = 50,
                device: str = "cuda") -> Dict:
    """
    训练模型
    
    Args:
        model: 模型
        train_loader: 训练数据加载器
        test_loader: 测试数据加载器
        learning_rate: 学习率
        epochs: 训练轮数
        device: 设备（cuda或cpu）
        
    Returns:
        训练结果字典
    """
    model = model.to(device)
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    # 训练历史
    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []
    
    best_test_acc = 0.0
    best_model_state = None
    patience_counter = 0
    
    print(f"\n开始训练，共 {epochs} 个epoch...")
    print(f"设备: {device}")
    print(f"学习率: {learning_rate}")
    print(f"批次大小: {config.BATCH_SIZE}")
    print("-" * 60)
    
    for epoch in range(epochs):
        # 训练
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # 评估
        test_loss, test_acc, _, _ = evaluate(model, test_loader, criterion, device)
        
        # 更新学习率
        scheduler.step(test_loss)
        
        # 记录历史
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)
        
        # 保存最佳模型
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        # 打印进度
        print(f"Epoch [{epoch+1}/{epochs}]")
        print(f"  训练 - Loss: {train_loss:.4f}, Acc: {train_acc:.4f} ({train_acc*100:.2f}%)")
        print(f"  测试 - Loss: {test_loss:.4f}, Acc: {test_acc:.4f} ({test_acc*100:.2f}%)")
        print(f"  最佳测试准确率: {best_test_acc:.4f} ({best_test_acc*100:.2f}%)")
        
        # 早停
        if patience_counter >= config.EARLY_STOPPING_PATIENCE:
            print(f"\n早停触发（{config.EARLY_STOPPING_PATIENCE}个epoch无改善）")
            break
        
        print("-" * 60)
    
    # 加载最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # 最终评估
    final_test_loss, final_test_acc, all_preds, all_labels = evaluate(
        model, test_loader, criterion, device
    )
    
    return {
        'model': model,
        'best_test_accuracy': best_test_acc,
        'final_test_accuracy': final_test_acc,
        'train_losses': train_losses,
        'train_accs': train_accs,
        'test_losses': test_losses,
        'test_accs': test_accs,
        'predictions': all_preds,
        'labels': all_labels
    }

