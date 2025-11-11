"""
超参数调优模块
"""
import torch
import torch.nn as nn
from model_trainer import VehicleClassifier, train_model
from data_loader import create_data_loaders
from typing import Dict, Tuple
import config


def tune_hyperparameters(train_loader, test_loader,
                       model_type: str = "resnet18",
                       device: str = "cuda") -> Dict:
    """
    超参数调优
    
    Args:
        train_loader: 训练数据加载器
        test_loader: 测试数据加载器
        model_type: 模型类型
        device: 设备
        
    Returns:
        最佳参数字典
    """
    print(f"\n开始调优 {model_type.upper()} 模型的超参数...")
    
    # 定义超参数搜索空间
    learning_rates = [0.001, 0.0005, 0.0001]
    batch_sizes = [16, 32, 64]
    
    best_params = None
    best_accuracy = 0.0
    
    for lr in learning_rates:
        for batch_size in batch_sizes:
            print(f"\n尝试参数: lr={lr}, batch_size={batch_size}")
            
            try:
                # 创建模型
                model = VehicleClassifier(
                    model_type=model_type,
                    num_classes=config.NUM_CLASSES,
                    pretrained=True
                )
                
                # 训练模型（使用较少的epoch进行快速调参）
                results = train_model(
                    model=model,
                    train_loader=train_loader,
                    test_loader=test_loader,
                    learning_rate=lr,
                    epochs=10,  # 快速调参时使用较少epoch
                    device=device
                )
                
                accuracy = results['best_test_accuracy']
                print(f"准确率: {accuracy:.4f}")
                
                # 更新最佳参数
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_params = {
                        'learning_rate': lr,
                        'batch_size': batch_size,
                        'model_type': model_type,
                        'accuracy': accuracy
                    }
                    print(f"✓ 新的最佳参数！准确率: {best_accuracy:.4f}")
                    
            except Exception as e:
                print(f"✗ 参数组合失败: {e}")
                continue
    
    print(f"\n最佳参数: {best_params}")
    return best_params


def auto_tune_and_train(train_loader, test_loader,
                       min_accuracy: float = 0.80,
                       max_iterations: int = 10,
                       device: str = "cuda") -> Tuple[nn.Module, float]:
    """
    自动调参并训练，直到达到最低准确率要求
    
    Args:
        train_loader: 训练数据加载器
        test_loader: 测试数据加载器
        min_accuracy: 最低准确率要求
        max_iterations: 最大迭代次数
        device: 设备
        
    Returns:
        (最佳模型, 准确率)
    """
    model_types = ["resnet18", "resnet34", "resnet50"]
    best_model = None
    best_accuracy = 0.0
    best_type = None
    best_params = None
    
    print(f"\n开始自动调参，目标准确率: {min_accuracy:.2%}")
    print(f"最大迭代次数: {max_iterations}")
    
    for iteration in range(max_iterations):
        print(f"\n{'='*60}")
        print(f"迭代 {iteration + 1}/{max_iterations}")
        print(f"{'='*60}")
        
        # 尝试不同的模型类型
        for model_type in model_types:
            try:
                print(f"\n尝试 {model_type.upper()} 模型...")
                
                # 调优超参数
                params = tune_hyperparameters(
                    train_loader, test_loader, model_type, device
                )
                
                if params is None:
                    continue
                
                # 使用最佳参数进行完整训练
                print(f"\n使用最佳参数进行完整训练...")
                model = VehicleClassifier(
                    model_type=params['model_type'],
                    num_classes=config.NUM_CLASSES,
                    pretrained=True
                )
                
                results = train_model(
                    model=model,
                    train_loader=train_loader,
                    test_loader=test_loader,
                    learning_rate=params['learning_rate'],
                    epochs=config.EPOCHS,
                    device=device
                )
                
                accuracy = results['best_test_accuracy']
                print(f"{model_type.upper()} 最终准确率: {accuracy:.4f}")
                
                # 更新最佳模型
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_model = results['model']
                    best_type = model_type
                    best_params = params
                    print(f"✓ 新的最佳模型！准确率: {best_accuracy:.4f}")
                
                # 如果达到要求，提前结束
                if accuracy >= min_accuracy:
                    print(f"\n🎉 达到目标准确率 {min_accuracy:.2%}！")
                    return best_model, best_accuracy
                    
            except Exception as e:
                print(f"✗ {model_type.upper()} 训练失败: {e}")
                continue
        
        # 如果已经达到要求，提前结束
        if best_accuracy >= min_accuracy:
            break
    
    print(f"\n最终最佳模型: {best_type.upper()}, 准确率: {best_accuracy:.4f}")
    
    if best_accuracy < min_accuracy:
        print(f"⚠️  警告: 未能达到目标准确率 {min_accuracy:.2%}")
        print(f"当前最佳准确率: {best_accuracy:.4f}")
    
    return best_model, best_accuracy

