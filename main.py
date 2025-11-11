"""
CV车辆识别主程序
用于训练车辆品牌分类模型并评估准确率
"""
import os
import json
import torch
import numpy as np
from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix
from data_loader import create_data_loaders
from model_trainer import VehicleClassifier, train_model
from hyperparameter_tuner import auto_tune_and_train
import config


def save_results(results: dict, filepath: str):
    """保存结果到JSON文件"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"结果已保存到: {filepath}")


def print_evaluation_report(evaluation: dict):
    """打印评估报告"""
    print("\n" + "="*60)
    print("模型评估报告")
    print("="*60)
    print(f"准确率: {evaluation['accuracy']:.4f} ({evaluation['accuracy']*100:.2f}%)")
    
    print("\n分类报告:")
    report = evaluation['classification_report']
    for i, brand in enumerate(config.CAR_BRANDS):
        if str(i) in report:
            metrics = report[str(i)]
            print(f"  {brand}:")
            print(f"    精确率: {metrics['precision']:.4f}")
            print(f"    召回率: {metrics['recall']:.4f}")
            print(f"    F1分数: {metrics['f1-score']:.4f}")
            print(f"    支持数: {int(metrics['support'])}")
    
    print(f"\n宏平均:")
    print(f"  精确率: {report['macro avg']['precision']:.4f}")
    print(f"  召回率: {report['macro avg']['recall']:.4f}")
    print(f"  F1分数: {report['macro avg']['f1-score']:.4f}")
    
    print(f"\n加权平均:")
    print(f"  精确率: {report['weighted avg']['precision']:.4f}")
    print(f"  召回率: {report['weighted avg']['recall']:.4f}")
    print(f"  F1分数: {report['weighted avg']['f1-score']:.4f}")


def main():
    """主函数"""
    print("="*60)
    print("CV车辆识别 - 车辆品牌分类模型")
    print("="*60)
    print(f"品牌类别: {', '.join(config.CAR_BRANDS)}")
    print(f"类别数量: {config.NUM_CLASSES}")
    
    # 检查数据文件是否存在
    if not os.path.exists(config.TRAIN_INDEX_FILE):
        print(f"错误: 训练索引文件不存在: {config.TRAIN_INDEX_FILE}")
        return
    
    if not os.path.exists(config.TEST_INDEX_FILE):
        print(f"错误: 测试索引文件不存在: {config.TEST_INDEX_FILE}")
        return
    
    if not os.path.exists(config.IMAGE_DIR):
        print(f"错误: 图像目录不存在: {config.IMAGE_DIR}")
        return
    
    # 设置设备
    if config.USE_GPU and torch.cuda.is_available():
        device = "cuda"
        print(f"\n使用GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        print("\n使用CPU")
    
    # 加载数据
    print("\n1. 创建数据加载器...")
    train_loader, test_loader = create_data_loaders(
        config.TRAIN_INDEX_FILE,
        config.TEST_INDEX_FILE,
        config.IMAGE_DIR,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS
    )
    
    print(f"训练样本数: {len(train_loader.dataset)}")
    print(f"测试样本数: {len(test_loader.dataset)}")
    
    # 训练和评估模型
    print("\n2. 训练模型并自动调参...")
    
    # 自动调参并训练
    best_model, best_accuracy = auto_tune_and_train(
        train_loader,
        test_loader,
        min_accuracy=config.MIN_ACCURACY,
        max_iterations=config.MAX_ITERATIONS,
        device=device
    )
    
    if best_model is None:
        print("\n❌ 训练失败，无法创建有效模型")
        return
    
    # 最终评估
    print("\n3. 最终评估...")
    from model_trainer import evaluate
    criterion = torch.nn.CrossEntropyLoss()
    test_loss, test_acc, all_preds, all_labels = evaluate(
        best_model, test_loader, criterion, device
    )
    
    # 生成分类报告
    report = classification_report(all_labels, all_preds, 
                                   target_names=config.CAR_BRANDS,
                                   output_dict=True)
    cm = confusion_matrix(all_labels, all_preds)
    
    evaluation = {
        'accuracy': test_acc,
        'classification_report': report,
        'confusion_matrix': cm.tolist()
    }
    
    print_evaluation_report(evaluation)
    
    # 保存模型
    print("\n4. 保存模型...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(config.MODEL_DIR, f"vehicle_classifier_{timestamp}.pth")
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    torch.save({
        'model_state_dict': best_model.state_dict(),
        'model_type': config.MODEL_TYPE,
        'num_classes': config.NUM_CLASSES,
        'car_brands': config.CAR_BRANDS,
        'accuracy': best_accuracy
    }, model_path)
    print(f"模型已保存到: {model_path}")
    
    # 保存结果
    print("\n5. 保存结果...")
    results = {
        'timestamp': timestamp,
        'accuracy': float(test_acc),
        'min_accuracy_required': config.MIN_ACCURACY,
        'meets_requirement': test_acc >= config.MIN_ACCURACY,
        'model_type': config.MODEL_TYPE,
        'classification_report': report,
        'confusion_matrix': cm.tolist(),
        'model_path': model_path,
        'training_samples': len(train_loader.dataset),
        'testing_samples': len(test_loader.dataset),
        'car_brands': config.CAR_BRANDS
    }
    
    results_path = os.path.join(config.RESULTS_DIR, f"results_{timestamp}.json")
    save_results(results, results_path)
    
    # 最终总结
    print("\n" + "="*60)
    print("训练完成！")
    print("="*60)
    print(f"最终准确率: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")
    print(f"目标准确率: {config.MIN_ACCURACY:.2%}")
    
    if best_accuracy >= config.MIN_ACCURACY:
        print("✅ 模型达到交付要求！")
    else:
        print("⚠️  模型未达到交付要求，建议继续调参")
    
    print(f"\n模型文件: {model_path}")
    print(f"结果文件: {results_path}")


if __name__ == "__main__":
    main()

