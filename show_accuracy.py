"""
快速查看模型准确率
"""
import json
import os
from pathlib import Path

results_dir = Path("results")
if results_dir.exists():
    json_files = list(results_dir.glob("results_*.json"))
    if json_files:
        # 获取最新的结果文件
        latest_file = max(json_files, key=lambda x: x.stat().st_mtime)
        
        with open(latest_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print("="*60)
        print("CV车辆识别 - 模型准确率")
        print("="*60)
        print(f"结果文件: {latest_file.name}")
        print(f"时间戳: {data['timestamp']}")
        print()
        print(f"准确率: {data['accuracy']:.4f} ({data['accuracy']*100:.2f}%)")
        print(f"要求准确率: {data['min_accuracy_required']:.2%}")
        print(f"是否达标: {'[OK] 是' if data['meets_requirement'] else '[NO] 否'}")
        print()
        print(f"模型类型: {data['model_type']}")
        print(f"训练样本: {data.get('training_samples', 'N/A')}")
        print(f"测试样本: {data.get('testing_samples', 'N/A')}")
        print()
        print("="*60)
    else:
        print("未找到结果文件")
else:
    print("results目录不存在")

