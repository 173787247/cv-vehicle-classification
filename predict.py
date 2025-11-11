"""
使用训练好的模型进行预测
"""
import os
import sys
import torch
from PIL import Image
from torchvision import transforms
from model_trainer import VehicleClassifier
import config


def load_model(model_path: str, device: str = "cuda"):
    """
    加载训练好的模型
    
    Args:
        model_path: 模型文件路径
        device: 设备
        
    Returns:
        加载的模型
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    checkpoint = torch.load(model_path, map_location=device)
    
    model = VehicleClassifier(
        model_type=checkpoint.get('model_type', config.MODEL_TYPE),
        num_classes=checkpoint.get('num_classes', config.NUM_CLASSES),
        pretrained=False
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model = model.to(device)
    
    return model


def predict_image(image_path: str, model_path: str = None, device: str = "cuda"):
    """
    预测单张图像
    
    Args:
        image_path: 图像路径
        model_path: 模型文件路径（如果为None，则使用最新的模型）
        device: 设备
        
    Returns:
        预测的品牌名称和概率
    """
    # 如果没有指定模型路径，查找最新的模型
    if model_path is None:
        if not os.path.exists(config.MODEL_DIR):
            print(f"错误: 模型目录不存在: {config.MODEL_DIR}")
            print("请先运行 main.py 训练模型")
            return None
        
        model_files = [f for f in os.listdir(config.MODEL_DIR) if f.endswith('.pth')]
        if not model_files:
            print(f"错误: 模型目录中没有找到模型文件: {config.MODEL_DIR}")
            print("请先运行 main.py 训练模型")
            return None
        
        # 使用最新的模型（按文件名排序）
        model_files.sort(reverse=True)
        model_path = os.path.join(config.MODEL_DIR, model_files[0])
        print(f"使用模型: {model_files[0]}")
    
    # 加载模型
    try:
        model = load_model(model_path, device)
    except Exception as e:
        print(f"错误: 加载模型失败: {e}")
        return None
    
    # 加载和预处理图像
    try:
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"错误: 无法加载图像 {image_path}: {e}")
        return None
    
    # 图像变换
    transform = transforms.Compose([
        transforms.Resize(config.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # 预测
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        predicted_class = torch.argmax(probabilities).item()
        confidence = probabilities[predicted_class].item()
    
    brand_name = config.CAR_BRANDS[predicted_class]
    
    return {
        'brand': brand_name,
        'class_id': predicted_class,
        'confidence': confidence,
        'all_probabilities': {config.CAR_BRANDS[i]: probabilities[i].item() 
                             for i in range(len(config.CAR_BRANDS))}
    }


def main():
    """主函数"""
    if len(sys.argv) < 2:
        print("使用方法: python predict.py <图像路径> [模型路径]")
        print("\n示例:")
        print('  python predict.py "C:\\baidunetdiskdownload\\CV-车辆检测\\image\\1\\License_1\\xxx.jpg"')
        print('  python predict.py "image_path.jpg" models/vehicle_classifier_20240101_120000.pth')
        return
    
    image_path = sys.argv[1]
    model_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    # 设置设备
    if config.USE_GPU and torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    
    print("="*60)
    print("CV车辆识别 - 预测")
    print("="*60)
    print(f"\n图像路径: {image_path}")
    print(f"设备: {device}")
    
    result = predict_image(image_path, model_path, device)
    
    if result:
        print(f"\n预测结果:")
        print(f"  品牌: {result['brand']}")
        print(f"  类别ID: {result['class_id']}")
        print(f"  置信度: {result['confidence']:.4f} ({result['confidence']*100:.2f}%)")
        print(f"\n所有类别的概率:")
        for brand, prob in sorted(result['all_probabilities'].items(), 
                                 key=lambda x: x[1], reverse=True):
            print(f"  {brand}: {prob:.4f} ({prob*100:.2f}%)")
    else:
        print("\n预测失败")


if __name__ == "__main__":
    main()

