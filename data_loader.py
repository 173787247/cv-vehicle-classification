"""
数据加载和预处理模块
"""
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from typing import List, Tuple, Optional
import config


class VehicleDataset(Dataset):
    """车辆图像数据集"""
    
    def __init__(self, image_paths: List[str], labels: List[int], 
                 transform: Optional[transforms.Compose] = None,
                 is_training: bool = True):
        """
        初始化数据集
        
        Args:
            image_paths: 图像路径列表
            labels: 标签列表（0-9）
            transform: 图像变换
            is_training: 是否为训练集
        """
        self.image_paths = image_paths
        self.labels = labels
        self.is_training = is_training
        
        # 如果没有提供transform，使用默认的
        if transform is None:
            if is_training and config.USE_DATA_AUGMENTATION:
                # 训练时使用数据增强
                self.transform = transforms.Compose([
                    transforms.Resize((256, 256)),
                    transforms.RandomCrop(config.IMAGE_SIZE),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                       std=[0.229, 0.224, 0.225])
                ])
            else:
                # 测试时只做标准化
                self.transform = transforms.Compose([
                    transforms.Resize(config.IMAGE_SIZE),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                       std=[0.229, 0.224, 0.225])
                ])
        else:
            self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """
        获取单个样本
        
        Args:
            idx: 样本索引
            
        Returns:
            image: 图像张量
            label: 标签
        """
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # 加载图像
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"警告: 无法加载图像 {image_path}: {e}")
            # 返回黑色图像
            image = Image.new('RGB', config.IMAGE_SIZE, (0, 0, 0))
        
        # 应用变换
        if self.transform:
            image = self.transform(image)
        
        return image, label


def parse_index_file(index_file: str, image_dir: str) -> Tuple[List[str], List[int]]:
    """
    解析索引文件，获取图像路径和标签
    
    Args:
        index_file: 索引文件路径
        image_dir: 图像目录路径
        
    Returns:
        image_paths: 图像路径列表
        labels: 标签列表（0-9，从1-10转换为0-9）
    """
    image_paths = []
    labels = []
    
    if not os.path.exists(index_file):
        raise FileNotFoundError(f"索引文件不存在: {index_file}")
    
    with open(index_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # 解析路径：格式为 "1\License_1\xxx.jpg"
            # 第一个数字是类别（1-10），需要转换为（0-9）
            parts = line.split('\\')
            if len(parts) >= 1:
                # 提取类别（第一个数字）
                category = int(parts[0]) - 1  # 转换为0-9
                
                # 构建完整路径
                # 将Windows路径分隔符统一处理
                relative_path = line.replace('\\', os.sep)
                full_path = os.path.join(image_dir, relative_path)
                
                # 如果文件不存在，尝试不同的路径格式
                if not os.path.exists(full_path):
                    # 尝试直接使用原始路径
                    alt_path = os.path.join(image_dir, line)
                    if os.path.exists(alt_path):
                        full_path = alt_path
                
                # 检查文件是否存在
                if os.path.exists(full_path):
                    image_paths.append(full_path)
                    labels.append(category)
                else:
                    print(f"警告: 图像文件不存在: {full_path}")
    
    print(f"成功加载 {len(image_paths)} 条数据")
    print(f"标签分布: {np.bincount(labels)}")
    
    return image_paths, labels


def create_data_loaders(train_index_file: str, test_index_file: str, 
                       image_dir: str, batch_size: int = None,
                       num_workers: int = None) -> Tuple[DataLoader, DataLoader]:
    """
    创建训练和测试数据加载器
    
    Args:
        train_index_file: 训练索引文件路径
        test_index_file: 测试索引文件路径
        image_dir: 图像目录路径
        batch_size: 批次大小
        num_workers: 数据加载线程数
        
    Returns:
        train_loader: 训练数据加载器
        test_loader: 测试数据加载器
    """
    if batch_size is None:
        batch_size = config.BATCH_SIZE
    if num_workers is None:
        num_workers = config.NUM_WORKERS
    
    # 解析索引文件
    print("加载训练数据...")
    train_paths, train_labels = parse_index_file(train_index_file, image_dir)
    
    print("加载测试数据...")
    test_paths, test_labels = parse_index_file(test_index_file, image_dir)
    
    # 创建数据集
    train_dataset = VehicleDataset(train_paths, train_labels, is_training=True)
    test_dataset = VehicleDataset(test_paths, test_labels, is_training=False)
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    return train_loader, test_loader

