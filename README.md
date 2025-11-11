# CV车辆识别 - 车辆品牌分类模型

## 项目概述

本项目实现了一个基于深度学习的车辆品牌识别模型，能够识别图像中的车辆并分类到10种预定义的品牌类别。

## 项目要求

- **基础能力**: 识别图像中是否有汽车
- **进阶能力**: 识别汽车的品牌（10类品牌）
- **品牌类别**: 
  1. 奥迪A4
  2. 本田雅阁
  3. 别克君越
  4. 大众迈腾
  5. 丰田花冠
  6. 丰田卡罗拉
  7. 丰田凯美瑞
  8. 福特福克斯
  9. 日产骐达
  10. 日产轩逸
- **准确率要求**: ≥ 80%
- **训练数据**: 使用 `re_id_1000_train.txt` 作为索引，从 `image` 文件夹加载对应图片
- **测试数据**: 使用 `re_id_1000_test.txt` 作为索引，从 `image` 文件夹加载对应图片
- **自动调参**: 准确率 < 80% 时自动重新调参训练（不使用测试数据训练）

## 技术栈

- **Python 3.8+**
- **PyTorch**: 深度学习框架
- **Torchvision**: 预训练模型和图像处理
- **ResNet/EfficientNet**: 卷积神经网络模型
- **数据增强**: 提升模型泛化能力
- **GPU加速**: 支持NVIDIA GPU（RTX 5080）

## 项目结构

```
cv-vehicle-classification/
├── main.py                 # 主程序入口
├── config.py              # 配置文件（数据路径指向百度网盘下载目录）
├── data_loader.py          # 数据加载和预处理
├── model_trainer.py        # 模型训练
├── hyperparameter_tuner.py # 超参数调优
├── predict.py              # 使用模型进行预测
├── test_imports.py         # 依赖测试脚本
├── requirements.txt        # 依赖包
├── README.md              # 项目说明
├── Dockerfile              # Docker镜像配置
├── docker-compose.yml      # Docker Compose配置
├── run.bat                 # Windows运行脚本
├── .gitignore             # Git忽略文件
├── models/                # 模型保存目录（自动创建）
└── results/               # 结果保存目录（自动创建）

注意：数据文件位于 C:\baidunetdiskdownload\CV-车辆检测\
```

## 数据路径配置

数据文件位于百度网盘下载目录，已在 `config.py` 中配置：
- 训练索引: `C:\baidunetdiskdownload\CV-车辆检测\re_id_1000_train.txt`
- 测试索引: `C:\baidunetdiskdownload\CV-车辆检测\re_id_1000_test.txt`
- 图像目录: `C:\baidunetdiskdownload\CV-车辆检测\image\`

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 方法一：直接运行（推荐）

#### Windows系统：
```bash
run.bat
```

### 方法二：手动运行

#### 1. 测试依赖

```bash
python test_imports.py
```

#### 2. 安装依赖（如果需要）

```bash
pip install -r requirements.txt
```

#### 3. 运行训练

```bash
python main.py
```

程序将自动：
1. 从百度网盘下载目录读取索引文件
2. 从 `image` 文件夹加载对应图片
3. 预处理图像（调整大小、数据增强等）
4. 训练模型并自动调参
5. 评估模型准确率
6. 如果准确率 < 80%，自动重新调参训练
7. 保存模型和结果到项目目录

### 使用训练好的模型进行预测

```bash
python predict.py "C:\baidunetdiskdownload\CV-车辆检测\image\1\License_1\xxx.jpg"
```

## 模型特点

### 1. 多种模型支持
- **ResNet18**: 快速训练，适合快速迭代
- **ResNet34**: 平衡性能和速度
- **ResNet50**: 更高准确率，训练时间较长

### 2. 自动超参数调优
- 自动搜索最佳学习率和批次大小
- 支持多种模型架构
- 自动选择最佳参数组合

### 3. 数据增强
- 随机裁剪
- 随机水平翻转
- 颜色抖动
- 提升模型泛化能力

### 4. 自动重训练机制
- 如果准确率 < 80%，自动尝试不同模型和参数
- 最多迭代10次，直到达到要求或达到最大迭代次数
- 保存最佳模型

## 评估指标

模型评估使用以下指标：
- **准确率 (Accuracy)**: 整体分类正确率
- **精确率 (Precision)**: 每个类别的精确率
- **召回率 (Recall)**: 每个类别的召回率
- **F1分数 (F1-Score)**: 精确率和召回率的调和平均
- **混淆矩阵 (Confusion Matrix)**: 各类别分类情况

## 输出示例

```
============================================================
模型评估报告
============================================================
准确率: 0.8523 (85.23%)

分类报告:
  奥迪A4:
    精确率: 0.8234
    召回率: 0.7891
    F1分数: 0.8059
    支持数: 128
...
```

## 注意事项

1. **GPU要求**: 建议使用GPU训练，CPU训练速度较慢
2. **内存要求**: 大规模数据可能需要较大内存（建议16GB+）
3. **训练时间**: 完整训练可能需要数小时，请耐心等待
4. **数据路径**: 确保数据文件在 `C:\baidunetdiskdownload\CV-车辆检测\` 目录下

## 故障排除

### 问题1: 图像文件找不到
- 检查索引文件中的路径格式
- 确保 `C:\baidunetdiskdownload\CV-车辆检测\image\` 文件夹存在且包含所有图片
- 检查路径分隔符（Windows使用`\`）

### 问题2: 准确率始终低于80%
- 检查数据质量，确保标签正确
- 尝试调整 `config.py` 中的参数（学习率、批次大小等）
- 增加训练轮数（epochs）
- 使用更大的模型（ResNet50）

### 问题3: GPU内存不足
- 减少 `BATCH_SIZE` 参数
- 使用较小的模型（ResNet18）
- 减少图像尺寸

### 问题4: CUDA不可用
- 检查NVIDIA驱动是否安装
- 检查PyTorch是否支持CUDA
- 如果GPU不可用，程序会自动使用CPU（速度较慢）

## 作者

AI Full Stack Development Course

## 许可证

MIT License

