"""
CV车辆识别配置文件
"""
import os

# 数据路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 数据文件在百度网盘下载目录
DATA_DIR = r"C:\baidunetdiskdownload\CV-车辆检测"
IMAGE_DIR = os.path.join(DATA_DIR, "image")

# 索引文件路径
TRAIN_INDEX_FILE = os.path.join(DATA_DIR, "re_id_1000_train.txt")
TEST_INDEX_FILE = os.path.join(DATA_DIR, "re_id_1000_test.txt")

# 模型保存路径（保存在项目目录）
MODEL_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# 模型参数
MIN_ACCURACY = 0.80  # 最低准确率要求
MAX_ITERATIONS = 10  # 最大调参迭代次数

# 车辆品牌类别（10类）
CAR_BRANDS = [
    "奥迪A4",
    "本田雅阁",
    "别克君越",
    "大众迈腾",
    "丰田花冠",
    "丰田卡罗拉",
    "丰田凯美瑞",
    "福特福克斯",
    "日产骐达",
    "日产轩逸"
]

NUM_CLASSES = len(CAR_BRANDS)  # 10个类别

# 图像预处理参数
IMAGE_SIZE = (224, 224)  # 输入图像尺寸
BATCH_SIZE = 32  # 批次大小
NUM_WORKERS = 4  # 数据加载线程数

# 训练参数
LEARNING_RATE = 0.001  # 初始学习率
EPOCHS = 50  # 训练轮数
EARLY_STOPPING_PATIENCE = 5  # 早停耐心值

# 数据增强参数
USE_DATA_AUGMENTATION = True  # 是否使用数据增强

# 模型类型（可选：resnet18, resnet34, resnet50, efficientnet_b0, efficientnet_b3）
MODEL_TYPE = "resnet18"  # 默认使用ResNet18（速度快）

# GPU设置
USE_GPU = True  # 是否使用GPU
CUDA_VISIBLE_DEVICES = "0"  # GPU设备ID

